import collections
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from tqdm import tqdm

import wrappers

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)

def total_to_instant_rewards(total_rewards, terminal_idxs, num_trajs):
    """Converts total rewards to instantaneous rewards for the Kitchen dataset.

    Args:
        total_rewards: array of total rewards at each timestep.
        terminal_idxs: array of terminal indices.
        num_trajs: number of trajectories.

    Returns:
        array of rewards.
    """
    reward_arrs = []
    terminal_idxs = np.concatenate([np.array([-1]), terminal_idxs, np.array([len(total_rewards)])])

    for i in range(1, num_trajs + 1):
        curr_returns = total_rewards[
            terminal_idxs[i - 1] + 1 : terminal_idxs[i] + 1
        ]
        change_idx = np.where(curr_returns[:-1] != curr_returns[1:])[0]
        new_rewards = np.zeros_like(curr_returns)
        new_rewards[change_idx + 1] = 1.0
        reward_arrs.append(new_rewards)

    reward_arrs = np.concatenate(reward_arrs)
    assert (
        reward_arrs.shape == total_rewards.shape
    ), f"{reward_arrs.shape} != {total_rewards.shape}"

    return reward_arrs



def qlearning_offline_dataset(env, dataset, env_name, terminate_on_end=False, observation_type="default"):

    if observation_type == "default":
        obs_ = []
        next_obs_ = []
    elif observation_type == "image_joints":
        assert "images" in dataset, "Images not found in dataset"
  
        images_ = []
        next_images_ = []
        joints_ = []
        next_joints_ = []

        if 'kitchen' in env_name:
            obs_dim = 9
        else: # metaworld
            obs_dim = 4

    else:
        raise ValueError("Invalid observation type")
    
    if 'kitchen' in env_name:
        max_steps = 280
    else:
        max_steps = 500

    N = dataset['rewards'].shape[0]
    action_ = []
    total_reward_ = []
    done_ = []

    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    # for i in range(N-1):
    for i in range(100):
        if observation_type == "default":
            obs = dataset["observations"][i].astype(np.float32)
            new_obs = dataset["observations"][i+1].astype(np.float32)
        elif observation_type == "image_joints":
            image = dataset["images"][i]

            next_image = dataset["images"][i+1]

            joint = dataset["observations"][i][:obs_dim].astype(np.float32)
            next_joint = dataset["observations"][i+1][:obs_dim].astype(np.float32)


        action = dataset['actions'][i].astype(np.float32)
        total_reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == max_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        if observation_type == "default":
            obs_.append(obs)
            next_obs_.append(new_obs)
        else:
            images_.append(image)
            next_images_.append(next_image)
            joints_.append(joint)
            next_joints_.append(next_joint)

        action_.append(action)
        total_reward_.append(total_reward)
        done_.append(done_bool)
        episode_step += 1

    if observation_type == "default":
        observations = np.array(obs_)
        next_observations = np.array(next_obs_)
    else:
        images = np.array(images_)
        next_images = np.array(next_images_)
        joints = np.array(joints_)
        next_joints = np.array(next_joints_)

        observations = {"image": images, "vector": joints}
        next_observations = {"image": next_images, "vector": next_joints}

    actions = np.array(action_)
    terminals = np.array(done_)

    # get true sparse rewards from total rewards    
    total_reward = np.array(total_reward_)
    terminal_idxs = np.where(terminals)[0]
    num_trajs = len(terminal_idxs) + 1
    rewards = total_to_instant_rewards(
        total_reward, terminal_idxs, num_trajs
    )

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "next_observations": next_observations,
    }


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])
    

class KitchenDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5, 
                 observation_type: str = "default",):
        self.observation_type = observation_type
        dataset = qlearning_kitchen_dataset(env, observation_type=observation_type)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'],
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'],
                         size=len(dataset['rewards']))
        
    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)

        if self.observation_type == "default":
            return Batch(observations=self.observations[indx],
                         actions=self.actions[indx],
                         rewards=self.rewards[indx],
                         masks=self.masks[indx],
                         next_observations=self.next_observations[indx])
        else:
            # multimodal observations
            return Batch(observations={"image": self.observations["image"][indx],
                                       "vector": self.observations["vector"][indx]},
                         actions=self.actions[indx],
                         rewards=self.rewards[indx],
                         masks=self.masks[indx],
                         next_observations={"image": self.next_observations["image"][indx],
                                            "vector": self.next_observations["vector"][indx]})


class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        import d4rl
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))


class ReplayBuffer(Dataset):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int):

        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity, ), dtype=np.float32)
        masks = np.empty((capacity, ), dtype=np.float32)
        dones_float = np.empty((capacity, ), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples: Optional[int]):
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[
            indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

def make_env_and_dataset(env_name: str,
                         seed: int, 
                         observation_type: str = "default") -> Tuple[gym.Env, Dataset]:

    if 'kitchen' in env_name:
        render_imgs = 'image' in observation_type
        env = gym.make(env_name, render_imgs=render_imgs)
        env = wrappers.FrankaKitchen(env, obs_type=observation_type)
    else:
        env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    if 'kitchen' in env_name:
        dataset = KitchenDataset(env, observation_type=observation_type)
    else:
        dataset = D4RLDataset(env)

    return env, dataset
