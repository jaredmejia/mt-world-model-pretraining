from typing import Callable, Optional

import gym
import numpy as np
import torch

from tensordict.tensordict import make_tensordict
from torchrl.collectors.utils import split_trajectories
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import Sampler
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.data.replay_buffers.writers import Writer
from torchrl.envs import CenterCrop, Compose, ToTensorImage, TransformedEnv
from torchrl.envs.libs.gym import GymWrapper

from .dataset_utils import qlearning_kitchen_dataset
from .transforms import KitchenFilterState


class KitchenExperienceReplay(TensorDictReplayBuffer):
    def __init__(
            self,
            env_name: str,
            observation_type: str = "default",
            batch_size: int = 256,
            sampler: Optional[Sampler] = None,
            writer: Optional[Writer] = None,
            collate_fn: Optional[Callable] = None,
            pin_memory: bool = False,
            prefetch: Optional[int] = None,
            transform: Optional["Transform"] = None,  # noqa-F821
            split_trajs: bool = False,
            use_timeout_as_done: bool = True,
    ):
        self.use_timeout_as_done = use_timeout_as_done  
        dataset = self._get_dataset_direct(env_name, observation_type)

        dataset["next", "observation"][dataset["next", "done"].squeeze()] = 0
        if "image" in observation_type:
            dataset["next", "pixels"][dataset["next", "done"].squeeze()] = 0

        if split_trajs:
            dataset = split_trajectories(dataset)

        storage = LazyMemmapStorage(dataset.shape[0])
        super().__init__(
            batch_size=batch_size,
            storage=storage,
            sampler=sampler,
            writer=writer,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            prefetch=prefetch,
            transform=transform,
        )
        self.extend(dataset)

    def _get_dataset_direct(self, env_name, observation_type):
        if 'image' in observation_type:
            from_pixels = True
            pixel_keys =["pixels", ("next", "pixels")]
            state_keys = ["observations", ("next", "observations")]
            env_transforms = Compose(
                ToTensorImage(in_keys=pixel_keys, out_keys=pixel_keys), CenterCrop(96, in_keys=pixel_keys, out_keys=pixel_keys),
                KitchenFilterState(in_keys=state_keys, out_keys=state_keys),
            )
        else:
            from_pixels = False
            env_transforms = None
        
        env = gym.make(env_name, render_imgs=from_pixels)
        env = GymWrapper(env, from_pixels=from_pixels)
        env = TransformedEnv(env, env_transforms)

        dataset = qlearning_kitchen_dataset(env, observation_type=observation_type)

        if observation_type == "image_joints":
            data_dict = {k: torch.from_numpy(item) for k, item in dataset.items() if isinstance(item, np.ndarray)}
            data_dict['observations'] = dataset['observations']['vector']
            data_dict['next_observations'] = dataset['next_observations']['vector']
            data_dict['pixels'] = dataset['observations']['image']
            data_dict['next_pixels'] = dataset['next_observations']['image']
        else:
            data_dict = {k: torch.from_numpy(item) for k, item in dataset.items() if isinstance(item, np.ndarray)}

        dataset = make_tensordict(data_dict)

        # rename keys to match torchrl conventions
        dataset.rename_key("observations", "observation")
        dataset.set("next", dataset.select())
        dataset.rename_key("next_observations", ("next", "observation"))
        dataset.rename_key("terminals", "terminal")
        if "timeouts" in dataset.keys():
            dataset.rename_key("timeouts", "timeout")
        if self.use_timeout_as_done:
            dataset.set(
                "done",
                dataset.get("terminal")
                | dataset.get("timeout", torch.zeros((), dtype=torch.bool)),
            )
        else:
            dataset.set("done", dataset.get("terminal"))
        dataset.rename_key("rewards", "reward")
        dataset.rename_key("actions", "action")

        if observation_type == "image_joints":
            dataset.rename_key("next_pixels", ("next", "pixels"))
        
        if env_transforms is not None:
            dataset = env_transforms(dataset)

        # checking dtypes
        for key, spec in env.observation_spec.items(True, True):
            dataset[key] = dataset[key].to(spec.dtype)
            dataset["next", key] = dataset["next", key].to(spec.dtype)
        for key, spec in env.input_spec.items(True, True):
            dataset[key] = dataset[key].to(spec.dtype)
        dataset["reward"] = dataset["reward"].to(env.reward_spec.dtype)
        dataset["done"] = dataset["done"].bool()

        dataset["done"] = dataset["done"].unsqueeze(-1)
        # dataset.rename_key("next_observations", "next/observation")
        dataset["reward"] = dataset["reward"].unsqueeze(-1)
        dataset["next"].update(
            dataset.select("reward", "done", "terminal", "timeout", strict=False)
        )
        dataset = (
            dataset.clone()
        )  # make sure that all tensors have a different data_ptr
        self._shift_reward_done(dataset)
        self.specs = env.specs.clone()
        return dataset

    def _shift_reward_done(self, dataset):
        dataset["reward"] = dataset["reward"].clone()
        dataset["done"] = dataset["done"].clone()
        dataset["reward"][1:] = dataset["reward"][:-1].clone()
        dataset["done"][1:] = dataset["done"][:-1].clone()
        dataset["reward"][0] = 0
        dataset["done"][0] = 0
        

class KitchenSubTrajectoryReplay(KitchenExperienceReplay):
    def __init__(
            self,
            env_name: str,
            observation_type: str = 'image_joints',
            batch_size: int = 256,
            batch_length: int = 50,
            transform: Optional[Callable] = None,
            split_trajs: bool = False,
            use_timeout_as_done: bool = True
    ):
        self.use_timeout_as_done = use_timeout_as_done  
        dataset = self._get_dataset_direct(env_name, observation_type)

        dataset["next", "observation"][dataset["next", "done"].squeeze()] = 0
        if "image" in observation_type:
            dataset["next", "pixels"][dataset["next", "done"].squeeze()] = 0

        if split_trajs:
            dataset = split_trajectories(dataset)

        self.valid_indices = self._get_valid_indices(dataset, batch_length)
        self.num_sub_trajs = len(self.valid_indices)
        self.batch_length = batch_length
        self.batch_size = batch_size
        self.dataset = dataset

        if transform is None:
            transform = Compose()
        elif not isinstance(transform, Compose):
            transform = Compose(transform)
        transform.eval()
        self._transform = transform
    
    def sample(self, batch_size: Optional[int] = None):
        if batch_size is None:
            batch_size = self.batch_size
        batch = self._get_sub_traj_batch(self.dataset, self.valid_indices, self.batch_length, batch_size)
        return self._transform(batch)
    

    def _get_valid_indices(self, dataset, batch_length):
        dataset_size = dataset['done'].shape[0]
        sub_traj_end_indices = torch.cat((torch.where(dataset['done'])[0], torch.tensor([dataset_size - 1])))
        max_traj_start_indices = sub_traj_end_indices - batch_length

        valid_indices = []
        prev_end_idx = 0
        for max_start, end_idx in zip(max_traj_start_indices, sub_traj_end_indices):
            valid_indices.append(torch.arange(prev_end_idx, max_start + 1))
            prev_end_idx = end_idx + 1

        valid_indices = torch.cat(valid_indices)
        return valid_indices
    

    def _get_sub_traj_batch(self, dataset, valid_indices, batch_length, batch_size):

        # first sample batch_size number of valid starting indices
        start_idxs = valid_indices[torch.randint(0, len(valid_indices), (batch_size,))]

        assert len(start_idxs) == batch_size

        batch = []
        for start_idx in start_idxs:
            batch.append(dataset[start_idx:start_idx + batch_length])
        batch_td = torch.stack(batch)
        
        return batch_td
            