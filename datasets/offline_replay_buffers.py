from typing import Callable, Optional

import h5py
import numpy as np
import os
import torch

import torchsnapshot
from tensordict import TensorDict
from tensordict.tensordict import make_tensordict
from torchrl.collectors.utils import split_trajectories
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import Sampler
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.data.replay_buffers.writers import Writer
from torchrl.envs import Compose

from .dataset_utils import qlearning_offline_dataset, multitask_qlearning_offline_dataset
from .metadata import METAWORLD_SUCCESS_DATA_PATHS, METAWORLD_IDS
from .env_makers import env_maker


class OfflineExperienceReplay(TensorDictReplayBuffer):
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
            base_transform: Optional["Transform"] = None,  # noqa-F821
            split_trajs: bool = False,
            use_timeout_as_done: bool = True,
            multitask: bool = False,
    ):
        self.use_timeout_as_done = use_timeout_as_done  
        self.base_transform = base_transform

        dataset = self._get_dataset_direct(env_name, observation_type, multitask=multitask)

        if observation_type != "image":
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
            transform=base_transform,
        )
        self.extend(dataset)

    def _get_dataset_direct(self, env_name, observation_type, multitask=False):

        from_pixels = "image" in observation_type
        env = env_maker(env_name, from_pixels=from_pixels, env_transforms=self.base_transform)

        if multitask:
            dataset = multitask_qlearning_offline_dataset(METAWORLD_SUCCESS_DATA_PATHS, METAWORLD_IDS, observation_type=observation_type)
            
        else:
            if 'kitchen' in env_name:
                raw_dataset = env.get_dataset(METAWORLD_SUCCESS_DATA_PATHS)
            
            else: # otherwise metaworld single env
                mw_data_path = os.path.join(os.path.dirname(__file__), METAWORLD_SUCCESS_DATA_PATHS[env_name])
                raw_dataset = h5py.File(mw_data_path, 'r')

            dataset = qlearning_offline_dataset(raw_dataset, env_name, observation_type=observation_type)

        data_dict = {k: torch.from_numpy(item) for k, item in dataset.items() if isinstance(item, np.ndarray)}

        if observation_type == "image_joints":
            data_dict['observations'] = dataset['observations']['vector']
            data_dict['next_observations'] = dataset['next_observations']['vector']

        elif observation_type == "image":
            data_dict['pixels'] = dataset['observations']
            data_dict['next_pixels'] = dataset['next_observations']

        dataset = make_tensordict(data_dict)

        # rename keys to match torchrl conventions
        if observation_type != 'image':
            dataset.rename_key("observations", "observation")
        
        dataset.set("next", dataset.select())
        dataset.rename_key("terminals", "terminal")
        
        if observation_type != 'image':
            dataset.rename_key("next_observations", ("next", "observation"))
        
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
        dataset["action"] = dataset["action"].to(torch.float32)

        if 'image' in observation_type:
            dataset.rename_key("next_pixels", ("next", "pixels"))
        
        # checking dtypes
        for key, spec in env.observation_spec.items(True, True):
            if key != "pixels" and observation_type != 'image':
                dataset[key] = dataset[key].to(spec.dtype)
                dataset["next", key] = dataset["next", key].to(spec.dtype)

        for key, spec in env.input_spec.items(True, True):
            dataset[key] = dataset[key].to(spec.dtype)

        dataset["reward"] = dataset["reward"].to(env.reward_spec.dtype)
        dataset["done"] = dataset["done"].bool()
        dataset["done"] = dataset["done"].unsqueeze(-1)
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
        

class SubTrajectoryReplay(OfflineExperienceReplay):
    def __init__(
            self,
            env_name: str,
            observation_type: str = 'image_joints',
            batch_size: int = 256,
            batch_length: int = 50,
            base_transform: Optional[Callable] = None,
            sample_transform: Optional[Callable] = None,
            split_trajs: bool = False,
            use_timeout_as_done: bool = True,
            multitask: bool = False,
    ):
        self.use_timeout_as_done = use_timeout_as_done  
        self.base_transform = base_transform
        dataset = self._get_dataset_direct(env_name, observation_type, multitask=multitask)

        if observation_type != "image":
            dataset["next", "observation"][dataset["next", "done"].squeeze()] = 0
        
        if "image" in observation_type:
            dataset["next", "pixels"][dataset["next", "done"].squeeze()] = 0

        if split_trajs:
            dataset = split_trajectories(dataset)

        self.valid_indices = self._get_valid_indices(dataset, batch_length)
        self.num_sub_trajs = len(self.valid_indices)
        self.batch_length = batch_length
        self.batch_size = batch_size
        self.dataset = dataset.contiguous()

        if sample_transform is None:
            sample_transform = Compose()
        elif not isinstance(sample_transform, Compose):
            sample_transform = Compose(sample_transform)
        sample_transform.eval()
        self.sample_transform = sample_transform
    
    def sample(self, batch_size: Optional[int] = None):
        if batch_size is None:
            batch_size = self.batch_size

        batch = self._get_sub_traj_batch(self.dataset, self.valid_indices, self.batch_length, batch_size)

        return self.sample_transform(batch)
    

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
            batch.append(dataset[start_idx:start_idx + batch_length].clone())
        batch_td = torch.stack(batch)
        
        return batch_td


class RSSMStateReplayBuffer(TensorDictReplayBuffer):
    def __init__(
            self,
            snapshot_path: str,
            batch_size: int = 256,
            sampler: Optional[Sampler] = None,
            writer: Optional[Writer] = None,
            collate_fn: Optional[Callable] = None,
            pin_memory: bool = False,
            prefetch: Optional[int] = None,
            transform: Optional["Transform"] = None,  # noqa-F821
    ):
        self.transform = transform
        dataset = self._get_rssm_dataset(snapshot_path)

        dataset["next", "observation"][dataset["next", "done"].squeeze()] = 0

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

    def _get_rssm_dataset(self, snapshot_path):
        snapshot = torchsnapshot.Snapshot(path=snapshot_path)
        encoded_dataset = TensorDict({}, [])
        target_state = {"state": encoded_dataset}
        snapshot.restore(app_state=target_state)

        # replace observation with belief + state
        encoded_dataset['observation'] = torch.cat((encoded_dataset['belief'], encoded_dataset['state']), dim=1)
        encoded_dataset['next', 'observation'] = torch.cat((encoded_dataset['next', 'belief'], encoded_dataset['next', 'state']), dim=1)
        
        # only keep necessary keys
        encoded_dataset = encoded_dataset.select('observation', 'action', 'reward', 'done', 'terminal', ('next', 'observation'), ('next', 'reward'), ('next', 'done'), ('next', 'terminal'))

        return encoded_dataset
    