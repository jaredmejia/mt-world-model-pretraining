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

from dataset_utils import qlearning_kitchen_dataset


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
            env_transforms = Compose(ToTensorImage(), CenterCrop(96))
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
        