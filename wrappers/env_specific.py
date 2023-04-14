from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import gym
from gym.spaces import Box, Dict
import numpy as np


class FrankaKitchen(gym.Wrapper):  # type: ignore
    """FrankaKitchen wrapper for image observation environments.

    Args:
        env (gym.Env): gym environment.
        obs_type (str): observation type. Either ``'image'``, ``'default'``, or ``'image_joints'``.
    """

    obs_type: str

    def __init__(self, env: gym.Env, obs_type: str = "default"):
        super().__init__(env)
        self.obs_type = obs_type
        img_height = env.img_height
        img_width = env.img_width
        if obs_type == "image":
            self.observation_space = Box(
                low=0,
                high=1,
                shape=(3, img_height, img_width),
                dtype=np.float32,
            )

            assert env.render_imgs is True

        elif obs_type == "default":
            self.observation_space = env.observation_space

        elif obs_type == "image_joints":
            spaces = {
                "image": Box(
                    low=0,
                    high=1,
                    shape=(img_height, img_width, 3),
                    dtype=np.uint8,
                ),
                "vector": Box(
                    low=env.observation_space.low[:9],
                    high=env.observation_space.high[:9],
                    shape=(9,),
                    dtype=np.float32,
                ),
            }
            self.observation_space = Dict(spaces)

            assert (
                env.render_imgs is True
            ), "env.render_imgs must be True for image_joints observation type."

        else:
            raise ValueError(
                "obs_type must be 'vector', 'image', image_joints'."
            )

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[Any, float, bool, Dict[str, Any]]:
        observation, reward, terminal, info = self.env.step(action)

        if self.obs_type == "image":
            img_observation = info["images"]
            img_observation = img_observation / 255.0
            observation = {"image": img_observation}
            
            # info = {}

        elif self.obs_type == "image_joints":
            img_observation = info["images"]
            img_observation = img_observation / 255.0
            joint_observation = observation[:9]

            img_observation = np.array(img_observation, dtype=np.float32)
            joint_observation = np.array(joint_observation, dtype=np.float32)
            observation = {"image": img_observation, "vector": joint_observation}

            # info = {}
        else:
            pass

        return observation, reward, terminal, info

    def reset(self, **kwargs: Any) -> np.ndarray:
        observation = self.env.reset(**kwargs)

        if self.obs_type == "image":
            img_observation = self.env.render()
            img_observation = img_observation / 255.0
            observation = {"image": img_observation}

        elif self.obs_type == "image_joints":
            img_observation = self.env.render()
            img_observation = img_observation / 255.0

            joint_observation = observation[:9]
            observation = {"image": img_observation, "vector": joint_observation}

        else:
            pass

        return observation
