# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import hydra
import os
import sys

import numpy as np
import torch
import torch.cuda
import tqdm
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer

from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import CenterCrop, Compose, EnvCreator, ParallelEnv, ToTensorImage, TransformedEnv
from torchrl.envs.libs.gym import GymEnv, GymWrapper
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal

from torchrl.objectives import SoftUpdate
from torchrl.objectives.iql import IQLLoss
from torchrl.record.loggers import generate_exp_name, get_logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import datasets


def env_maker(env_name, frame_skip=1, device="cpu", from_pixels=False):
    if 'kitchen' in env_name:
        import d4rl
        import gym
        custom_env = gym.make(env_name, render_imgs=from_pixels)
        custom_env = GymWrapper(custom_env, frame_skip=frame_skip, from_pixels=from_pixels )
        if from_pixels:
            custom_env = TransformedEnv(custom_env, Compose(ToTensorImage(), CenterCrop(96)))
        return custom_env
    
    return GymEnv(
        env_name, device=device, frame_skip=frame_skip, from_pixels=from_pixels
    )


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    buffer_scratch_dir="/tmp/",
    device="cpu",
    prefetch=3,
):
    if prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=buffer_scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=buffer_scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    return replay_buffer


@hydra.main(version_base=None, config_path=".", config_name="offline_config")
def main(cfg: "DictConfig"):  # noqa: F821

    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        and torch.cuda.device_count() > 0
        and cfg.device == "cuda:0"
        else torch.device("cpu")
    )

    exp_name = generate_exp_name("Online_IQL", cfg.exp_name)
    logger = get_logger(
        logger_type=cfg.logger,
        logger_name="iql_logging",
        experiment_name=exp_name,
        wandb_kwargs={"mode": cfg.mode, "entity": cfg.entity},
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    def env_factory(num_workers):
        """Creates an instance of the environment."""

        # 1.2 Create env vector
        vec_env = ParallelEnv(
            create_env_fn=EnvCreator(lambda: env_maker(env_name=cfg.env_name, from_pixels=cfg.from_pixels)),
            num_workers=num_workers,
        )

        return vec_env

    # Sanity check
    test_env = env_factory(num_workers=5)
    num_actions = test_env.action_spec.shape[-1]

    # Create Agent
    # Define Actor Network
    in_keys = ["observation"]
    action_spec = test_env.action_spec
    actor_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": 2 * num_actions,
        "activation_class": nn.ReLU,
        "dropout": cfg.actor_dropout,
    }

    actor_net = MLP(**actor_net_kwargs)

    dist_class = TanhNormal
    dist_kwargs = {
        "min": action_spec.space.minimum[-1],
        "max": action_spec.space.maximum[-1],
        "tanh_loc": cfg.tanh_loc,
    }

    actor_extractor = NormalParamExtractor(
        scale_mapping=f"biased_softplus_{cfg.default_policy_scale}",
        scale_lb=cfg.scale_lb,
    )

    actor_net = nn.Sequential(actor_net, actor_extractor)
    in_keys_actor = in_keys
    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=[
            "loc",
            "scale",
        ],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_mode="random",
        return_log_prob=False,
    )

    # Define Critic Network
    qvalue_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": 1,
        "activation_class": nn.ReLU,
    }

    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )

    # Define Value Network
    value_net_kwargs = {
        "num_cells": [256, 256],
        "out_features": 1,
        "activation_class": nn.ReLU,
    }
    value_net = MLP(**value_net_kwargs)
    value = ValueOperator(
        in_keys=in_keys,
        module=value_net,
    )

    model = nn.ModuleList([actor, qvalue, value]).to(device)

    # init nets
    with torch.no_grad():
        td = test_env.reset()
        td = td.to(device)
        actor(td)
        qvalue(td)
        value(td)

    del td
    test_env.close()
    test_env.eval()

    # Create IQL loss
    loss_module = IQLLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        value_network=model[2],
        num_qvalue_nets=2,
        gamma=cfg.gamma,
        temperature=cfg.temperature,
        expectile=cfg.expectile,
        loss_function="smooth_l1",
    )

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, cfg.target_update_polyak)

    # Make Replay Buffer
    replay_buffer = datasets.KitchenExperienceReplay('kitchen-complete-v0', observation_type=cfg.observation_type)

    # Optimizers
    params = list(loss_module.parameters())
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    rewards = []
    rewards_eval = []

    # Main loop
    target_net_updater.init_()

    collected_frames = 0

    r0 = None
    loss = None

    for i in tqdm.tqdm(range(1, cfg.max_steps + 1),
                       smoothing=0.1):
        (
            actor_losses,
            q_losses,
            value_losses,
        ) = ([], [], [])

        # sample from replay buffer
        sampled_tensordict = replay_buffer.sample(cfg.batch_size).clone()

        loss_td = loss_module(sampled_tensordict)

        actor_loss = loss_td["loss_actor"]
        q_loss = loss_td["loss_qvalue"]
        value_loss = loss_td["loss_value"]

        loss = actor_loss + q_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        q_losses.append(q_loss.item())
        actor_losses.append(actor_loss.item())
        value_losses.append(value_loss.item())

        # update qnet_target params
        target_net_updater.step()

        if q_loss is not None:
            train_log = {
                    "actor_loss": np.mean(actor_losses),
                    "q_loss": np.mean(q_losses),
                    "value_loss": np.mean(value_losses),
            }
            
        for key, value in train_log.items():
            logger.log_scalar(key, value, step=collected_frames)

        if i % cfg.eval_interval == 0:

            with set_exploration_mode("mean"), torch.no_grad():
                eval_rollout = test_env.rollout(
                    max_steps=cfg.max_frames_per_traj,
                    policy=model[0],
                    auto_cast_to_device=True,
                ).clone()
                eval_reward = eval_rollout["reward"].sum(-2).mean().item()
                rewards_eval.append((i, eval_reward))
                eval_str = f"eval cumulative reward: {rewards_eval[-1][1]: 4.4f} (init: {rewards_eval[0][1]: 4.4f})"
                logger.log_scalar("test_reward", rewards_eval[-1][1], step=collected_frames)


if __name__ == "__main__":
    main()
