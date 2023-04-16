import hydra
import os
import sys

import numpy as np
import torch
import torch.cuda
from torch import nn, optim
import tqdm

from functorch import vmap
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs import CenterCrop, Compose, EnvCreator, ParallelEnv, ToTensorImage, TransformedEnv
from torchrl.envs.libs.gym import GymEnv, GymWrapper
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import ConvNet, MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate
from torchrl.objectives.iql import IQLLoss
from torchrl.record.loggers import generate_exp_name, get_logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from datasets import KitchenExperienceReplay, KitchenFilterState
from modules import PixelVecNet


def env_maker(env_name, frame_skip=1, device="cpu", from_pixels=False):
    if 'kitchen' in env_name:
        import d4rl
        import gym
        custom_env = gym.make(env_name, render_imgs=from_pixels)
        custom_env = GymWrapper(custom_env, frame_skip=frame_skip, from_pixels=from_pixels )
        if from_pixels:
            pixel_keys =["pixels", ("next", "pixels")]
            state_keys = ["observation", ("next", "observation")]
            env_transforms = Compose(
                ToTensorImage(in_keys=pixel_keys, out_keys=pixel_keys), CenterCrop(96, in_keys=pixel_keys, out_keys=pixel_keys),
                KitchenFilterState(in_keys=state_keys, out_keys=state_keys),
            )
            custom_env = TransformedEnv(custom_env, env_transforms)
        return custom_env
    
    return GymEnv(
        env_name, device=device, frame_skip=frame_skip, from_pixels=from_pixels
    )


def get_actor(cfg, test_env, num_actions, in_keys):
    # Define Actor Network
    action_spec = test_env.action_spec
    mlp_kwargs = {
        "num_cells": [256, 256],
        "out_features": 2 * num_actions,
        "activation_class": nn.ReLU,
        "dropout": cfg.actor_dropout,
    }
    if "pixels" in in_keys:
        cnn_kwargs = {
                "bias_last_layer": True,
                "depth": None,
                "num_cells": [32, 64, 64],
                "kernel_sizes": [8, 4, 3],
                "strides": [4, 2, 1],
                "aggregator_class": nn.GroupNorm,
                "aggregator_kwargs": {"num_channels": 64, "num_groups": 4},
        }
        learned_spatial_embedding_kwargs = {
            "height": 8,
            "width": 8,
            "channels": 64,
            "num_features": 8,
        }

        actor_net = PixelVecNet(
            mlp_kwargs=mlp_kwargs,
            cnn_kwargs=cnn_kwargs,
            learned_spatial_embedding_kwargs=learned_spatial_embedding_kwargs,
        )
    else:
        actor_net = MLP(**mlp_kwargs)

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

    td_actor_net = TensorDictModule(
        actor_net,
        in_keys=in_keys,
        out_keys=['loc-scale']
    )
    td_actor_extractor = TensorDictModule(
        actor_extractor,
        in_keys=['loc-scale'],
        out_keys=['loc', 'scale']
    )
    actor_module = TensorDictSequential(td_actor_net, td_actor_extractor)

    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_mode="random",
        return_log_prob=False,
    )
    return actor


def get_critic(in_keys):
    # Define Critic Network
    mlp_kwargs = {
        "num_cells": [256, 256],
        "out_features": 1,
        "activation_class": nn.ReLU,
    }

    if "pixels" in in_keys:
        cnn_kwargs = {
                "bias_last_layer": True,
                "depth": None,
                "num_cells": [32, 64, 64],
                "kernel_sizes": [8, 4, 3],
                "strides": [4, 2, 1],
                "aggregator_class": nn.GroupNorm,
                "aggregator_kwargs": {"num_channels": 64, "num_groups": 4},
        }
        learned_spatial_embedding_kwargs = {
            "height": 8,
            "width": 8,
            "channels": 64,
            "num_features": 8,
        }

        qvalue_net = PixelVecNet(
            mlp_kwargs=mlp_kwargs,
            cnn_kwargs=cnn_kwargs,
            learned_spatial_embedding_kwargs=learned_spatial_embedding_kwargs,
            include_action=True
        )
    
    else:
        qvalue_net = MLP(
            **mlp_kwargs,
        )

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )

    return qvalue


def get_value(in_keys):
    # Define Value Network
    mlp_kwargs = {
        "num_cells": [256, 256],
        "out_features": 1,
        "activation_class": nn.ReLU,
    }

    if "pixels" in in_keys:
        cnn_kwargs = {
                "bias_last_layer": True,
                "depth": None,
                "num_cells": [32, 64, 64],
                "kernel_sizes": [8, 4, 3],
                "strides": [4, 2, 1],
                "aggregator_class": nn.GroupNorm,
                "aggregator_kwargs": {"num_channels": 64, "num_groups": 4},
        }
        learned_spatial_embedding_kwargs = {
            "height": 8,
            "width": 8,
            "channels": 64,
            "num_features": 8,
        }

        value_net = PixelVecNet(
            mlp_kwargs=mlp_kwargs,
            cnn_kwargs=cnn_kwargs,
            learned_spatial_embedding_kwargs=learned_spatial_embedding_kwargs,
        )
    
    else:
        value_net = MLP(**mlp_kwargs)

    value = ValueOperator(
        in_keys=in_keys,
        module=value_net,
    )

    return value


def get_q_val_estimate(tensordict, iql_loss_module, device):
    with torch.no_grad():
        td_q = tensordict.select(*iql_loss_module.qvalue_network.in_keys).detach()
        td_q = td_q.reshape(-1).to(device)
        td_q = vmap(iql_loss_module.qvalue_network, (None, 0))(
            td_q, iql_loss_module.target_qvalue_network_params
        )
        avg_q_val = td_q.get("state_action_value").mean().item()
        max_q_val = td_q.get("state_action_value").max().item()
    
    return avg_q_val, max_q_val


def get_value_estimate(tensordict, iql_loss_module, device):
    with torch.no_grad():
        td_val = tensordict.select(*iql_loss_module.value_network.in_keys).detach()
        td_val = td_val.reshape(-1).to(device)
        iql_loss_module.value_network(
            td_val,
            params=iql_loss_module.value_network_params,
        )
        avg_val = td_val.get("state_value").mean().item()
        max_val = td_val.get("state_value").max().item()
    
    return avg_val, max_val


@hydra.main(version_base=None, config_path=".", config_name="offline_config")
def main(cfg: "DictConfig"):  # noqa: F821

    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        and torch.cuda.device_count() > 0
        and cfg.device == "cuda:0"
        else torch.device("cpu")
    )

    exp_name = generate_exp_name("Offline_IQL", cfg.exp_name)
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

    if cfg.observation_type == "image_joints":
        assert cfg.from_pixels, "observation_type 'image_joints' requires from_pixels is True"
        in_keys = ["observation", "pixels"]
    else:
        in_keys = ["observation"]

    # Create Agent
    actor = get_actor(cfg, test_env, num_actions, in_keys)

    # Create Critic
    qvalue = get_critic(in_keys)

    # Create Value
    value = get_value(in_keys)

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
    replay_buffer = KitchenExperienceReplay('kitchen-complete-v0', observation_type=cfg.observation_type)

    # Optimizers
    params = list(loss_module.parameters())
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    rewards = []
    rewards_eval = []

    # Main loop
    target_net_updater.init_()

    i = 0

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
            logger.log_scalar(key, value, step=i)

        if i % cfg.eval_interval == 0:

            with set_exploration_mode("mean"), torch.no_grad():
                eval_rollout = test_env.rollout(
                    max_steps=cfg.max_frames_per_traj,
                    policy=model[0],
                    auto_cast_to_device=True,
                ).clone()
                
                # log reward
                eval_reward = eval_rollout["reward"].sum(-2).mean().item()
                rewards_eval.append((i, eval_reward))
                eval_str = f"eval cumulative reward: {rewards_eval[-1][1]: 4.4f} (init: {rewards_eval[0][1]: 4.4f})"
                logger.log_scalar("test_reward", rewards_eval[-1][1], step=i)

                # log q-value estimates
                q_value_avg, q_value_max = get_q_val_estimate(eval_rollout, loss_module, device)
                logger.log_scalar("q_value_avg", q_value_avg, step=i)
                logger.log_scalar("q_value_max", q_value_max, step=i)

                # log value estimates
                value_avg, value_max = get_value_estimate(eval_rollout, loss_module, device)
                logger.log_scalar("value_avg", value_avg, step=i)
                logger.log_scalar("value_max", value_max, step=i)


if __name__ == "__main__":
    main()
