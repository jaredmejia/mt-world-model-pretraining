import os
import sys

import numpy as np
from tensordict.nn import TensorDictModule
import hydra
import torch
from torch import nn
from torchrl.envs import EnvCreator, ParallelEnv
from torchrl.modules import MLP
from torchrl.record.loggers import generate_exp_name, get_logger
import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from datasets import OfflineExperienceReplay, RSSMStateReplayBuffer, env_maker, get_env_transforms
from modules import PixelVecNet

def get_bc_actor(cfg, test_env, in_keys):
    action_spec = test_env.action_spec
    mlp_kwargs = {
        "num_cells": [256, 256],
        "out_features": action_spec.shape[-1],
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
            "num_features": 8,
        }

        actor_net = PixelVecNet(
            mlp_kwargs=mlp_kwargs,
            cnn_kwargs=cnn_kwargs,
            learned_spatial_embedding_kwargs=learned_spatial_embedding_kwargs,
        )
    else:
        actor_net = MLP(**mlp_kwargs)

    actor_net = TensorDictModule(
        actor_net,
        in_keys=in_keys,
        out_keys=["action"],
    )

    return actor_net


@hydra.main(version_base=None, config_path=".", config_name="multitask_bc_config")
def main(cfg: "DictConfig"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = generate_exp_name("BC", cfg.exp_name)
    logger = get_logger(
        logger_type=cfg.logger,
        logger_name="bc_logging",
        experiment_name=exp_name,
        wandb_kwargs={"mode": cfg.mode, "entity": cfg.entity, "project": "BC_agents"},
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # set environment and transforms
    env_transforms = get_env_transforms(
        cfg.env_name,
        cfg.image_size,
        from_pixels=cfg.from_pixels,
        train_type='iql',
    )

    def env_factory(num_workers):
        """Creates an instance of the environment."""

        # 1.2 Create env vector
        vec_env = ParallelEnv(
            create_env_fn=EnvCreator(lambda: env_maker(env_name=cfg.env_name, from_pixels=cfg.from_pixels, image_size=cfg.image_size, env_transforms=env_transforms.clone())),
            num_workers=num_workers,
        )

        return vec_env

    # Sanity check
    test_env = env_factory(num_workers=1)

    if cfg.observation_type == "image_joints":
        assert cfg.from_pixels, "observation_type 'image_joints' requires from_pixels is True"
        in_keys = ["observation", "pixels"]
    else:
        in_keys = ["observation"]

    # create agent
    bc_agent = get_bc_actor(cfg, test_env, in_keys).to(device)

    # define loss
    loss_fn = nn.MSELoss()

    # define optimizer
    optimizer = torch.optim.Adam(bc_agent.parameters(), lr=1e-3)

    # Make Replay Buffer
    print("Creating Replay Buffer...")
    if cfg.env_task != "multitask": 
        replay_buffer = OfflineExperienceReplay(cfg.env_name, observation_type=cfg.observation_type, base_transform=env_transforms)

    else: # assume rssm states
        replay_buffer = RSSMStateReplayBuffer(cfg.data_snapshot_path, batch_size=cfg.batch_size)

    print("Replay Buffer Created!")

    save_path = os.path.join(cfg.save_path, cfg.env_task, exp_name)

    for i in tqdm.tqdm(range(1, cfg.max_steps + 1), smoothing=0.1):
        sampled_tensordict = replay_buffer.sample(cfg.batch_size).to(device)

        pred = bc_agent(sampled_tensordict.clone())

        loss = loss_fn(pred["action"], sampled_tensordict["action"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.log_scalar("loss", loss.item(), step=i)

        if i % cfg.ckpt_interval == 0 or i == cfg.max_steps:
            os.makedirs(save_path, exist_ok=True)
            torch.save(bc_agent.state_dict(), os.path.join(save_path, f"bc_agent_{i}.pt"))

if __name__ == "__main__":
    main()