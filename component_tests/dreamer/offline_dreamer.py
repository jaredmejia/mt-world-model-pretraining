import dataclasses
from datetime import datetime
from pathlib import Path
import os
import sys

import hydra
import torch
import torch.cuda
import tqdm
from dreamer_utils import (
    call_record,
    EnvConfig,
    grad_norm,
    make_recorder_env,
    parallel_env_constructor,
    transformed_env_constructor,
    compute_obs_reco_imagined,
    make_pixel_vec_dreamer_world_model
)
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

# float16
from torch.cuda.amp import autocast, GradScaler
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs import CenterCrop, Compose, EnvCreator, ParallelEnv, ToTensorImage, TransformedEnv
from torchrl.envs.transforms import TensorDictPrimer, ObservationNorm
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import MLP
from torchrl.modules.models.model_based import (
    ObsEncoder,
    RSSMPosterior,
    RSSMPrior,
)
from torchrl.modules.tensordict_module.exploration import (
    AdditiveGaussianWrapper,
    OrnsteinUhlenbeckProcessWrapper,
)
from torchrl.objectives.dreamer import (
    DreamerActorLoss,
    DreamerModelLoss,
    DreamerValueLoss,
)
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.trainers.helpers.collectors import (
    make_collector_offpolicy,
    OffPolicyCollectorConfig,
)
from torchrl.trainers.helpers.envs import (
    correct_for_frame_skip,
    initialize_observation_norm_transforms,
    retrieve_observation_norms_state_dict,
)
from torchrl.trainers.helpers.logger import LoggerConfig
from torchrl.trainers.helpers.models import DreamerConfig, make_dreamer, _dreamer_make_mbenv
from torchrl.trainers.helpers.replay_buffer import make_replay_buffer, ReplayArgsConfig
from torchrl.trainers.helpers.trainers import TrainerConfig
from torchrl.trainers.trainers import Recorder, RewardNormalizer


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from datasets import SubTrajectoryReplay, env_maker, get_env_transforms
from modules import PixelVecNet, PixelVecObsDecoder
from modules.objectives import PixelVecDreamerModelLoss

config_fields = [
    (config_field.name, config_field.type, config_field)
    for config_cls in (
        OffPolicyCollectorConfig,
        EnvConfig,
        LoggerConfig,
        ReplayArgsConfig,
        DreamerConfig,
        TrainerConfig,
    )
    for config_field in dataclasses.fields(config_cls)
]
Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
cs = ConfigStore.instance()
cs.store(name="multitask_offline_config", node=Config)


def retrieve_stats_from_state_dict(obs_norm_state_dict):
    return {
        "loc": obs_norm_state_dict["loc"],
        "scale": obs_norm_state_dict["scale"],
    }


def get_env_obs_stats(cfg, custom_env):
    key, init_env_steps, stats = None, None, None
    if not cfg.vecnorm and cfg.norm_stats:
        if not hasattr(cfg, "init_env_steps"):
            raise AttributeError("init_env_steps missing from arguments.")
        key = ("next", "pixels") if cfg.from_pixels else ("next", "observation_vector")
        init_env_steps = cfg.init_env_steps
        stats = {"loc": None, "scale": None}
    elif cfg.from_pixels:
        stats = {"loc": 0.5, "scale": 0.5}

    proof_env = transformed_env_constructor(
        cfg=cfg, use_env_creator=False, stats=stats, custom_env=custom_env
    )()
    initialize_observation_norm_transforms(
        proof_environment=proof_env, num_iter=init_env_steps, key=key
    )
    _, obs_norm_state_dict = retrieve_observation_norms_state_dict(proof_env)[0]
    proof_env.close()

    return obs_norm_state_dict


def get_pixel_vec_world_model(cfg, proof_env, use_decoder_in_env=False, device=None):
    mlp_kwargs = {
        "num_cells": [512],
        "out_features": 512,
        "activation_class": nn.ReLU,
        "dropout": 0.0,
    }
    vec_dim = proof_env.observation_spec["observation"].shape[0]

    pixel_vec_decoder_net = PixelVecObsDecoder(vec_dim=vec_dim)
    pixel_vec_encoder_net = PixelVecNet(mlp_kwargs=mlp_kwargs, cnn=ObsEncoder())

    rssm_prior = RSSMPrior(
        hidden_dim=cfg.rssm_hidden_dim,
        rnn_hidden_dim=cfg.rssm_hidden_dim,
        state_dim=cfg.state_dim,
        action_spec=proof_env.action_spec,
    )

    rssm_posterior = RSSMPosterior(
        hidden_dim=cfg.rssm_hidden_dim, state_dim=cfg.state_dim
    )

    reward_module = MLP(
        out_features=1, depth=2, num_cells=cfg.mlp_num_units, activation_class=nn.ELU
    )

    pixel_vec_world_model = make_pixel_vec_dreamer_world_model(
        pixel_vec_encoder_net, pixel_vec_decoder_net, rssm_prior, rssm_posterior, reward_module
    ).to(device)

    # Initialize the world model
    with torch.no_grad(), set_exploration_mode("random"):
        tensordict = proof_env.rollout(4)
        tensordict = tensordict.to_tensordict().to(device)
        tensordict = tensordict.to(device)
        pixel_vec_world_model(tensordict)

    model_based_env = _dreamer_make_mbenv(
        reward_module,
        rssm_prior,
        pixel_vec_decoder_net,
        proof_env,
        use_decoder_in_env,
        cfg.state_dim,
        cfg.rssm_hidden_dim,
    )
    model_based_env = model_based_env.to(device)

    return pixel_vec_world_model, model_based_env


def make_logger(cfg, observation_type):
    exp_name = generate_exp_name("Dreamer", cfg.exp_name)
    logger = get_logger(
        logger_type=cfg.logger,
        logger_name="./offline_dreamer_logs",
        experiment_name=exp_name,
        wandb_kwargs={
            "project": f"{cfg.env_task}-training",
            "group": f"{observation_type}",
            "offline": cfg.offline_logging,
            'entity': 'neuropioneers'
        },
    )
    return logger


def offline_kitchen_transforms(batch_size, image_size, state_dim, hidden_dim, obs_stats=None):
    if isinstance(batch_size, int):
        batch_size = (batch_size,)

    fill_keys = TensorDictPrimer(primers={'state': UnboundedContinuousTensorSpec(
                    shape=torch.Size([*batch_size, state_dim]), dtype=torch.float32), 'belief': UnboundedContinuousTensorSpec(
                    shape=torch.Size([*batch_size, hidden_dim]), dtype=torch.float32)}, default_value=0, random=False)
    pixel_keys =["pixels", ("next", "pixels")]
    img_transform = CenterCrop(image_size, in_keys=pixel_keys, out_keys=pixel_keys)

    if obs_stats is not None:
        obs_norm = ObservationNorm(**obs_stats, in_keys=["pixels"])
        kitchen_transforms = Compose(fill_keys, img_transform, obs_norm)
    else:
        kitchen_transforms = Compose(fill_keys, img_transform)

    return kitchen_transforms


def update_world_model(world_model_loss, scaler1, world_model_opt, world_model, cfg, logger, i, j, sampled_tensordict):
    with autocast(dtype=torch.float16):
        model_loss_td, sampled_tensordict = world_model_loss(
            sampled_tensordict
        )
        loss_world_model = (
            model_loss_td["loss_model_kl"]
            + model_loss_td["loss_model_reco"]
            + model_loss_td["loss_model_reward"]
        )

        # If we are logging videos, we keep some frames.
        if (
            cfg.record_video
            and i % cfg.record_interval == 0
        ):
            sampled_tensordict_save = (
                sampled_tensordict[:4].detach().clone().to_tensordict()
            )
        else:
            sampled_tensordict_save = None

        scaler1.scale(loss_world_model).backward()
        scaler1.unscale_(world_model_opt)
        clip_grad_norm_(world_model.parameters(), cfg.grad_clip)
        scaler1.step(world_model_opt)
        if j == cfg.optim_steps_per_batch - 1:
            logger.log_scalar(
                "loss_world_model",
                loss_world_model.detach().item(),
                step=i,
            )
            logger.log_scalar(
                "grad_world_model",
                grad_norm(world_model_opt),
                step=i,
            )
            logger.log_scalar(
                "loss_model_kl",
                model_loss_td["loss_model_kl"].detach().item(),
                step=i,
            )
            logger.log_scalar(
                "loss_model_reco",
                model_loss_td["loss_model_reco"].detach().item(),
                step=i,
            )
            logger.log_scalar(
                "loss_model_reward",
                model_loss_td["loss_model_reward"].detach().item(),
                step=i,
            )
        world_model_opt.zero_grad()
        scaler1.update()

    return sampled_tensordict, sampled_tensordict_save


def update_actor(actor_loss, scaler2, actor_opt, actor_model, cfg, logger, i, j, sampled_tensordict):
    with autocast(dtype=torch.float16):
        actor_loss_td, sampled_tensordict = actor_loss(sampled_tensordict)
    scaler2.scale(actor_loss_td["loss_actor"]).backward()
    scaler2.unscale_(actor_opt)
    clip_grad_norm_(actor_model.parameters(), cfg.grad_clip)
    scaler2.step(actor_opt)
    if j == cfg.optim_steps_per_batch - 1:
        logger.log_scalar(
            "loss_actor",
            actor_loss_td["loss_actor"].detach().item(),
            step=i,
        )
        logger.log_scalar(
            "grad_actor",
            grad_norm(actor_opt),
            step=i,
        )
    actor_opt.zero_grad()
    scaler2.update()

    return sampled_tensordict


def update_value(value_loss, scaler3, value_opt, value_model, cfg, logger, i, j, sampled_tensordict):
    with autocast(dtype=torch.float16):
        value_loss_td, sampled_tensordict = value_loss(sampled_tensordict)
    scaler3.scale(value_loss_td["loss_value"]).backward()
    scaler3.unscale_(value_opt)
    clip_grad_norm_(value_model.parameters(), cfg.grad_clip)
    scaler3.step(value_opt)
    if j == cfg.optim_steps_per_batch - 1:
        logger.log_scalar(
            "loss_value",
            value_loss_td["loss_value"].detach().item(),
            step=i,
        )
        logger.log_scalar(
            "grad_value",
            grad_norm(value_opt),
            step=i,
        )
    value_opt.zero_grad()
    scaler3.update()

    return sampled_tensordict


def save_wmodels(model_based_env, cond_wmodel, save_path, cfg, i):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

        # save cfg
        with open(os.path.join(save_path, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f)

    torch.save(model_based_env.state_dict(), os.path.join(save_path, f"model_based_env_{i}.pt"))
    torch.save(cond_wmodel.state_dict(), os.path.join(save_path, f"cond_wmodel_{i}.pt"))


@hydra.main(version_base=None, config_path=".", config_name="multitask_offline_config")
def main(cfg: "DictConfig"):  # noqa: F821

    observation_type = "image"

    # set transforms
    env_transforms_args = (cfg.env_name, cfg.image_size)
    env_transforms_kwargs = {
        "from_pixels": cfg.from_pixels,
        "state_dim": cfg.state_dim,
        "hidden_dim": cfg.rssm_hidden_dim,
    }
    base_env_transforms = get_env_transforms(
        *env_transforms_args, batch_size=(), train_type=None, **env_transforms_kwargs
    )
    buffer_sample_transforms = get_env_transforms(
        *env_transforms_args, batch_size=(cfg.batch_size, cfg.batch_length), train_type='dreamer', **env_transforms_kwargs
    )
    proof_env_transforms = get_env_transforms(
        *env_transforms_args, batch_size=(), train_type='dreamer', **env_transforms_kwargs
    )

    # get proof env
    proof_env = env_maker(env_name=cfg.env_name, from_pixels=cfg.from_pixels, image_size=cfg.image_size, env_transforms=proof_env_transforms.clone())

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the different components of dreamer
    if observation_type == "image_joints":
        world_model, model_based_env = get_pixel_vec_world_model(cfg, proof_env, use_decoder_in_env=True, device=device)
        world_model_loss = PixelVecDreamerModelLoss(world_model)
    
    else:
        world_model, model_based_env, actor_model, value_model, policy = make_dreamer(
            obs_norm_state_dict=None,
            cfg=cfg,
            device=device,
            use_decoder_in_env=True,
            action_key="action",
            value_key="state_value",
            proof_environment=proof_env
        )
        world_model_loss = DreamerModelLoss(world_model)

    # get logger
    logger = make_logger(cfg, observation_type=observation_type)

    # load offline data into sub trajectory replay buffer
    replay_buffer = SubTrajectoryReplay(
        cfg.env_name, 
        observation_type=observation_type, 
        batch_size=cfg.batch_size,
        batch_length=cfg.batch_length,
        base_transform=base_env_transforms,
        sample_transform=buffer_sample_transforms,
        multitask= cfg.env_task == 'multitask',
    )

    # get optimizers
    world_model_opt = torch.optim.Adam(world_model.parameters(), lr=cfg.world_model_lr)

    # create gradscalers
    scaler1 = GradScaler()

    max_steps_train = 3000
    ckpt_save_interval = 500
    ckpt_save_min_steps = 2000
    save_path = os.path.join(
        'ckpts', 
        f'{cfg.env_task}-offline-dreamer', 
        f'{cfg.exp_name}'
        f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}',
    )

    for i in tqdm.tqdm(range(1, max_steps_train + 1),
                        smoothing=0.1):
        
        for j in range(cfg.optim_steps_per_batch):
        
            # sample from replay buffer
            sampled_tensordict = replay_buffer.sample(cfg.batch_size).to(
                device, non_blocking=True
            )

            # update world model
            sampled_tensordict, sampled_tensordict_save = update_world_model(world_model_loss, scaler1, world_model_opt, world_model, cfg, logger, i, j, sampled_tensordict)

        if (
            cfg.record_video
            and i % cfg.record_interval == 0
        ):
            compute_obs_reco_imagined(
                logger,
                sampled_tensordict_save,
                model_based_env,
                world_model,
                observation_type=observation_type
            )

        # save model
        if (i > ckpt_save_min_steps and i % ckpt_save_interval == 0) or i == max_steps_train:
            save_wmodels(
                model_based_env,
                world_model,
                save_path,
                cfg,
                i,
            )


if __name__ == "__main__":
    main()
