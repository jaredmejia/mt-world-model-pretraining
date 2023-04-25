import argparse
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
)
from hydra.core.config_store import ConfigStore
import moviepy.editor as mpy
from omegaconf import OmegaConf
import wandb

# float16
from torchvision.io import write_video
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs import CenterCrop, Compose, EnvCreator, ParallelEnv, ToTensorImage, TransformedEnv
from torchrl.envs.transforms import TensorDictPrimer, ObservationNorm
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
from torchrl.trainers.helpers.models import DreamerConfig, make_dreamer
from torchrl.trainers.helpers.replay_buffer import make_replay_buffer, ReplayArgsConfig
from torchrl.trainers.helpers.trainers import TrainerConfig
from torchrl.trainers.trainers import Recorder, RewardNormalizer


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets import SubTrajectoryReplay, env_maker, get_env_transforms
from dreamer_utils import conditional_model_rollout, recover_pixels


def load_wmodels(model_based_env, cond_wmodel, ckpt_dir, ckpt_num=None):
    if ckpt_num is None:
        # load the latest checkpoint
        model_based_env_path = get_latest_ckpt(ckpt_dir, "model_based_env")
        cond_wmodel_path = get_latest_ckpt(ckpt_dir, "cond_wmodel")
    else:
        model_based_env_path = os.path.join(ckpt_dir, f"model_based_env_{ckpt_num}.pt")
        cond_wmodel_path = os.path.join(ckpt_dir, f"cond_wmodel_{ckpt_num}.pt")
    
    model_based_env.load_state_dict(torch.load(model_based_env_path))
    cond_wmodel.load_state_dict(torch.load(cond_wmodel_path))

    return model_based_env, cond_wmodel


def get_latest_ckpt(ckpt_dir, prefix):
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.startswith(prefix)]
    ckpt_files.sort()
    latest_ckpt = ckpt_files[-1]

    return os.path.join(ckpt_dir, latest_ckpt)


def save_rollout(tensordict, model_based_env, cond_wmodel, save_dir):
    true_pixels = recover_pixels(tensordict[("next", "pixels")])

    # get the model-based rollout
    rollout_td = conditional_model_rollout(tensordict, model_based_env.world_model[0], cond_wmodel)

    # get the model-based rollout pixels
    with torch.no_grad():
        imagine_pixels = recover_pixels(
            model_based_env.decode_obs(rollout_td)["next", "reco_pixels"],
        )

    stacked_pixels = torch.cat([true_pixels, imagine_pixels], dim=-1)    

    os.makedirs(save_dir, exist_ok=True)
    out_fname = os.path.join(save_dir, f"rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")

    save_video(stacked_pixels, out_fname)


def save_video(stacked_pixels, out_fname):
    wandb_video = wandb.Video(stacked_pixels.cpu(), fps=6)
    prepped_video = wandb_video._prepare_video(wandb_video.data)
    clip = mpy.ImageSequenceClip(list(prepped_video), fps=6)
    clip.write_videofile(out_fname)    


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=False, default="kitchen-complete-v0-offline-dreamer-2023-04-21-19-13-10")
    parser.add_argument("--ckpt_num", type=int, required=False, default=None)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--num_trajs", type=int, default=4)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir else os.path.join(args.ckpt_dir, f"rollout_{args.ckpt_num if args.ckpt_num else 'latest'}")

    cfg_path = os.path.join(args.ckpt_dir, "config.yaml")
    cfg = OmegaConf.load(cfg_path)

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
    world_model, model_based_env, actor_model, value_model, policy = make_dreamer(
        obs_norm_state_dict=None,
        cfg=cfg,
        device=device,
        use_decoder_in_env=True,
        action_key="action",
        value_key="state_value",
        proof_environment=proof_env,
    )

    # load world models
    model_based_env, cond_wmodel = load_wmodels(model_based_env, world_model, args.ckpt_dir, args.ckpt_num)

    # load offline data into sub trajectory replay buffer
    replay_buffer = SubTrajectoryReplay(
        cfg.env_name, 
        observation_type='image_joints', 
        batch_size=cfg.batch_size,
        batch_length=cfg.batch_length,
        base_transform=base_env_transforms,
        sample_transform=buffer_sample_transforms,
        multitask= cfg.env_task == 'multitask',
    )
    sample_td = replay_buffer.sample().to(device)

    # save rollout
    save_rollout(sample_td[:4].clone().to_tensordict(), model_based_env, cond_wmodel, out_dir)


if __name__ == "__main__":
    main()