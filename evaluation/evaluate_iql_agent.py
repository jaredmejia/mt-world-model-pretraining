"""Script for evaluating IQL agents on MetaWorld tasks."""

import argparse
from datetime import datetime
import os
import sys

import metaworld
from omegaconf import OmegaConf 
import pickle
import torch
from tqdm import tqdm

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(parent_dir, 'component_tests', 'iql'))
from iql_offline import get_actor

sys.path.append(os.path.join(parent_dir, 'component_tests', 'dreamer'))
from gen_rollout import save_video

from datasets import get_env_transforms, eval_metaworld_env_maker, env_maker


def save_eval_rollout_vid(env_names, rollout, save_dir, max_steps=280):
    stacked_pixels = []
    for env_name in env_names:
        stacked_pixels.append(rollout[env_name]['pixels'][:max_steps].clone())
    stacked_pixels = torch.stack(stacked_pixels, dim=0)
    stacked_pixels = (stacked_pixels * 255).to(torch.uint8)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'rollout_vids.mp4')
    save_video(stacked_pixels, save_path)


def get_rollout_stats(env_names, rollout, num_tasks=50):
    rollout_stats = {}
    for env_name in env_names:
        rollout_stats[env_name] = {}
        rollout_stats[env_name]['success'] = rollout[env_name]['success'].sum().item() / num_tasks
        rollout_stats[env_name]['reward'] = rollout[env_name]['reward'].sum().item() / num_tasks
        rollout_stats[env_name]['done'] = rollout[env_name]['done'].sum().item() / num_tasks
    
    return rollout_stats


def eval_metaworld(env_names, eval_tasks, policy, max_steps=280, metaworld_transforms=None):
    rollouts = {}
    for env_name in env_names:
        print(f'evaluating {env_name}...')
        rollouts[env_name] = []
        for env_task in tqdm(eval_tasks[env_name]):
            env = eval_metaworld_env_maker(env_name, env_task, env_transforms=metaworld_transforms)

            with torch.no_grad():
                rollout_td = env.rollout(max_steps=max_steps, policy=policy, auto_cast_to_device=True).clone()
            
            rollouts[env_name].append(rollout_td.cpu())
            env.close()

        rollouts[env_name] = torch.cat(rollouts[env_name], dim=0)
    
    return rollouts


def load_actor(cfg, ckpt_path, sample_env, num_actions, in_keys, device):
    actor = get_actor(cfg, sample_env, num_actions, in_keys)
    actor.load_state_dict(torch.load(ckpt_path, map_location=device))
    actor.to(device)
    actor.eval()

    return actor


def get_eval_tasks(cfg, multitask=False):
    mt10 = metaworld.MT10()

    if multitask:
        eval_env_names = list(mt10.train_classes.keys())
    
    else:
        eval_env_names = [cfg.env_name]

    eval_tasks = {}
    for env_name in eval_env_names:
        env_tasks = [task for task in mt10.train_tasks if task.env_name == env_name]
        print(f'{env_name}: {len(env_tasks)} tasks')
        eval_tasks[env_name] = env_tasks

    return mt10, eval_env_names, eval_tasks
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--actor_ckpt_path', type=str, required=True)
    parser.add_argument('--multitask', type=bool, default=False)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--save_video', type=bool, default=False)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sample_env_transforms = get_env_transforms(
        cfg.env_name,
        cfg.image_size,
        from_pixels=cfg.from_pixels,
        train_type='iql',
    )

    sample_env = env_maker(cfg.env_name, cfg.image_size, env_transforms=sample_env_transforms)
    num_actions = sample_env.action_spaces.shape[-1]
    
    if cfg.observation_type == "image_joints":
        assert cfg.from_pixels, "observation_type 'image_joints' requires from_pixels is True"
        in_keys = ["observation", "pixels"]
    else:
        in_keys = ["observation"]

    # load actor
    actor = load_actor(args.actor_ckpt_path, sample_env, num_actions, in_keys, device)

    # get eval env names and tasks
    mt10, eval_env_names, eval_tasks = get_eval_tasks(args.multitask)

    eval_transforms = get_env_transforms(
        'metaworld',
        cfg.image_size,
        eval=True
    )

    # evaluate
    eval_rollouts = eval_metaworld(eval_env_names, eval_tasks, actor, metaworld_transforms=eval_transforms)

    # get stats
    eval_stats = get_rollout_stats(eval_env_names, eval_rollouts)
    for env_name in eval_env_names:
        for key, value in eval_stats[env_name].items():
            print(f'{env_name} {key}: {value}')
    
    # save stats
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'eval_stats.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(eval_stats, f)

    # save video
    if args.save_video:
        save_eval_rollout_vid(eval_env_names, eval_rollouts, args.save_dir)


if __name__ == '__main__':
    main()
