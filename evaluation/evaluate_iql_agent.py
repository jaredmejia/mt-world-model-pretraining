"""Script for evaluating IQL agents on MetaWorld tasks."""

import argparse
from datetime import datetime
import os
import sys

import metaworld
import numpy as np
from omegaconf import OmegaConf
import pickle
from tensordict import TensorDict
import torch
from torchrl.data import UnboundedContinuousTensorSpec
from torchrl.envs.transforms import TensorDictPrimer
from tqdm import tqdm

from torchrl.envs import EnvCreator, ParallelEnv

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.join(parent_dir, "component_tests", "iql"))
from iql_offline import get_actor as get_iql_actor

sys.path.append(os.path.join(parent_dir, "component_tests", "dreamer"))
from gen_rollout import save_video

sys.path.append(os.path.join(parent_dir, "component_tests", "iql_dreamer"))
from iql_rssm_offline import prep_wmodel

sys.path.append(os.path.join(parent_dir, "component_tests", "bc"))
from bc_train import get_bc_actor


from datasets import get_env_transforms, eval_metaworld_env_maker, env_maker
from datasets.metadata import METAWORLD_SUCCESS_DATA_PATHS
from datasets.transforms import AppendEnvID


def expand_to_length(tensor, length=500):
    if len(tensor) < length:
        tensor = torch.cat(
            [tensor, tensor[-1].clone().repeat(length - len(tensor), 1, 1, 1)], dim=0
        )

    return tensor


def combine_pixels(list_of_tensors):
    max_length = max([len(tensor) for tensor in list_of_tensors])
    combined_pixels = []

    for tensor in list_of_tensors:
        combined_pixels.append(expand_to_length(tensor, length=max_length))

    return torch.stack(combined_pixels, dim=0)


def save_eval_rollout_vids(eval_pixels, save_dir):
    for env_name in eval_pixels.keys():
        out_dir = os.path.join(save_dir, env_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"Saving rollout video for {env_name} to {out_dir}")

        env_pixels = eval_pixels[env_name]

        if len(env_pixels["success"]) > 0:
            success_pixels = combine_pixels(env_pixels["success"])
            success_pixels = (success_pixels * 255).to(torch.uint8)

            save_path = os.path.join(out_dir, "success.mp4")
            save_video(success_pixels, save_path)

        if len(env_pixels["failure"]) > 0:
            failure_pixels = combine_pixels(env_pixels["failure"])
            failure_pixels = (failure_pixels * 255).to(torch.uint8)

            save_path = os.path.join(out_dir, "failure.mp4")
            save_video(failure_pixels, save_path)


def conditional_iql_rssm_rollout(
    env, policy, cond_wmodel, max_num_steps, fill_hidden_keys, save_keys
):
    env_td = env.reset()

    input_td = env_td.clone().unsqueeze(0)
    input_td = fill_hidden_keys(input_td)
    input_td["observation"] = torch.cat(
        (
            input_td["observation"].clone(),
            input_td["belief"].clone(),
            input_td["state"].clone(),
        ),
        dim=1,
    )
    input_td[("next", "pixels")] = input_td["pixels"].clone()

    rollout = []
    for _ in range(max_num_steps):
        with torch.no_grad():
            # get action prediction
            iql_actor_out = policy(input_td.clone())

            # update belief and state
            cond_wmodel_out = cond_wmodel(iql_actor_out.clone())

        # step env
        env_td = env.step(cond_wmodel_out.clone()[0].select("action"))

        # update input_td
        input_td = TensorDict({}, batch_size=1)

        input_td["next", "pixels"] = env_td[("next", "pixels")].clone().unsqueeze(0)
        input_td["done"] = env_td["next", "done"].clone().unsqueeze(0)
        input_td["reward"] = env_td["next", "reward"].clone().unsqueeze(0)
        input_td["success"] = env_td["next", "success"].clone().unsqueeze(0)

        # save rollout
        rollout.append(input_td.clone().select(*save_keys))

        if input_td["success"].sum():
            break

        input_td["observation"] = torch.cat(
            (
                env_td.clone()["next", "observation"].unsqueeze(0),
                cond_wmodel_out.clone()["next", "belief"],
                cond_wmodel_out.clone()["next", "state"],
            ),
            dim=1,
        )
        input_td["belief"] = cond_wmodel_out.clone()["next", "belief"]
        input_td["state"] = cond_wmodel_out.clone()["next", "state"]

    rollout_td = torch.cat(rollout, dim=0)

    return rollout_td


def eval_with_rssm_metaworld(
    mt10,
    eval_env_names,
    eval_env_tasks,
    actor,
    cond_wmodel,
    fill_hidden_keys,
    eval_transforms,
    max_num_steps=500,
    device=None,
    max_success_vids=8,
    max_failure_vids=8,
    multitask=True
):
    eval_pixels = {}
    eval_stats = {}
    save_keys = ("reward", "success", ("next", "pixels"))

    for env_name in eval_env_names:
        eval_pixels[env_name] = {"success": [], "failure": []}
        eval_stats[env_name] = {}
        rewards = []
        successes = 0

        env_transforms = eval_transforms.clone()
        if multitask:
            env_transforms.append(AppendEnvID(env_name))

        print(f"evaluating {env_name}...")
        for env_task in tqdm(eval_env_tasks[env_name]):

            # create eval env
            eval_env = eval_metaworld_env_maker(
                mt10, env_name, env_task, env_transforms=env_transforms.clone()
            )
            eval_env.to(device)

            # rollout
            rollout_td = conditional_iql_rssm_rollout(
                eval_env, actor, cond_wmodel, max_num_steps, fill_hidden_keys.clone(), save_keys
            ).clone().cpu()

            # save pixels
            if (
                rollout_td["success"].sum()
                and len(eval_pixels[env_name]["success"]) < max_success_vids
            ):
                eval_pixels[env_name]["success"].append(
                    rollout_td["next", "pixels"].clone()
                )

            elif len(eval_pixels[env_name]["failure"]) < max_failure_vids:
                eval_pixels[env_name]["failure"].append(
                    rollout_td["next", "pixels"].clone()
                )

            # save stats
            rewards.append(rollout_td["reward"].sum().item())
            successes += rollout_td["success"].sum().item()

        # aggregate stats
        eval_stats[env_name]["reward_mean"] = np.mean(rewards)
        eval_stats[env_name]["reward_std"] = np.std(rewards)
        eval_stats[env_name]["success_rate"] = successes / len(eval_env_tasks[env_name])

    return eval_pixels, eval_stats


def eval_metaworld(
    mt10,
    env_names,
    eval_tasks,
    policy,
    max_steps=500,
    metaworld_transforms=None,
    multitask=False,
    max_success_vids=8,
    max_failure_vids=8,
    device=None,
):
    eval_pixels = {}
    eval_stats = {}
    for env_name in env_names:
        eval_pixels[env_name] = {"success": [], "failure": []}
        eval_stats[env_name] = {}
        rewards = []
        successes = 0

        env_transforms = metaworld_transforms.clone()

        if multitask:
            env_transforms.append(AppendEnvID(env_name))

        print(f"evaluating {env_name}...")
        for env_task in tqdm(eval_tasks[env_name]):
            # create env
            env = ParallelEnv(
                create_env_fn=EnvCreator(
                    lambda: eval_metaworld_env_maker(
                        mt10,
                        env_name,
                        env_task,
                        env_transforms=env_transforms.clone(),
                    )
                ),
                num_workers=1,
            )

            # rollout
            with torch.no_grad():
                rollout_td = env.rollout(
                    max_steps=max_steps, policy=policy, auto_cast_to_device=True
                ).clone().cpu()

            # save pixels
            if (
                rollout_td["success"].sum()
                and len(eval_pixels[env_name]["success"]) < max_success_vids
            ):
                eval_pixels[env_name]["success"].append(
                    rollout_td["next", "pixels"].clone()
                )

            elif len(eval_pixels[env_name]["failure"]) < max_failure_vids:
                eval_pixels[env_name]["failure"].append(
                    rollout_td["next", "pixels"].clone()
                )

            # save stats
            rewards.append(rollout_td["reward"].sum().item())
            successes += rollout_td["success"].sum().item()

        # aggregate stats
        eval_stats[env_name]["reward_mean"] = np.mean(rewards)
        eval_stats[env_name]["reward_std"] = np.std(rewards)
        eval_stats[env_name]["success_rate"] = successes / len(eval_tasks[env_name])

    return eval_pixels, eval_stats


def load_actor(cfg, ckpt_path, sample_env, num_actions, in_keys, device):

    if cfg.agent_type == "bc":
        actor = get_bc_actor(cfg, sample_env, in_keys)
    else:
        actor = get_iql_actor(cfg, sample_env, num_actions, in_keys)

    actor.load_state_dict(torch.load(ckpt_path, map_location=device))
    actor.to(device)
    actor.eval()

    return actor


def get_eval_tasks(cfg, multitask=False):
    mt10 = metaworld.MT10()

    if multitask:
        eval_env_names = list(METAWORLD_SUCCESS_DATA_PATHS.keys())

    else:
        eval_env_names = [cfg.env_name]

    eval_tasks = {}
    for env_name in eval_env_names:
        env_tasks = [task for task in mt10.train_tasks if task.env_name == env_name]
        eval_tasks[env_name] = env_tasks

        print(f"{env_name}: {len(env_tasks)} tasks")

    return mt10, eval_env_names, eval_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--actor_ckpt_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_video", type=bool, default=True)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_env_transforms = get_env_transforms(
        cfg.env_name,
        cfg.image_size,
        from_pixels=cfg.from_pixels,
        train_type="iql",
    )

    sample_env = env_maker(
        cfg.env_name, cfg.image_size, env_transforms=sample_env_transforms
    )
    num_actions = sample_env.action_spec.shape[-1]

    if cfg.observation_type == "image_joints":
        assert (
            cfg.from_pixels
        ), "observation_type 'image_joints' requires from_pixels is True"
        in_keys = ["observation", "pixels"]
    else:
        in_keys = ["observation"]

    # get eval env names and tasks
    mt10, eval_env_names, eval_tasks = get_eval_tasks(cfg, multitask=cfg.env_task == "multitask")

    eval_transforms = get_env_transforms(
        "metaworld", cfg.image_size, eval=True, multitask=cfg.env_task == "multitask"
    )

    # load actor
    actor = load_actor(
        cfg, args.actor_ckpt_path, sample_env, num_actions, in_keys, device
    )

    # load world model if needed
    wmodel_ckpt_path = cfg.get("wmodel_ckpt_path", None)
    if wmodel_ckpt_path is not None:
        wmodel_ckpt_path = os.path.join("../component_tests/dreamer", wmodel_ckpt_path)
        cond_wmodel, dreamer_cfg = prep_wmodel(wmodel_ckpt_path, device)

        fill_hidden_keys = TensorDictPrimer(
            primers={
                "state": UnboundedContinuousTensorSpec(
                    shape=torch.Size([1, dreamer_cfg.state_dim]), dtype=torch.float32
                ),
                "belief": UnboundedContinuousTensorSpec(
                    shape=torch.Size([1, dreamer_cfg.rssm_hidden_dim]),
                    dtype=torch.float32,
                ),
            },
            default_value=0,
            random=False,
        )

        # evaluate iql-rssm
        eval_pixels, eval_stats = eval_with_rssm_metaworld(
            mt10,
            eval_env_names,
            eval_tasks,
            actor,
            cond_wmodel,
            fill_hidden_keys,
            eval_transforms,
            max_num_steps=500,
            device=device,
        )

    else:
        # evaluate iql
        eval_pixels, eval_stats = eval_metaworld(
            mt10,
            eval_env_names,
            eval_tasks,
            actor,
            max_steps=500,
            multitask=cfg.env_task == "multitask",
            metaworld_transforms=eval_transforms,
        )

    # print stats
    print("\n\nEvaluation stats:\n")
    for env_name in eval_stats.keys():
        print(f"{env_name}:")
        for stat_name, stat_value in eval_stats[env_name].items():
            print(f"\t{stat_name}: {stat_value}")

        print("\n")

    # save stats
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"eval_stats.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(eval_stats, f)

    # save video
    if args.save_video:
        save_eval_rollout_vids(eval_pixels, os.path.join(args.save_dir, "videos"))

if __name__ == "__main__":
    main()
