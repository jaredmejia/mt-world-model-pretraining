"""Script to take an offline dataset of images as observations and generate a dataset of
hidden states and belief states from an RSSM as the observations."""
import argparse
from functools import partial
import os
import sys

import torch
import torchsnapshot
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict
from torchrl.trainers.helpers.models import make_dreamer
from tqdm import tqdm

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "./dreamer",
    )
)
from dreamer_utils import transformed_env_constructor
from gen_rollout import load_wmodels
from datasets import env_maker, get_env_transforms

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
)
from datasets import SubTrajectoryReplay


def get_hidden_per_traj(
    dataset_td, cond_wmodel, create_transforms_fn, cfg, device
):
    dataset_size = dataset_td.shape[0]
    traj_end_indices = torch.cat(
        (torch.where(dataset_td["done"])[0], torch.tensor([dataset_size - 1]))
    )

    keep_keys = list(dataset_td.keys(include_nested=True))
    keep_keys = [key for key in keep_keys if key != "next"]
    keep_keys.extend(["state", "belief", ("next", "state"), ("next", "belief")])

    start_idx = 0
    encoded_trajs = []

    dataset_td = dataset_td.to(device)

    for end_idx in tqdm(traj_end_indices):
        curr_traj_td = dataset_td[start_idx : end_idx + 1].clone()

        assert (
            curr_traj_td["done"][-1] == 1 and curr_traj_td["done"].sum() == 1
        ) or end_idx == dataset_size - 1, (
            "Only last elem should be done if not on last trajectory"
        )

        curr_traj_td = curr_traj_td.unsqueeze(0)
        curr_traj_transforms = create_transforms_fn(batch_size=curr_traj_td.shape)
        curr_traj_td_transformed = curr_traj_transforms(curr_traj_td)

        cond_wmodel.eval()
        with torch.no_grad():
            curr_traj_cond_wmodel_out = cond_wmodel(
                curr_traj_td_transformed.clone()
            ).select(*keep_keys)

        encoded_trajs.append(curr_traj_cond_wmodel_out.cpu().squeeze())
        start_idx = end_idx + 1

    encoded_td = torch.cat(encoded_trajs, dim=0)
    encoded_td_size = encoded_td.shape[0]
    encoded_traj_end_indices = torch.cat(
        (torch.where(encoded_td["done"])[0], torch.tensor([encoded_td_size - 1]))
    )
    assert torch.equal(
        encoded_traj_end_indices, traj_end_indices
    ), "Trajectory end indices should be the same for encoded and original dataset"


    return encoded_td


def save_snapshot(tensordict, save_path):
    state = {"state": tensordict}
    snapshot = torchsnapshot.Snapshot.take(app_state=state, path=save_path)
    print(f"Saved snapshot to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        default="../dreamer/ckpts/kitchen-complete-v0-offline-dreamer-2023-04-21-19-25-38",
    )
    parser.add_argument("--out_path", type=str, required=True, default="./encoded_data")
    args = parser.parse_args()

    # load config
    ckpt_dir = args.ckpt_dir
    cfg_path = os.path.join(ckpt_dir, "config.yaml")
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
    get_transforms_per_traj = partial(get_env_transforms, *env_transforms_args, train_type='dreamer', **env_transforms_kwargs)

    # get proof env
    proof_env = env_maker(env_name=cfg.env_name, from_pixels=cfg.from_pixels, image_size=cfg.image_size, env_transforms=proof_env_transforms.clone())

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the different components of dreamer
    (
        world_model,
        model_based_env,
        actor_model,
        value_model,
        policy,
    ) = make_dreamer(
        obs_norm_state_dict=None,
        cfg=cfg,
        device=device,
        use_decoder_in_env=True,
        action_key="action",
        value_key="state_value",
        proof_environment=proof_env,
    )

    # load world models
    model_based_env, cond_wmodel = load_wmodels(
        model_based_env, world_model, ckpt_dir
    )

    # load original dataset
    replay_buffer = SubTrajectoryReplay(
        cfg.env_name, 
        observation_type="image_joints", 
        batch_size=cfg.batch_size,
        batch_length=cfg.batch_length,
        base_transform=base_env_transforms,
        sample_transform=buffer_sample_transforms,
        multitask= cfg.env_task == 'multitask',
    )
    orig_dataset_td = replay_buffer.dataset

    # get encoded tensordict
    encoded_dataset_td = get_hidden_per_traj(
        orig_dataset_td,
        cond_wmodel,
        get_transforms_per_traj,
        cfg,
        device,
    )
    assert (
        encoded_dataset_td.shape[0] == orig_dataset_td.shape[0]
    ), "Encoded dataset should have same number of elements as original dataset"

    print(f'Original Dataset: {orig_dataset_td}')
    print(f'Encoded Dataset: {encoded_dataset_td}')

    # save encoded tensordict
    os.makedirs(args.out_path, exist_ok=True)
    save_snapshot(
        encoded_dataset_td.cpu(),
        os.path.join(args.out_path, "encoded_dataset_td.pt"),
    )

if __name__ == "__main__":
    main()