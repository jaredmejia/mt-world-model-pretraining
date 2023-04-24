"""Script to take an offline dataset of images as observations and generate a dataset of
hidden states and belief states from an RSSM as the observations."""
import argparse
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
from offline_dreamer import create_custom_env, offline_kitchen_transforms

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
    sub_traj_end_indices = torch.cat(
        (torch.where(dataset_td["done"])[0], torch.tensor([dataset_size - 1]))
    )

    start_idx = 0
    encoded_trajs = []

    for end_idx in tqdm(sub_traj_end_indices):
        curr_traj_td = dataset_td[start_idx : end_idx + 1].clone()

        assert (
            curr_traj_td["done"][-1] == 1 and curr_traj_td["done"].sum() == 1
        ) or end_idx == dataset_size - 1, (
            "Only last elem should be done if not on last trajectory"
        )

        curr_traj_td = curr_traj_td.unsqueeze(0)
        curr_traj_transforms = create_transforms_fn(
            curr_traj_td.shape,
            cfg.image_size,
            cfg.state_dim,
            cfg.rssm_hidden_dim,
        )
        curr_traj_td_transformed = curr_traj_transforms(curr_traj_td)

        cond_wmodel.eval()
        with torch.no_grad():
            curr_traj_cond_wmodel_out = cond_wmodel(
                curr_traj_td_transformed.clone().to(device)
            )

        encoded_trajs.append(curr_traj_cond_wmodel_out.squeeze())
        start_idx = end_idx + 1

    encoded_td = torch.cat(encoded_trajs, dim=0)

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

    # create custom env
    custom_env = create_custom_env(
        "kitchen-complete-v0", render_imgs=True, image_size=cfg.image_size
    )

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
        proof_environment=transformed_env_constructor(
            cfg, stats={"loc": 0.0, "scale": 1.0}, custom_env=custom_env
        )(),
    )

    # load world models
    model_based_env, cond_wmodel = load_wmodels(
        model_based_env, world_model, ckpt_dir
    )

    # load original dataset
    kitchen_transforms = offline_kitchen_transforms(
        (cfg.batch_size, cfg.batch_length),
        cfg.image_size,
        cfg.state_dim,
        cfg.rssm_hidden_dim,
    )
    replay_buffer = SubTrajectoryReplay(
        "kitchen-complete-v0",
        observation_type="image_joints",
        batch_size=cfg.batch_size,
        batch_length=cfg.batch_length,
        transform=kitchen_transforms,
    )
    kitchen_dataset_td = replay_buffer.dataset

    # get encoded tensordict
    kitchen_encoded_td = get_hidden_per_traj(
        kitchen_dataset_td,
        cond_wmodel,
        offline_kitchen_transforms,
        cfg,
        device,
    )
    assert (
        kitchen_encoded_td.shape[0] == kitchen_dataset_td.shape[0]
    ), "Encoded dataset should have same number of elements as original dataset"

    # save encoded tensordict
    os.makedirs(args.out_path, exist_ok=True)
    save_snapshot(
        kitchen_encoded_td.cpu(),
        os.path.join(args.out_path, "kitchen_encoded_td.pt"),
    )

if __name__ == "__main__":
    main()