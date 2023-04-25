"""Script to train an IQL agent using the hidden states of a pretrained RSSM as observations."""

from datetime import datetime
import hydra
import os
import sys

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.cuda
from torch import nn, optim
import tqdm

from tensordict import TensorDict
from torchrl.envs import CenterCrop, Compose,EnvCreator, ParallelEnv, ToTensorImage
from torchrl.envs.transforms import TensorDictPrimer
from torchrl.data import  UnboundedContinuousTensorSpec
from torchrl.envs.utils import set_exploration_mode
from torchrl.objectives import SoftUpdate
from torchrl.objectives.iql import IQLLoss
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.trainers.helpers.models import make_dreamer


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grandparent_dir = os.path.dirname(parent_dir)

sys.path.append(os.path.join(parent_dir, "iql"))
from iql_offline import get_actor, get_critic, get_value, get_q_val_estimate, get_value_estimate

sys.path.append(os.path.join(parent_dir, "dreamer"))
from gen_rollout import load_wmodels
from dreamer_utils import transformed_env_constructor

sys.path.append(grandparent_dir)
from datasets import RSSMStateReplayBuffer, env_maker, get_env_transforms


def iql_rssm_rollout(env, policy, cond_wmodel, num_steps, device, fill_hidden_keys):
    # TODO: parallelize this
    env_td = env.reset()
    env_td = env_td.to(device)

    input_td = env_td.clone().unsqueeze(0)
    input_td = fill_hidden_keys(input_td)
    input_td['observation'] = torch.cat((input_td['belief'].clone(), input_td['state'].clone()), dim=1)
    input_td[('next', 'pixels')] = input_td['pixels'].clone()

    rollout = []
    for i in range(num_steps):
        input_td = input_td.to(device)

        # get action prediction
        iql_actor_out = policy(input_td.clone())
        
        # update belief and state
        cond_wmodel_out = cond_wmodel(iql_actor_out.clone())

        # save rollout
        rollout.append(cond_wmodel_out.clone().select('done', 'reward', 'observation', 'action', 'next'))

        # step env
        env_td = env.step(cond_wmodel_out.clone()[0].select('action'))

        # update input_td
        input_td = TensorDict({}, batch_size=1)

        input_td['next', 'pixels'] = env_td[('next', 'pixels')].clone().unsqueeze(0)
        input_td['done'] = env_td['next', 'done'].clone().unsqueeze(0)
        input_td['reward'] = env_td['next', 'reward'].clone().unsqueeze(0)

        input_td['observation'] = torch.cat((cond_wmodel_out.clone()['next', 'belief'], cond_wmodel_out.clone()['next', 'state']), dim=1)
        input_td['belief'] = cond_wmodel_out.clone()['next', 'belief']
        input_td['state'] = cond_wmodel_out.clone()['next', 'state']

        if input_td['done'].item():
            break

    rollout_td = torch.cat(rollout, dim=0)

    return rollout_td


def get_eval_env(env_name, from_pixels, image_size):
    pixel_keys =["pixels", ("next", "pixels")]
    env_transforms = Compose(ToTensorImage(in_keys=pixel_keys, out_keys=pixel_keys), CenterCrop(image_size, in_keys=pixel_keys, out_keys=pixel_keys))
    eval_env = env_maker(env_name=env_name, from_pixels=from_pixels, image_size=image_size, env_transforms=env_transforms)

    return eval_env


def prep_wmodel(ckpt_dir, device):
    # load config
    cfg_path = os.path.join(ckpt_dir, "config.yaml")
    dreamer_cfg = OmegaConf.load(cfg_path)

    # load sample env
    custom_env = env_maker(env_name=dreamer_cfg.env_name, from_pixels=dreamer_cfg.from_pixels, image_size=dreamer_cfg.image_size, env_transforms=None)

    # get model
    (
        world_model,
        model_based_env,
        actor_model,
        value_model,
        policy,
    ) = make_dreamer(
        obs_norm_state_dict=None,
        cfg=dreamer_cfg,
        device=device,
        use_decoder_in_env=True,
        action_key="action",
        value_key="state_value",
        proof_environment=transformed_env_constructor(
            dreamer_cfg, stats={"loc": 0.0, "scale": 1.0}, custom_env=custom_env
        )(),
    )
    model_based_env, cond_wmodel = load_wmodels(
        model_based_env, world_model, ckpt_dir
    )

    return cond_wmodel, dreamer_cfg


@hydra.main(version_base=None, config_path=".", config_name="multitask_iql_rssm")
def main(cfg: "DictConfig"):  # noqa: F821

    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        and torch.cuda.device_count() > 0
        and cfg.device == "cuda:0"
        else torch.device("cpu")
    )

    exp_name = generate_exp_name("Offline_IQL-RSSM", cfg.exp_name)
    logger = get_logger(
        logger_type=cfg.logger,
        logger_name="iql_rssm_logging",
        experiment_name=exp_name,
        wandb_kwargs={"mode": cfg.mode, "entity": cfg.entity, "project": f"{cfg.env_task}offline_iql_rssm"},
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # set environment and transforms
    env_transforms = get_env_transforms(
        cfg.env_name,
        cfg.image_size,
        from_pixels=cfg.from_pixels,
        train_type='iql',
        multitask= cfg.env_task == "multitask",
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
    test_env = env_factory(num_workers=5)
    num_actions = test_env.action_spec.shape[-1]

    in_keys = ["observation"]

    # Create Agent
    actor = get_actor(cfg, test_env, num_actions, in_keys)

    # Create Critic
    qvalue = get_critic(cfg.critic_dropout, in_keys)

    # Create Value
    value = get_value(cfg.value_dropout, in_keys)

    model = nn.ModuleList([actor, qvalue, value]).to(device)
    
    # Make Replay Buffer
    print("Creating Replay Buffer...")
    replay_buffer = RSSMStateReplayBuffer(cfg.data_snapshot_path, batch_size=cfg.batch_size)
    print("Replay Buffer Created!")

    # init nets
    with torch.no_grad():
        td = replay_buffer.sample(cfg.batch_size)
        td = td.to(device)
        actor(td)
        qvalue(td)
        value(td)

    del td
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

    # Optimizers
    params = list(loss_module.parameters())
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # RSSM
    wmodel_ckpt_path = os.path.join(hydra.utils.get_original_cwd(), cfg.wmodel_ckpt_path)
    cond_wmodel, dreamer_cfg = prep_wmodel(wmodel_ckpt_path, device)

    # Eval items
    if cfg.env_task != "multitask":
        fill_dreamer_hidden_keys = TensorDictPrimer(primers={'state': UnboundedContinuousTensorSpec(
                shape=torch.Size([1, dreamer_cfg.state_dim]), dtype=torch.float32), 'belief': UnboundedContinuousTensorSpec(
                shape=torch.Size([1, dreamer_cfg.rssm_hidden_dim]), dtype=torch.float32)}, default_value=0, random=False)
        eval_env = get_eval_env(cfg.env_name, cfg.from_pixels, cfg.image_size)
    
    else:
        print("Skipping evaluations for multitask training")

    # Main loop
    target_net_updater.init_()

    loss = None

    ckpt_steps = [10000, 50000, 100000, 200000, 250000, 300000]
    save_path = os.path.join(
        'ckpts', 
        f'{cfg.env_task}-iql-rssm', 
        f'{cfg.exp_name}'
        f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}',
    )

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

        if i % cfg.eval_interval == 0 :
            
            if cfg.env_task != "multitask":
                with set_exploration_mode("mean"), torch.no_grad():
                    # evaluating in sequence for now
                    rollout_td_list = []
                    for _ in range(cfg.env_per_collector):
                        rollout_td = iql_rssm_rollout(eval_env, model[0], cond_wmodel, cfg.max_frames_per_traj, device, fill_dreamer_hidden_keys)
                        rollout_td_list.append(rollout_td)
                    
                    # log reward
                    eval_reward = sum([rollout_td['reward'].sum().item() for rollout_td in rollout_td_list]) / cfg.env_per_collector
                    logger.log_scalar("test_reward", eval_reward, step=i)
        
            # log q-value estimates
            q_value_avg, q_value_max = get_q_val_estimate(sampled_tensordict, loss_module, device)
            logger.log_scalar("q_value_avg", q_value_avg, step=i)
            logger.log_scalar("q_value_max", q_value_max, step=i)

            # log value estimates
            value_avg, value_max = get_value_estimate(sampled_tensordict, loss_module, device)
            logger.log_scalar("value_avg", value_avg, step=i)
            logger.log_scalar("value_max", value_max, step=i)
        
        if i in ckpt_steps or i == cfg.max_steps:
            os.makedirs(save_path, exist_ok=True)
            torch.save(model[0].state_dict(), os.path.join(save_path, f'actor_{i}.pth'))
            torch.save(model[1].state_dict(), os.path.join(save_path, f'qvalue_{i}.pth'))
            torch.save(model[2].state_dict(), os.path.join(save_path, f'value_{i}.pth'))


if __name__ == "__main__":
    main()
