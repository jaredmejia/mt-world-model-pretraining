# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import uuid
from datetime import datetime

import hydra
import torch.cuda
from hydra.core.config_store import ConfigStore
# import gym
from typing import Optional, Sequence
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from torchrl.envs import ObservationTransform, TransformedEnv

from torchrl.envs import EnvCreator, ParallelEnv
from torchrl.envs.transforms import RewardScaling

from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.data import BoundedTensorSpec
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import OrnsteinUhlenbeckProcessWrapper
from torchrl.record import VideoRecorder
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.trainers.helpers.collectors import (
    make_collector_offpolicy,
    OffPolicyCollectorConfig,
)
from torchrl.envs.libs.gym import GymEnv, GymWrapper

from torchrl.trainers.helpers.envs import (
    correct_for_frame_skip,
    EnvConfig,
    initialize_observation_norm_transforms,
    parallel_env_constructor,
    retrieve_observation_norms_state_dict,
    transformed_env_constructor,
)
from torchrl.trainers.helpers.logger import LoggerConfig
from torchrl.trainers.helpers.losses import LossConfig, make_sac_loss
from torchrl.trainers.helpers.models import make_sac_model, SACModelConfig
from torchrl.trainers.helpers.replay_buffer import make_replay_buffer, ReplayArgsConfig
from torchrl.trainers.helpers.trainers import make_trainer, TrainerConfig
from torchrl.collectors import SyncDataCollector

class MT10Wrapper(GymWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self._env = env

    # def step(self, action):
    #     obs, reward, done, info = self.env.step(action)
    #     return obs, reward, done, info

    # def reset(self):
    #     return self._env.reset()
    
class MetaWorldTransform(ObservationTransform):
        def __init__(
            self,
            in_keys: Optional[Sequence[str]] = None,
            out_keys: Optional[Sequence[str]] = None,
            in_keys_inv: Optional[Sequence[str]] = None,
            out_keys_inv: Optional[Sequence[str]] = None,
        ):
            if in_keys is None:
                in_keys = ["observation"]
            if out_keys is None:
                out_keys = ["observation"]

            super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)

        def _apply_transform(self, obs: torch.Tensor):
            return obs.squeeze().float()

        @_apply_to_composite
        def transform_observation_spec(self, observation_spec):
            #   return BoundedTensorSpec(
            #         minimum=observation_spec.minimum,
            #         maximum=observation_spec.maximum,
            #         shape=torch.zeros(39).shape,
            #         dtype=observation_spec.dtype,
            #         device=observation_spec.device,
            #   )
            observation_spec.shape = torch.zeros(39).shape
            observation_spec.dtype = torch.float32
            return observation_spec
            


config_fields = [
    (config_field.name, config_field.type, config_field)
    for config_cls in (
        TrainerConfig,
        OffPolicyCollectorConfig,
        EnvConfig,
        LossConfig,
        SACModelConfig,
        LoggerConfig,
        ReplayArgsConfig,
    )
    for config_field in dataclasses.fields(config_cls)
]

Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
cs = ConfigStore.instance()
cs.store(name="config", node=Config)

DEFAULT_REWARD_SCALING = {
    "Hopper-v1": 5,
    "Walker2d-v1": 5,
    "HalfCheetah-v1": 5,
    "cheetah": 5,
    "Ant-v2": 5,
    "Humanoid-v2": 20,
    "humanoid": 100,
}


@hydra.main(version_base=None, config_path="./", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821

    cfg = correct_for_frame_skip(cfg)

    if not isinstance(cfg.reward_scaling, float):
        cfg.reward_scaling = DEFAULT_REWARD_SCALING.get(cfg.env_name, 5.0)

    device = (
        torch.device("cpu")
        if torch.cuda.device_count() == 0
        else torch.device("cuda:0")
    )

    exp_name = "_".join(
        [
            "SAC",
            cfg.exp_name,
            str(uuid.uuid4())[:8],
            datetime.now().strftime("%y_%m_%d-%H_%M_%S"),
        ]
    )

    exp_name = generate_exp_name("SAC", cfg.exp_name)
    logger = get_logger(
        logger_type=cfg.logger, logger_name="sac_logging", experiment_name=exp_name
    )
    video_tag = exp_name if cfg.record_video else ""

    key, init_env_steps, stats = None, None, None
    if not cfg.vecnorm and cfg.norm_stats:
        if not hasattr(cfg, "init_env_steps"):
            raise AttributeError("init_env_steps missing from arguments.")
        key = ("next", "pixels") if cfg.from_pixels else ("next", "observation_vector")
        init_env_steps = cfg.init_env_steps
        stats = {"loc": None, "scale": None}
    elif cfg.from_pixels:
        stats = {"loc": 0.5, "scale": 0.5}

    # proof_env = transformed_env_constructor(
    #     cfg=cfg,
    #     use_env_creator=False,
    #     stats=stats,
    # )()
    # create meta world environment based on cfg.env_name
    # door_open_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[cfg.env_name]
    # proof_env = MT10Wrapper(door_open_goal_observable_cls())
    def env_maker(env_name):
        mt_env =  GymWrapper(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]())
        mt_env = TransformedEnv(mt_env, MetaWorldTransform(in_keys=["observation", ("next", "observation")], out_keys=["observation", ("next", "observation")]))
        # custom_env.done_spec = mt_env.done_spec
        # custom_env.output_spec = mt_env.output_spec
        return mt_env

    def env_factory(num_workers):
        """Creates an instance of the environment."""

        # 1.2 Create env vector
        vec_env = ParallelEnv(
            create_env_fn=EnvCreator(lambda: env_maker(cfg.env_name)),
            num_workers=num_workers,
        )

        return vec_env
    
    # proof_env = env_factory(1)
    proof_env = env_maker(cfg.env_name)
    # initialize_observation_norm_transforms(
    #     proof_environment=proof_env, num_iter=init_env_steps, key=key
    # )
    # _, obs_norm_state_dict = retrieve_observation_norms_state_dict(proof_env)[0]
    
    model = make_sac_model(
        proof_env,
        cfg=cfg,
        device=device,
        in_keys=["observation"],
    )
    loss_module, target_net_updater = make_sac_loss(model, cfg)

    actor_model_explore = model[0]
    if cfg.ou_exploration:
        actor_model_explore = OrnsteinUhlenbeckProcessWrapper(
            actor_model_explore,
            annealing_num_steps=cfg.annealing_frames,
            sigma=cfg.ou_sigma,
            theta=cfg.ou_theta,
        ).to(device)
    if device == torch.device("cpu"):
        # mostly for debugging
        actor_model_explore.share_memory()

    if cfg.gSDE:
        with torch.no_grad(), set_exploration_mode("random"):
            # get dimensions to build the parallel env
            proof_td = actor_model_explore(proof_env.reset().to(device))
        action_dim_gsde, state_dim_gsde = proof_td.get("_eps_gSDE").shape[-2:]
        del proof_td
    else:
        action_dim_gsde, state_dim_gsde = None, None
    proof_env.close()
    # create_env_fn = parallel_env_constructor(
    #     cfg=cfg,
    #     obs_norm_state_dict=None,
    #     action_dim_gsde=action_dim_gsde,
    #     state_dim_gsde=state_dim_gsde,
    # )

    # collector = make_collector_offpolicy(
    #     make_env=create_env_fn,
    #     actor_model_explore=actor_model_explore,
    #     cfg=cfg,
    #     # make_env_kwargs=[
    #     #     {"device": device} if device >= 0 else {}
    #     #     for device in args.env_rendering_devices
    #     # ],
    # )
        # Make Off-Policy Collector
    collector = SyncDataCollector(
        env_factory,
        create_env_kwargs={"num_workers": cfg.env_per_collector},
        policy=model[0],
        frames_per_batch=cfg.frames_per_batch,
        max_frames_per_traj=cfg.max_frames_per_traj,
        total_frames=cfg.total_frames,
        device='cpu',
    )
    collector.set_seed(cfg.seed)

    replay_buffer = make_replay_buffer(device, cfg)

    # recorder = transformed_env_constructor(
    #     cfg,
    #     video_tag=video_tag,
    #     norm_obs_only=True,
    #     obs_norm_state_dict=obs_norm_state_dict,
    #     logger=logger,
    #     use_env_creator=False,
    # )()
    # if isinstance(create_env_fn, ParallelEnv):
    #     raise NotImplementedError("This behaviour is deprecated")
    # elif isinstance(create_env_fn, EnvCreator):
    #     recorder.transform[1:].load_state_dict(create_env_fn().transform.state_dict())
    # elif isinstance(create_env_fn, TransformedEnv):
    #     recorder.transform = create_env_fn.transform.clone()
    # else:
    #     raise NotImplementedError(f"Unsupported env type {type(create_env_fn)}")
    # if logger is not None and video_tag:
    #     recorder.insert_transform(0, VideoRecorder(logger=logger, tag=video_tag))

    # reset reward scaling, as it was just overwritten by state_dict load
    # for t in recorder.transform:
    #     if isinstance(t, RewardScaling):
    #         t.scale.fill_(1.0)
    #         t.loc.fill_(0.0)

    trainer = make_trainer(
        collector,
        loss_module,
        None,
        target_net_updater,
        actor_model_explore,
        replay_buffer,
        logger,
        cfg,
    )

    final_seed = collector.set_seed(cfg.seed)
    print(f"init seed: {cfg.seed}, final seed: {final_seed}")

    trainer.train()
    return (logger.log_dir, trainer._log_dict)


if __name__ == "__main__":
    main()