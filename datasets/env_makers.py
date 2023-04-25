import torch
from torchrl.data import BinaryDiscreteTensorSpec, BoundedTensorSpec
from torchrl.envs import TransformedEnv
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.gym_like import default_info_dict_reader

def env_maker(env_name, frame_skip=1, from_pixels=False, image_size=64, env_transforms=None):
    if 'kitchen' in env_name:
        import d4rl
        import gym
        custom_env = gym.make(env_name, render_imgs=from_pixels)
        custom_env = GymWrapper(custom_env, frame_skip=frame_skip, from_pixels=from_pixels)
    
    else:
        import metaworld

        mt10 = metaworld.MT10()

        custom_env = mt10.train_classes[env_name]()
        task = [task for task in mt10.train_tasks if task.env_name == env_name][0]
        custom_env.set_task(task)

        original_render = custom_env.render
        # override the render method for training_env
        def render(self, mode='rgb_array', width=128, height=96):
            return original_render(mode, resolution=(width, height))
            
        custom_env.render = render.__get__(custom_env, type(custom_env))
        custom_env = GymWrapper(custom_env, frame_skip=frame_skip, from_pixels=from_pixels)
        custom_env.action_spec = BoundedTensorSpec(
                    minimum=custom_env.action_spec.space.minimum.to(torch.float32),
                    maximum=custom_env.action_spec.space.maximum.to(torch.float32),
                    shape=custom_env.action_spec.shape,
                    dtype=torch.float32,
                    device=custom_env.action_spec.device,
        )

    if env_transforms is not None:
        custom_env = TransformedEnv(custom_env, env_transforms)
    
    return custom_env


def eval_metaworld_env_maker(mt10, env_name, env_task, env_transforms=None, seed=47):
    env = mt10.train_classes[env_name]()
    env.set_task(env_task)
    env.seed(seed)
    
    original_render = env.render
    # override the render method for training_env
    def render(self, mode='rgb_array', width=128, height=96):
        return original_render(mode, resolution=(width, height))
        
    env.render = render.__get__(env, type(env))

    # adding success reader from dict
    success_reader = default_info_dict_reader(["success"], {'success': BinaryDiscreteTensorSpec(n=1, shape=torch.Size(), dtype=torch.bool)})
    env = GymWrapper(env, frame_skip=1, from_pixels=True).set_info_dict_reader(info_dict_reader=success_reader)

    if env_transforms is not None:
        env = TransformedEnv(env, env_transforms)

    return env
