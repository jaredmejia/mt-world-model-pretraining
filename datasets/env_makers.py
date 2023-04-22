from torchrl.envs import CenterCrop, Compose, EnvCreator, ParallelEnv, ToTensorImage, TransformedEnv
from torchrl.envs.libs.gym import GymEnv, GymWrapper
from .transforms import KitchenFilterState, MetaWorldFilterState

def env_maker(env_name, frame_skip=1, device="cpu", from_pixels=False):
    if 'kitchen' in env_name:
        import d4rl
        import gym
        custom_env = gym.make(env_name, render_imgs=from_pixels)
        custom_env = GymWrapper(custom_env, frame_skip=frame_skip, from_pixels=from_pixels )
    
    else:
        import metaworld

        mt10 = metaworld.MT10()

        custom_env = mt10.train_classes[env_name]()
        task = [task for task in mt10.train_tasks if task.env_name == env_name][0]
        custom_env.set_task(task)

        original_render = custom_env.render
        # override the render method for training_env
        def render(self, mode='human', width=64, height=64):
            return original_render(mode)
            
        custom_env.render = render.__get__(custom_env, type(custom_env))
        # custom_env.camera_name = 'corner2'
        custom_env = GymWrapper(custom_env, frame_skip=frame_skip, from_pixels=from_pixels )

    if from_pixels:
        pixel_keys =["pixels", ("next", "pixels")]
        state_keys = ["observation", ("next", "observation")]
        
        if 'kitchen' in env_name:
            env_transforms = Compose(
                ToTensorImage(in_keys=pixel_keys, out_keys=pixel_keys), CenterCrop(96, in_keys=pixel_keys, out_keys=pixel_keys),
                KitchenFilterState(in_keys=state_keys, out_keys=state_keys),
            )
        else: # metaworld
            env_transforms = Compose(
                ToTensorImage(in_keys=pixel_keys, out_keys=pixel_keys), CenterCrop(96, in_keys=pixel_keys, out_keys=pixel_keys),
                MetaWorldFilterState(in_keys=state_keys, out_keys=state_keys),
            )

        custom_env = TransformedEnv(custom_env, env_transforms)
        
        return custom_env

    return GymEnv(
        env_name, device=device, frame_skip=frame_skip, from_pixels=from_pixels
    )