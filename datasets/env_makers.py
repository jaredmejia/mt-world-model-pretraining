from torchrl.envs import TransformedEnv
from torchrl.envs.libs.gym import GymWrapper

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
        def render(self, mode='human', width=image_size, height=image_size):
            return original_render(mode, resolution=(width, height))
            
        custom_env.render = render.__get__(custom_env, type(custom_env))
        custom_env = GymWrapper(custom_env, frame_skip=frame_skip, from_pixels=from_pixels)

    # env_transforms = get_env_transforms(env_name, from_pixels=from_pixels, **transform_kwargs)
    custom_env = TransformedEnv(custom_env, env_transforms)
    
    return custom_env
