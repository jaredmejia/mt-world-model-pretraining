from typing import Optional, Sequence

import torch
from torchrl.envs import CenterCrop, Compose, ObservationTransform, ToTensorImage
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec
from torchrl.envs.transforms import TensorDictPrimer, ObservationNorm

class KitchenFilterState(ObservationTransform):
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
            return obs[..., :9]

        @_apply_to_composite
        def transform_observation_spec(self, observation_spec):
              return BoundedTensorSpec(
                    minimum=-1.0,
                    maximum=1.0,
                    shape=torch.zeros(9).shape,
                    dtype=observation_spec.dtype,
                    device=observation_spec.device,
              )
        
class MetaWorldFilterState(ObservationTransform):
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
            return obs[..., :4].to(torch.float32)

        @_apply_to_composite
        def transform_observation_spec(self, observation_spec):
              return BoundedTensorSpec(
                    minimum=-1.0,
                    maximum=1.0,
                    shape=torch.zeros(4).shape,
                    dtype=torch.float32,
                    device=observation_spec.device,
              )
        
class MetaWorldConvertToFloat(ObservationTransform):
        def __init__(
            self,
            in_keys: Optional[Sequence[str]] = None,
            out_keys: Optional[Sequence[str]] = None,
            in_keys_inv: Optional[Sequence[str]] = None,
            out_keys_inv: Optional[Sequence[str]] = None,
        ):
            if in_keys is None:
                in_keys = ["action"]
            if out_keys is None:
                out_keys = ["action"]

            super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)

        def _apply_transform(self, obs: torch.Tensor):
            return obs.to(torch.float32)

        @_apply_to_composite
        def transform_observation_spec(self, observation_spec):
              return BoundedTensorSpec(
                    minimum=observation_spec.minimum.to(torch.float32),
                    maximum=observation_spec.maximum.to(torch.float32),
                    shape=observation_spec.shape,
                    dtype=torch.float32,
                    device=observation_spec.device,
              )
        

def fill_dreamer_hidden_keys(batch_size, state_dim, hidden_dim):
    default_hidden_transform = TensorDictPrimer(primers={'state': UnboundedContinuousTensorSpec(
                    shape=torch.Size([*batch_size, state_dim]), dtype=torch.float32), 'belief': UnboundedContinuousTensorSpec(
                    shape=torch.Size([*batch_size, hidden_dim]), dtype=torch.float32)}, default_value=0, random=False)
    return default_hidden_transform


def get_env_transforms(env_name, image_size, from_pixels=True, train_type='iql', batch_size=None, state_dim=None, hidden_dim=None, obs_stats=None):
    
    state_keys = ["observation", ("next", "observation")]
    if from_pixels:
        pixel_keys =["pixels", ("next", "pixels")]
    else:
        pixel_keys = None
        
    
    transforms = Compose()
    
    if train_type == 'dreamer':
        if isinstance(batch_size, int):
            batch_size = (batch_size,)

        dreamer_transform = fill_dreamer_hidden_keys(batch_size, state_dim, hidden_dim)
        
        transforms.append(dreamer_transform)

        if obs_stats is not None:
            obs_norm = ObservationNorm(**obs_stats, in_keys=["pixels"])
            transforms.append(obs_norm)

    if 'kitchen' in env_name:

        env_transforms = (
            ToTensorImage(in_keys=pixel_keys, out_keys=pixel_keys), CenterCrop(image_size, in_keys=pixel_keys, out_keys=pixel_keys),
            KitchenFilterState(in_keys=state_keys, out_keys=state_keys),
        )

    else: # metaworld
        env_transforms = (
            ToTensorImage(in_keys=pixel_keys, out_keys=pixel_keys), CenterCrop(image_size, in_keys=pixel_keys, out_keys=pixel_keys),
            MetaWorldFilterState(in_keys=state_keys, out_keys=state_keys),
            MetaWorldConvertToFloat(in_keys=["action", ("next", "action")], out_keys=["action", ("next", "action")]),
        )
         
    for env_transform in env_transforms:
        transforms.append(env_transform)

    return transforms


