from typing import Optional, Sequence

import torch
from torchrl.envs import ObservationTransform
from torchrl.envs.transforms.transforms import _apply_to_composite
from torchrl.data import BoundedTensorSpec


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
            return obs[..., :4]

        @_apply_to_composite
        def transform_observation_spec(self, observation_spec):
              return BoundedTensorSpec(
                    minimum=-1.0,
                    maximum=1.0,
                    shape=torch.zeros(4).shape,
                    dtype=observation_spec.dtype,
                    device=observation_spec.device,
              )