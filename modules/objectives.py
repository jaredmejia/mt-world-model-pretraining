
import torch
from tensordict import TensorDict
from torchrl.objectives.dreamer import DreamerModelLoss
from torchrl.objectives.utils import distance_loss


class PixelVecDreamerModelLoss(DreamerModelLoss):
    """World Model loss for Dreamer with pixel and vector observations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, tensordict: TensorDict) -> torch.Tensor:
        tensordict = tensordict.clone(recurse=False)
        tensordict.rename_key_(("next", "reward"), ("next", "true_reward"))
        tensordict = self.world_model(tensordict)

        # compute model loss
        kl_loss = self.kl_loss(
            tensordict.get(("next", "prior_mean")),
            tensordict.get(("next", "prior_std")),
            tensordict.get(("next", "posterior_mean")),
            tensordict.get(("next", "posterior_std")),
        ).unsqueeze(-1)

        # pixel reconstruction loss
        pixel_reco_loss = distance_loss(
            tensordict.get(("next", "pixels")),
            tensordict.get(("next", "reco_pixels")),
            self.reco_loss,
        )
        if not self.global_average:
            pixel_reco_loss = pixel_reco_loss.sum((-3, -2, -1))
        pixel_reco_loss = pixel_reco_loss.mean().unsqueeze(-1)

        # vector reconstruction loss
        vector_reco_loss = distance_loss(
            tensordict.get(("next", "observation")),
            tensordict.get(("next", "reco_vec")),
            self.reco_loss,
        )
        vector_reco_loss = vector_reco_loss.mean().unsqueeze(-1)

        reco_loss = pixel_reco_loss + vector_reco_loss

        reward_loss = distance_loss(
            tensordict.get(("next", "true_reward")),
            tensordict.get(("next", "reward")),
            self.reward_loss,
        )

        if not self.global_average:
            reward_loss = reward_loss.squeeze(-1)

        reward_loss = reward_loss.mean().unsqueeze(-1)

        return (
            TensorDict(
                {
                    "loss_model_kl": self.lambda_kl * kl_loss,
                    "loss_model_reco": self.lambda_reco * reco_loss,
                    "loss_model_reward": self.lambda_reward * reward_loss,
                },
                [],
            ),
            tensordict.detach(),
        )
