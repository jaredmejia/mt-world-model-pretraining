from typing import Optional, Tuple

import torch
from torch import nn

from torchrl.modules import MLP, ConvNet


class LearnedSpatialEmbedding(nn.Module):
    """Learned spatial embedding for cnn outputs, as introduced in PTR paper."""
    def __init__(
            self,
            height: int,
            width: int,
            channels: int,
            num_features: int = 8,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.num_features = num_features
        self.spatial_embedding = nn.Parameter(torch.randn(channels, height, width, num_features), requires_grad=True)

    def forward(self, x):
        bs = x.shape[0]
        assert len(x.shape) == 4, "Input must be of shape (bs, channels, height, width)"

        features = torch.sum(
            x.unsqueeze(-1) * self.spatial_embedding.unsqueeze(0),
            dim=(2, 3)
        )
        features = features.view(bs, -1)

        return features

class PixelVecNet(nn.Module):
    """Encodes pixel input with a CNN, concatenates with vector input and passes through a MLP."""
    
    def __init__(
            self,
            mlp: Optional[nn.Module] = None,
            cnn: Optional[nn.Module] = None,
            learned_spatial_embedding: Optional[nn.Module] = None,
            mlp_kwargs: Optional[dict] = None,
            cnn_kwargs: Optional[dict] = None,
            learned_spatial_embedding_kwargs: Optional[dict] = None,
            include_action: bool = False,
    ):
        super().__init__()
        
        if mlp and mlp_kwargs:
            raise ValueError("Cannot specify both mlp and mlp_kwargs")
        if cnn and cnn_kwargs:
            raise ValueError("Cannot specify both cnn and cnn_kwargs")
        if learned_spatial_embedding and learned_spatial_embedding_kwargs:
            raise ValueError("Cannot specify both learned_spatial_embedding and learned_spatial_embedding_kwargs")
        
        if 'device' in cnn_kwargs and 'device' in mlp_kwargs:
            assert cnn_kwargs['device'] == mlp_kwargs['device'], "CNN and MLP must be on the same device"
        
        if mlp:
            self.mlp = mlp
        else:
            self.mlp = MLP(**mlp_kwargs)

        if cnn:
            self.cnn = cnn
        else:
            self.cnn = ConvNet(**cnn_kwargs)

        if learned_spatial_embedding:
            self.learned_spatial_embedding = learned_spatial_embedding
        elif learned_spatial_embedding_kwargs:
            self.learned_spatial_embedding = LearnedSpatialEmbedding(**learned_spatial_embedding_kwargs)
        else:
            self.learned_spatial_embedding = None

        self.include_action = include_action

    def forward(self, *inputs: Tuple[torch.Tensor]) -> torch.Tensor:
        if self.include_action:
            return self._forward_with_action(*inputs)
        else:
            return self._forward(*inputs)

    def _forward_with_action(self, action_inputs: torch.Tensor, vector_inputs: torch.Tensor, pixel_inputs: torch.Tensor) -> torch.Tensor:
        vector_inputs = torch.cat((vector_inputs, action_inputs), dim=1)
        return self._forward(vector_inputs, pixel_inputs)
    
    def _forward(self, vector_inputs: torch.Tensor, pixel_inputs: torch.Tensor) -> torch.Tensor:
        assert len(pixel_inputs.shape) == 4, "Pixel inputs must be of shape (bs, channels, height, width)"
        assert len(vector_inputs.shape) == 2, "Vector inputs must be of shape (bs, vector_dim)"
        assert pixel_inputs.shape[0] == vector_inputs.shape[0], "Batch size of pixel and vector inputs must be equal"

        bs = pixel_inputs.shape[0]

        # encode pixel inputs
        encoded_pixels = self.cnn(pixel_inputs)

        # spatial embedding
        if self.learned_spatial_embedding is not None:
            encoded_pixels = self.learned_spatial_embedding(encoded_pixels)
        
        encoded_pixels = encoded_pixels.view(bs, -1)

        # concatenate pixel and vector inputs and pass through MLP
        cat_input = torch.cat((encoded_pixels, vector_inputs), dim=1)
        outputs = self.mlp(cat_input)

        return outputs
