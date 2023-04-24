import torch
from torch import nn


class PixelVecObsDecoder(nn.Module):
    def __init__(self, vec_dim, depth=32):
        super().__init__()

        self.state_to_latent = nn.Sequential(
            nn.LazyLinear(depth * 8 * 2 * 2),
            nn.ReLU(),
        )
        self.pixel_decoder = nn.Sequential(
            nn.LazyConvTranspose2d(depth * 4, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(depth * 4, depth * 2, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(depth * 2, depth, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(depth, 3, 6, stride=2),
        )
        self.vec_decoder = nn.Sequential(
            nn.LazyLinear(depth * 4),
            nn.ReLU(),
            nn.LazyLinear(vec_dim),
        )


    def forward(self, state, rnn_hidden):
        latent = self.state_to_latent(torch.cat([state, rnn_hidden], dim=-1))
        *batch_sizes, D = latent.shape
        
        # decode pixels
        pixel_decode_input = latent.view(-1, D, 1, 1)
        pixels_decoded = self.pixel_decoder(pixel_decode_input)
        _, C, H, W = pixels_decoded.shape
        pixels_decoded = pixels_decoded.view(*batch_sizes, C, H, W)

        # decode vec
        vec_decoded = self.vec_decoder(latent)
        vec_decoded = vec_decoded.view(*batch_sizes, -1)

        return pixels_decoded, vec_decoded
