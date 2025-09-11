import argparse
from typing import Optional, Union

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from models.PredFormer_TS.main_frame import PredFormerLayer


def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')

class PredFormer_Model(nn.Module):
    def __init__(self, hparams,):
        super().__init__()
        self.image_height = hparams.height
        self.image_width = hparams.width
        self.patch_size = hparams.patch_size
        self.num_patches_h = self.image_height // self.patch_size
        self.num_patches_w = self.image_width // self.patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.num_frames_in = hparams.pre_seq
        self.dim = hparams.dim
        self.num_channels = hparams.num_channels
        self.heads = hparams.heads
        self.dim_head = hparams.dim_head
        self.dropout = hparams.dropout
        self.attn_dropout = hparams.attn_dropout
        self.drop_path = hparams.drop_path
        self.scale_dim = hparams.scale_dim
        self.Ndepth = hparams.Ndepth  # Ensure this is defined
        self.depth = hparams.depth  # Ensure this is defined

        assert self.image_height % self.patch_size == 0, 'Image height must be divisible by the patch size.'
        assert self.image_width % self.patch_size == 0, 'Image width must be divisible by the patch size.'
        self.patch_dim = self.num_channels * self.patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size),
            nn.Linear(self.patch_dim, self.dim),
        )
        self.pos_embedding = nn.Parameter(sinusoidal_embedding(self.num_frames_in * self.num_patches, self.dim),
                                          requires_grad=False).view(1, self.num_frames_in, self.num_patches, self.dim)

        self.blocks = nn.ModuleList([
            PredFormerLayer(self.dim, self.depth, self.heads, self.dim_head, self.dim * self.scale_dim, self.dropout,
                            self.attn_dropout, self.drop_path)
            for _ in range(self.Ndepth)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_channels * self.patch_size ** 2)
        )

    @staticmethod
    def add_model_specific_args(
            parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None
    ):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        parser.add_argument("--pre_seq", type=int, default=4)
        parser.add_argument("--patch_size", type=int, default=4)
        parser.add_argument("--height", type=int, default=32)
        parser.add_argument("--width", type=int, default=32)
        parser.add_argument("--num_channels", type=int, default=2)
        parser.add_argument("--dim", type=int, default=256)
        parser.add_argument("--dim_head", type=int, default=32)
        parser.add_argument("--heads", type=int, default=8)
        parser.add_argument("--scale_dim", type=int, default=4)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--attn_dropout", type=float, default=0.1)
        parser.add_argument("--drop_path", type=float, default=0.1)
        parser.add_argument("--depth", type=int, default=1)
        parser.add_argument("--Ndepth", type=int, default=4)

        return parser

    def forward(self, x):
        x = x[:, :self.num_frames_in,...]
        B, T, C, H, W = x.shape

        # Patch Embedding
        x = self.to_patch_embedding(x)

        # Posion Embedding
        x += self.pos_embedding.to(x.device)

        # PredFormer Encoder
        for blk in self.blocks:
            x = blk(x)

        # MLP head
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.view(B, T, self.num_patches_h, self.num_patches_w, C, self.patch_size, self.patch_size)
        x = x.permute(0, 1, 4, 2, 5, 3, 6).reshape(B, T, C, H, W)

        return x, {}

if __name__ == "__main__":
    arg = PredFormer_Model.add_model_specific_args()
    args = arg.parse_args()
    model = PredFormer_Model(args)
    data = torch.rand(2, 8, 2, 32, 32)
    model.eval()
    preds, aux = model(data)
    print("of")
