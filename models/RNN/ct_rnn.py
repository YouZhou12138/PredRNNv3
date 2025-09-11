import argparse
from typing import Optional, Union

import torch
import torch.nn as nn

from models.RNN.main_frame import LSTMCell
from utils.tools import str2bool


class CT_RNN(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.patch_size = hparams.patch_size
        self.in_channel = hparams.n_image
        self.out_channel = hparams.n_out
        self.num_layers = hparams.num_layers
        self.num_hidden = hparams.n_hidden
        self.proj = nn.Conv2d(self.in_channel,  self.num_hidden,
                              kernel_size=self.patch_size, stride=self.patch_size )

        self.resolution_h = int(hparams.input_size_h / self.patch_size)
        self.resolution_w = int(hparams.input_size_w / self.patch_size)
        self.norm = nn.LayerNorm([self.num_hidden, self.resolution_h, self.resolution_w])

        cell_list = []

        for i in range(self.num_layers):
            cell_list.append(
                LSTMCell(self.num_hidden, hparams.n_heads,
                         hparams.filter_size, hparams.gate_act, hparams.drop,
                         attn_drop=hparams.attn_drop, drop_path=hparams.drop_path,
                         mlp_ratio=hparams.mlp_ratio, resolution_h=self.resolution_h,
                         resolution_w=self.resolution_w, abs_pe=hparams.abs_pe,DBMM=hparams.DBMM)
            )
        self.cell_list = nn.ModuleList(cell_list)

        self.conv_last = nn.ConvTranspose2d(self.num_hidden, self.out_channel,
                                            kernel_size=self.patch_size, stride=self.patch_size)


    @staticmethod
    def add_model_specific_args(
        parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None
    ):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        parser.add_argument("--context_length", type=int, default=10)
        parser.add_argument("--target_length", type=int, default=10)
        parser.add_argument("--val_target_length", type=int, default=10)
        parser.add_argument("--patch_size", type=int, default=4)
        parser.add_argument("--input_size_h", type=int, default=64)
        parser.add_argument("--input_size_w", type=int, default=64)
        parser.add_argument("--n_image", type=int, default=1)
        parser.add_argument("--n_hidden", type=int, default=128)
        parser.add_argument("--n_out", type=int, default=1)
        parser.add_argument("--n_heads", type=int, default=4)
        parser.add_argument("--mlp_ratio", type=int, default=4)
        parser.add_argument("--drop", type=float, default=0.0)
        parser.add_argument("--attn_drop", type=float, default=0.0)
        parser.add_argument("--drop_path", type=float, default=0.0)
        #----------------------Ablation experiment------------------------------
        parser.add_argument("--gate_act", type=str, default="silu")
        parser.add_argument("--abs_pe", type=str2bool, default=False)
        parser.add_argument("--DBMM", type=str2bool, default=True)
        parser.add_argument("--filter_size", type=int, default=3)
        parser.add_argument("--num_layers", type=int, default=4)

        return parser

    def forward(self, frames, sampling=None):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        b, t, c, h, w = frames.shape
        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([b, self.num_hidden, self.resolution_h, self.resolution_w]).to(frames.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([b, self.num_hidden, self.resolution_h, self.resolution_w]).to(frames.device)
        target_length = self.hparams.target_length if self.training else self.hparams.val_target_length
        for i in range(self.hparams.context_length + target_length - 1):
            if i < self.hparams.context_length:
                if sampling and i > 0:
                    proba = (torch.rand(b) < sampling[0]).type_as(frames)[:, None, None, None]
                    net = proba * frames[:, i] + (1 - proba) * x_gen
                else:
                    net = frames[:, i]
            else:
                if sampling:
                    proba = (torch.rand(b) < sampling[1]).type_as(frames)[:, None, None, None]
                    net = proba * frames[:, i] + (1 - proba) * x_gen
                else:
                    net = x_gen

            net = self.norm(self.proj(net))
            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=0).transpose(0, 1).contiguous()

        return next_frames, {}

if __name__ == "__main__":
    arg = CT_RNN.add_model_specific_args()
    args = arg.parse_args()
    model = CT_RNN(args)
    data = torch.rand(2, 20, 1, 64, 64)
    model.eval()
    preds, aux = model(data)
    print("of")