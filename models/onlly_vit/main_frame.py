
import torch
import torch.nn as nn
from models.RNN.branch import trans_b


class LSTMCell(nn.Module):
    def __init__(self, num_hidden, num_heads, gate_act,
                 drop, attn_drop, drop_path, mlp_ratio, resolution_h, resolution_w,
                ):
        super().__init__()

        self.linear= nn.Linear(2 * num_hidden, num_hidden, bias=False)

        self.trans_batch = nn.Sequential(
            trans_b(num_hidden, input_resolution_h=resolution_h, input_resolution_w=resolution_w,
                    num_heads=num_heads, gate_act=gate_act, abs_pe=False,
                    drop=drop, attn_drop=attn_drop,drop_path=drop_path, mlp_ratio=mlp_ratio),
            trans_b(num_hidden, input_resolution_h=resolution_h, input_resolution_w=resolution_w,
                    num_heads=num_heads, gate_act=gate_act, abs_pe=False,
                    drop=drop, attn_drop=attn_drop, drop_path=drop_path, mlp_ratio=mlp_ratio)
            )

        self.linear_o = nn.Linear(num_hidden, num_hidden, bias=True)
        self.linear_last = nn.Linear(num_hidden, num_hidden, bias=True)

    def forward(self, x_t, h_t, c_t):
        x = self.linear(torch.cat((x_t, h_t), dim=2))
        x_trans = self.trans_batch(x)

        gate_c = torch.sigmoid(x_trans)
        cell_c = torch.tanh(x_trans)
        c_new = gate_c * (c_t + cell_c)

        o_t = torch.sigmoid(x_trans + self.linear_o(c_new))

        h_new = o_t * torch.tanh(self.linear_last(c_new))

        return h_new, c_new








