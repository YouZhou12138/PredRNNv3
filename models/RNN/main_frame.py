
import torch
import torch.nn as nn
from models.RNN.branch import conv_b, trans_b, merge
from models.layers.layer_utils import get_sinusoid_encoding_table


class LSTMCell(nn.Module):
    def __init__(self, num_hidden, num_heads, filter_size, gate_act,
                 drop, attn_drop, drop_path, mlp_ratio, resolution_h, resolution_w,
                 abs_pe=False, DBMM=True,
                ):
        super().__init__()

        self.conv= nn.Conv2d(3 * num_hidden, num_hidden, kernel_size=1, padding=0, bias=False)
        self.conv_batch = nn.Sequential(
            conv_b(num_hidden, filter_size),
            conv_b(num_hidden, filter_size),
           )
        self.abs_pe = abs_pe
        if abs_pe:
            self.pos_embed = nn.Parameter(
                get_sinusoid_encoding_table(resolution_h * resolution_w, num_hidden, T=10000),
                requires_grad=False,
            ).view(1, resolution_h * resolution_w, num_hidden)

        self.trans_batch = nn.Sequential(
            trans_b(num_hidden, input_resolution_h=resolution_h, input_resolution_w=resolution_w,
                    num_heads=num_heads, gate_act=gate_act, abs_pe=abs_pe,
                    drop=drop, attn_drop=attn_drop,drop_path=drop_path, mlp_ratio=mlp_ratio),
            trans_b(num_hidden, input_resolution_h=resolution_h, input_resolution_w=resolution_w,
                    num_heads=num_heads, gate_act=gate_act, abs_pe=abs_pe,
                    drop=drop, attn_drop=attn_drop, drop_path=drop_path, mlp_ratio=mlp_ratio)
            )

        self.conv_o = merge(num_hidden, filter_size, DBMM)
        self.conv_last = merge(num_hidden, filter_size, DBMM)

    def forward(self, x_t, h_t, c_t, m_t):
        b, c, h, w = h_t.shape
        x = self.conv(torch.cat((x_t, h_t, m_t), dim=1))
        x_cnn = self.conv_batch(x)
        if self.abs_pe:
            x_trans = (self.trans_batch(x.flatten(2).movedim(1, 2)+self.pos_embed.to(x.device))
                       .movedim(1, 2).reshape(b, c, h, w))
        else:
            x_trans = self.trans_batch(x.flatten(2).movedim(1, 2)).movedim(1, 2).reshape(b, c, h, w)

        gate_m = torch.sigmoid(x_cnn)
        cell_m = torch.tanh(x_cnn)
        m_new = gate_m * (m_t + cell_m)

        gate_c = torch.sigmoid(x_trans)
        cell_c = torch.tanh(x_trans)
        c_new = gate_c * (c_t + cell_c)

        o_t = torch.sigmoid(x_cnn + x_trans + self.conv_o(m_new, c_new))

        h_new = o_t * torch.tanh(self.conv_last(m_new, c_new))

        return h_new, c_new, m_new









