import torch
from timm.layers import DropPath, trunc_normal_
from torch import nn

from models.RNN.window_atten import WindowAttention
from models.layers.attention import Attention
from models.layers.blocks import Gate, Mlp


class conv_b(nn.Module):
    def __init__(self, num_hidden, filter_size, reduction=16,):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, padding=filter_size // 2, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, padding=filter_size // 2, bias=False),
        )
        self.conv2 =nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_hidden, num_hidden // reduction , kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_hidden // reduction, num_hidden, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = x1 * self.conv2(x1)
        return x + x2


class trans_b(nn.Module):
    def __init__(self, num_hidden, input_resolution_h, input_resolution_w, num_heads,
                 gate_act="gelu", abs_pe=False,
                 drop=0., attn_drop=0.,drop_path=0.,
                 norm_layer=nn.LayerNorm, mlp_ratio=4, ):
        super().__init__()
        self.abs_pe= abs_pe
        if self.abs_pe:
            self.attn = Attention(
                num_hidden, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop
            )
        else:
            self.attn = WindowAttention(
                num_hidden, window_size=(input_resolution_h, input_resolution_w),
                num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop
            )

        self.drop = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(num_hidden)
        self.norm2 = norm_layer(num_hidden)
        # If the program throws an error during the testing phase, please change "self.norm" to "self.norm3".
        self.norm = norm_layer(num_hidden)
        if gate_act == "silu":
            self.gate = Gate(num_hidden, mlp_ratio * num_hidden, drop=drop)
            self.apply(self._init_weights)
        elif gate_act == "gelu":
            self.gate = Mlp(num_hidden, mlp_ratio * num_hidden, drop=drop)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.gate(self.norm2(x)))
        return self.norm(x)


class merge(nn.Module):
    def __init__(self, num_hidden, filter_size, DBMM=True):
        super().__init__()
        self.DBMM = DBMM
        if DBMM:
            self.conv_cnn = nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, padding=filter_size // 2,
                                      bias=False)
            self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(num_hidden*2, num_hidden, kernel_size=1, bias=False)

    def forward(self, x_cnn, x_trans):
        if self.DBMM:
            x = self.conv_cnn(x_cnn)
            x_trans = x_trans * self.sigmoid(x)
            x = torch.cat((x, x_trans), dim=1)
            return self.conv1(x)
        else:
            x = torch.cat((x_cnn, x_trans), dim=1)
            return self.conv1(x)




