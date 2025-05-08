import torch
import torch.nn as nn


class EmotionalResidualUnit(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 情绪门控分支
        self.conv_v = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, 3, padding=1), nn.Sigmoid()
        )
        self.conv_a = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, 3, padding=1), nn.Sigmoid()
        )
        # 最终融合
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels + 2, in_channels, 3, padding=1), nn.Tanh()
        )

    def forward(self, x, v, a):
        B, C, H, W = x.shape
        v_map = v.view(B, 1, 1, 1).expand(B, 1, H, W)
        a_map = a.view(B, 1, 1, 1).expand(B, 1, H, W)

        # 计算门控权重
        v_gate = self.conv_v(torch.cat([x, v_map], dim=1))
        a_gate = self.conv_a(torch.cat([x, a_map], dim=1))

        # 特征融合
        m = (v_gate * x) + (a_gate * x)
        y = self.final_conv(torch.cat([m, v_map, a_map], dim=1))
        return y + x  # 残差连接
