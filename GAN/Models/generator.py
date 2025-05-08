import torch
import torch.nn as nn
from .eru import EmotionalResidualUnit


class Generator(nn.Module):
    def __init__(self, z_dim=100, emotion_dim=2):
        super().__init__()
        self.progressive_blocks = nn.ModuleList()
        self.to_rgb = nn.ModuleList()

        # 初始块 (4x4)
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(z_dim + emotion_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.progressive_blocks.append(EmotionalResidualUnit(512))
        self.to_rgb.append(nn.Conv2d(512, 3, 3, padding=1))

        # 渐进增加分辨率
        for i in range(1, 6):  # 8x8 → 128x128
            in_ch = 512 // (2 ** (i - 1))
            out_ch = in_ch // 2
            self.progressive_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    EmotionalResidualUnit(out_ch),
                )
            )
            self.to_rgb.append(nn.Conv2d(out_ch, 3, 3, padding=1))

    def forward(self, z, v, a, stage=5):  # stage: 0(4x4) → 5(128x128)
        # Expand dimensions of v and a to match z
        v = v.unsqueeze(1)  # [B] -> [B, 1]
        a = a.unsqueeze(1)  # [B] -> [B, 1]

        cond = torch.cat([z, v, a], dim=1).unsqueeze(-1).unsqueeze(-1)
        x = self.initial(cond)

        for i in range(stage + 1):
            block = self.progressive_blocks[i]

            # 统一处理所有块
            if isinstance(block, EmotionalResidualUnit):
                # 直接调用 ERU 单元
                x = block(x, v, a)
            else:
                # 遍历 Sequential 中的每一层
                for layer in block:
                    if isinstance(layer, EmotionalResidualUnit):
                        x = layer(x, v, a)
                    else:
                        x = layer(x)

        return torch.tanh(self.to_rgb[stage](x))
