import torch.nn as nn
from Models.eru import EmotionalResidualUnit


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.progressive_blocks = nn.ModuleList()
        self.from_rgb = nn.ModuleList()
        channels = [64, 128, 256, 512, 512, 512]

        # 1. 初始化 from_rgb 层
        for ch in channels:
            self.from_rgb.append(nn.Conv2d(3, ch, 1))

        # 2. 初始化渐进块
        for i in range(len(channels)):
            block = []
            if i > 0:
                block.append(nn.AvgPool2d(2))
            in_ch = channels[i]
            out_ch = channels[min(i + 1, len(channels) - 1)]
            block.extend(
                [
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.LeakyReLU(0.2),
                    EmotionalResidualUnit(out_ch),
                ]
            )
            self.progressive_blocks.append(nn.Sequential(*block))

        # 3. 动态适应输入尺寸
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 关键修改：将任意尺寸池化为 1x1
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, v, a, stage=5):
        features = []
        stage = min(stage, len(self.progressive_blocks) - 1)

        # 1. 初始RGB转换
        x = self.from_rgb[stage](x)

        # 2. 处理从当前 stage 到最高分辨率的块
        for i in range(stage, len(self.progressive_blocks)):
            for layer in self.progressive_blocks[i]:
                if isinstance(layer, nn.AvgPool2d) and x.size(2) <= 2:
                    continue
                if isinstance(layer, EmotionalResidualUnit):
                    x = layer(x, v, a)
                    features.append((x, v, a))
                else:
                    x = layer(x)

        # 3. 最终判断
        validity = self.final(x)
        return validity, features
