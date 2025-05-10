import torch.nn as nn
from Models.eru import EmotionalResidualUnit


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.progressive_blocks = nn.ModuleList()
        self.from_rgb = nn.ModuleList()
        channels = [64, 128, 256, 512, 512, 512]

        # 1. Initialize the from_rgb layers
        for ch in channels:
            self.from_rgb.append(nn.Conv2d(3, ch, 1))

        # 2. Initialize progressive blocks
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

        # 3. Dynamically adapt to input size
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Key modification: Pool any size to 1x1
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, v, a, stage=5):
        features = []
        stage = min(stage, len(self.progressive_blocks) - 1)

        # 1. Initial RGB conversion
        x = self.from_rgb[stage](x)

        # 2. Process blocks from the current stage to the highest resolution
        for i in range(stage, len(self.progressive_blocks)):
            for layer in self.progressive_blocks[i]:
                if isinstance(layer, nn.AvgPool2d) and x.size(2) <= 2:
                    continue
                if isinstance(layer, EmotionalResidualUnit):
                    x = layer(x, v, a)
                    features.append((x, v, a))
                else:
                    x = layer(x)

        # 3. Final evaluation
        validity = self.final(x)
        return validity, features
