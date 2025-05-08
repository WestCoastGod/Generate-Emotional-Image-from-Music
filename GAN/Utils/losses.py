import torch
import torch.nn as nn
import torch.nn.functional as F


class AFMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_features, fake_features):
        """
        real/fake_features: 判别器返回的ERU特征列表
        每个元素是 (feature_map, v, a)
        """
        loss_v = 0
        loss_a = 0

        for (real_f, real_v, real_a), (fake_f, fake_v, fake_a) in zip(
            real_features, fake_features
        ):
            # 计算情绪门控差异
            B, C, H, W = real_f.shape
            real_v_map = real_v.view(B, 1, 1, 1).expand(B, 1, H, W)
            real_a_map = real_a.view(B, 1, 1, 1).expand(B, 1, H, W)
            fake_v_map = fake_v.view(B, 1, 1, 1).expand(B, 1, H, W)
            fake_a_map = fake_a.view(B, 1, 1, 1).expand(B, 1, H, W)

            # 情绪权重差异 (L1距离)
            loss_v += F.l1_loss(
                torch.sigmoid(real_f[:, :1]),  # 真实图像的v权重
                torch.sigmoid(fake_f[:, :1]),  # 生成图像的v权重
            )
            loss_a += F.l1_loss(
                torch.sigmoid(real_f[:, 1:2]),  # 真实图像的a权重
                torch.sigmoid(fake_f[:, 1:2]),  # 生成图像的a权重
            )

        return loss_v / len(real_features), loss_a / len(real_features)


class WGAN_GPLoss(nn.Module):
    def __init__(self, lambda_gp=10):
        super().__init__()
        self.lambda_gp = lambda_gp

    def gradient_penalty(self, D, real_imgs, fake_imgs, v, a):
        """WGAN-GP梯度惩罚"""
        alpha = torch.rand(real_imgs.size(0), 1, 1, 1).to(real_imgs.device)
        interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(
            True
        )
        d_interpolates, _ = D(interpolates, v, a)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp

    def forward(self, D, real_imgs, fake_imgs, v, a):
        gp = self.gradient_penalty(D, real_imgs, fake_imgs, v, a)
        return gp
