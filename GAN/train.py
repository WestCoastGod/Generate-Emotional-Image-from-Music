import torch
from Models.generator import Generator
from Models.discriminator import Discriminator
from Utils.dataloader import EmotionDataset
from Utils.losses import AFMLoss, WGAN_GPLoss
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F


def train(
    img_dir,
    label_csv,
    epochs=1711,
    batch_size=16,
    lr=0.001,
    lambda_gp=10,
    device=None,
):
    # Initialize device
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    G = Generator().to(device)
    D = Discriminator().to(device)

    # Optimizers
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss functions
    afm_loss = AFMLoss()
    wgan_gp = WGAN_GPLoss(lambda_gp=lambda_gp)

    # Data loading
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset = EmotionDataset(img_dir, label_csv, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    best_loss_G = float("inf")  # Initialize the best generator loss

    # Training loop
    for epoch in range(epochs):
        for i, (real_imgs, v, a) in enumerate(dataloader):
            real_imgs, v, a = real_imgs.to(device), v.to(device), a.to(device)
            z = torch.randn(real_imgs.size(0), 100).to(device)

            # Determine current stage for progressive growing
            stage = min(epoch // 10, 5)  # Floor division to determine stage

            # Downsample real images to match the current stage resolution
            resolution = 4 * (2**stage)  # Calculate resolution based on stage
            real_imgs = F.interpolate(
                real_imgs,
                size=(resolution, resolution),
                mode="bilinear",
                align_corners=False,
            )

            # --- Train Discriminator ---
            D.train()
            G.eval()

            with torch.no_grad():
                fake_imgs = G(z, v, a, stage=stage)

            D_real, real_features = D(real_imgs, v, a, stage=stage)
            D_fake, fake_features = D(fake_imgs.detach(), v, a, stage=stage)

            real_features = [
                (rf.detach(), v.detach(), a.detach()) for (rf, v, a) in real_features
            ]
            fake_features = [
                (ff.detach(), v.detach(), a.detach()) for (ff, v, a) in fake_features
            ]

            loss_adv = -torch.mean(D_real) + torch.mean(D_fake)
            loss_gp = wgan_gp(D, real_imgs, fake_imgs, v, a)
            loss_v, loss_a = afm_loss(real_features, fake_features)
            loss_D = loss_adv + loss_gp + 100 * (loss_v + loss_a)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # --- Train Generator ---
            D.eval()
            G.train()

            fake_imgs = G(z, v, a, stage=stage)
            D_fake, fake_features = D(fake_imgs, v, a, stage=stage)

            with torch.no_grad():
                _, real_features = D(real_imgs, v, a, stage=stage)

            loss_adv = -torch.mean(D_fake)
            loss_v, loss_a = afm_loss(real_features, fake_features)
            loss_G = loss_adv + 100 * (loss_v + loss_a)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # Save the best generator model
            if loss_G.item() < best_loss_G:
                best_loss_G = loss_G.item()
                torch.save(
                    G.state_dict(),
                    r"C:\Users\cxoox\Desktop\AIST3110_Project\GAN\Weights\generator_best.pth",
                )

            # Logging
            if i % 50 == 0:
                print(
                    f"Epoch {epoch}, Batch {i}: D_loss={loss_D.item():.3f}, G_loss={loss_G.item():.3f}"
                )

    # Save final models
    torch.save(
        G.state_dict(),
        r"C:\Users\cxoox\Desktop\AIST3110_Project\GAN\Weights\generator_final.pth",
    )
    torch.save(
        D.state_dict(),
        r"C:\Users\cxoox\Desktop\AIST3110_Project\GAN\Weights\discriminator_final.pth",
    )


if __name__ == "__main__":
    # Paths
    img_dir = r"C:\Users\cxoox\Desktop\AIST3110_Project\GAN\Data\Landscape"
    label_csv = r"C:\Users\cxoox\Desktop\AIST3110_Project\GAN\Data\Landscape\valence_arousal.csv"

    # Train the model
    print("Starting training...")
    train(img_dir, label_csv)
    print("Training complete.")
