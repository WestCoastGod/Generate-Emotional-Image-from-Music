import torch
from Models.generator import Generator
import matplotlib.pyplot as plt


def generate_emotion_image(v, a, save_path=r"C:\Users\cxoox\Desktop\GAN\image.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator().to(device)
    G.load_state_dict(
        torch.load(
            r"C:\Users\cxoox\Desktop\AIST3110_Project\GAN\Weights\generator_best.pth"
        )
    )
    G.eval()

    z = torch.randn(1, 100).to(device)
    v_tensor = torch.tensor([v]).float().to(device)
    a_tensor = torch.tensor([a]).float().to(device)

    with torch.no_grad():
        img = G(z, v_tensor, a_tensor, stage=5)
        img = img.squeeze().permute(1, 2, 0).cpu().numpy()
        img = (img * 0.5 + 0.5) * 255  # [-1,1] → [0,255]

    plt.imsave(save_path, img.astype("uint8"))


# Sample
generate_emotion_image(v=0.8, a=0.3, save_path=r"C:\Users\cxoox\Desktop\image.png")
