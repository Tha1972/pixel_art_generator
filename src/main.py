import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from PIL import Image
import numpy as np
from noise_scheduler import LinearNoiseScheduler
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader
from helpers import diffusion_loss
from model import UNet, train_diffusion
from data_loader import ImageFolderDataset
import matplotlib.pyplot as plt
import math


@torch.no_grad()
def sample(
    model,
    scheduler,
    generated_image=None,
    image_size=16,
    device="mps",
):
    model.eval()

    if generated_image is not None:
        x = generated_image.to(device)
    else:
        x = torch.randn(1, 3, image_size, image_size, device=device)

    for t in reversed(range(scheduler.num_timesteps)):
        t_tensor = torch.tensor([t], device=device)

        pred_noise = model(x, t_tensor)

        alpha = scheduler.alphas[t]
        alpha_bar = scheduler.alpha_cum_prod[t]
        beta = scheduler.betas[t]

        x = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise
        )

        if t > 0:
            noise = torch.randn_like(x)
            noise = noise * 1.2
            x = x + torch.sqrt(beta) * noise

    return x

def generate_images(imgage_count):
    images = []

    for _ in range(imgage_count):
        sampled = sample(model, scheduler, image_size=16, device=device)

        img = sampled.squeeze(0)
        img = (img + 1) / 2
        img = img.clamp(0, 1)
        img = img.permute(1, 2, 0).cpu().numpy()

        images.append(img)

    return images


def train_and_save_model():
    model = UNet()

    dataset = ImageFolderDataset("data/images/images")
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True
    )

    train_diffusion(
        model=model,
        dataloader=dataloader,
        epochs=50,
        lr=1e-4,
        num_timesteps=100,
        device="cuda"  
    )

    torch.save(model.state_dict(), "models/unet_diffusion.pth")

if __name__ == "__main__":
    # train_and_save_model()
  
    device = "cuda" 

    model = UNet()
    model.load_state_dict(torch.load("models/unet_diffusion.pth", map_location=device))
    model.to(device)

    scheduler = LinearNoiseScheduler(
        num_timesteps=100,
        beta_start=1e-4,
        beta_end=0.02
        )

    images = generate_images(4)

    n_cols = 4
    n_rows = math.ceil(len(images) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")

    for ax in axes[len(images):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()