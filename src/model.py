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


""" U-Net Architecture

Input: 16x16x3

Encoder:
EL1 (16x16x3)   -> (16x16x64)
Down1           -> (8x8x64)
EL2 (8x8x64)    -> (8x8x128)
Down2           -> (4x4x128)

Bottleneck:
BL  (4x4x128)   -> (4x4x256)

Decoder:
Up1             -> (8x8x256)
DL1 (8x8x256+128) -> (8x8x128)
Up2             -> (16x16x128)
DL2 (16x16x128+64) -> (16x16x64)

Output:
OL  (16x16x64)  -> (16x16x3)

"""
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Time embeddings
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        self.down1 = nn.Conv2d(64, 64, 4, stride=2, padding=1)   

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        self.down2 = nn.Conv2d(128, 128, 4, stride=2, padding=1)

        # Bottleneck
        self.mid = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")   
        self.dec1 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest") 
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )

        # Output
        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, x, t):
        # Timestep embedding
        t = t.float().unsqueeze(1)
        t_emb = self.time_mlp(t)

        # Encoder
        e1 = self.enc1(x)
        e1 = e1 + t_emb[:, :64, None, None]

        x = self.down1(e1)

        e2 = self.enc2(x)
        e2 = e2 + t_emb[:, :128, None, None]

        x = self.down2(e2)

        # Bottleneck
        x = self.mid(x)
        x = x + t_emb[:, :256, None, None]

        # Decoder
        x = self.up1(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec1(x)
        x = x + t_emb[:, :128, None, None]

        x = self.up2(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec2(x)
        x = x + t_emb[:, :64, None, None]

        return self.out(x)
        


def train_diffusion(
    model,
    dataloader,
    epochs,
    lr=1e-4,
    num_timesteps=100,
    device="cuda"
):
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = LinearNoiseScheduler(
        num_timesteps=num_timesteps,
        beta_start=1e-4,
        beta_end=0.02
        )
    
    scheduler.betas = scheduler.betas.to(device)
    scheduler.alphas = scheduler.alphas.to(device)
    scheduler.alpha_cum_prod = scheduler.alpha_cum_prod.to(device)
    scheduler.sqrt_alpha_cum_prod = scheduler.sqrt_alpha_cum_prod.to(device)
    scheduler.sqrt_one_minus_alpha_cum_prod = scheduler.sqrt_one_minus_alpha_cum_prod.to(device)

    for epoch in range(epochs):
        total_loss = 0.0

        for i, images in enumerate(dataloader):
            images = images.to(device)

            loss = diffusion_loss(model, images, scheduler)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.6f}")