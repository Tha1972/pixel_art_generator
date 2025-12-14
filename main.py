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


""" U-Net Architecture

Input: 16x16x3

Encoder:    
EL1(16x16x3) -> EL2(8x8x32) -> EL3(4x4x64)

Decoder:
DL1(4x4x64) -> DL2(8x8x32) -> DL3(16x16x3)

"""

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Downsample (Encoder)
        self.e_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.e_down1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)

        self.e_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.e_down2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)

        # Upsample (Decoder)
        self.d_conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.d_up1 = nn.Upsample(scale_factor=2)

        self.d_conv2 = nn.Conv2d(in_channels=96, out_channels=3, kernel_size=3, padding=1)
        self.d_up2 = nn.Upsample(scale_factor=2)

        # Output
        self.out = nn.Conv2d(in_channels=35, out_channels=3, kernel_size=1)

        # Time Embeddings
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )

    def forward(self, x, t):
        # Encoder
        e1 = F.silu(self.e_conv1(x))         
        x = F.silu(self.e_down1(e1))        

        e2 = F.silu(self.e_conv2(x))         
        x = F.silu(self.e_down2(e2))   

        # Time embeddings
        t = t.float().unsqueeze(1)    
        t_emb = self.time_mlp(t)
        x = x + t_emb[:, :, None, None]

        # Decoder
        x = F.silu(self.d_conv1(x))       
        x = self.d_up1(x)                  

        x = torch.cat([x, e2], dim=1)      
        x = F.silu(self.d_conv2(x))      

        x = self.d_up2(x)                 
        x = torch.cat([x, e1], dim=1)  

        x = self.out(x)       

        return x
        

class ImageFolderDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
        ]

        self.image_paths = self.image_paths[:2000]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = np.array(img)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = img * 2 - 1            # normalize to [-1, 1]

        return img


def q_sample(x0, t, scheduler):
    noise = torch.randn_like(x0)

    sqrt_ab = scheduler.sqrt_alpha_cum_prod[t][:, None, None, None]
    sqrt_omab = scheduler.sqrt_one_minus_alpha_cum_prod[t][:, None, None, None]

    x_t = sqrt_ab * x0 + sqrt_omab * noise
    return x_t, noise

def diffusion_loss(model, x0, scheduler):
    B = x0.size(0)
    device = x0.device

    t = torch.randint(0, scheduler.num_timesteps, (B,), device=device)

    x_t, noise = q_sample(x0, t, scheduler)
    pred_noise = model(x_t, t)

    return F.mse_loss(pred_noise, noise)

def train_diffusion(
    model,
    dataloader,
    epochs,
    lr=1e-4,
    num_timesteps=100,
    device="cpu"
):
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = LinearNoiseScheduler(
        num_timesteps=num_timesteps,
        beta_start=1e-4,
        beta_end=0.02
        )

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


def get_image():
    img = Image.open("data/images/images/image_88627.JPEG")
    arr = np.array(img)
    return arr


model = UNet()

dataset = ImageFolderDataset("data/images/images")
dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    drop_last=True
)

print("Starting train")
train_diffusion(
    model=model,
    dataloader=dataloader,
    epochs=50,
    lr=1e-4,
    num_timesteps=100,
    device="cpu"   # or "cuda"
)

