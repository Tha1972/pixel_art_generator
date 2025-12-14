import torch 
import torch.nn.functional as F
from PIL import Image
import numpy as np


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


def get_image(path):
    img = Image.open(path)
    arr = np.array(img)
    return arr
