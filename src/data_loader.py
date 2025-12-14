
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch 
import os
import pandas as pd

class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, label_selected):

        label_map = {
            1: "[1. 0. 0. 0. 0.]",
            2: "[0. 1. 0. 0. 0.]",
            3: "[0. 0. 1. 0. 0.]",
            4: "[0. 0. 0. 1. 0.]",
            5: "[0. 0. 0. 0. 1.]",
        }

        selected_label = label_map[label_selected]

        df = pd.read_csv("data/labels.csv")
        first_label_paths = df[df["Label"] == selected_label]["Image Path"].tolist()
        filtered_paths = [i.split("/")[2].split(".")[0] for i in first_label_paths]

        self.image_paths = [
                    os.path.join(image_dir, f)
                    for f in os.listdir(image_dir)
                    if os.path.join(image_dir, f).split("\\")[1].split(".")[0] in filtered_paths
                ]

        print(f"Total images found: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = np.array(img)

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img = img * 2 - 1     

        return img
