import os
import torchvision
import ovum
import torch
import pathlib

result = './result/'
level  = 50    # Euler steps (flow matching)
weight = 'log/exp-5/checkpoint/10000.pt'
device = 'cuda'

# Load model
model = ovum.Model(device=device)
model.activateLayer()
model.loadWeight(weight)
model.eval()

# Inference
invention = model.getInvention(number=8, level=level)  # (32, 3, 64, 64)

folder = pathlib.Path(result)
os.makedirs(folder, exist_ok=True)
torchvision.utils.save_image(
    invention, folder / 'batch.png'
)