import os
import torchvision
import waterhen
import torch

weight_path = 'log/exp/version/2000000.pt'
output_path = './output'
level       = 50    # Euler steps (flow matching)
device      = 'cuda'

# Load model
model = waterhen.Model(device=device)
model.activateLayer()

weight = torch.load(weight_path)
weight = {k.replace('module.', '', 1): v for k, v in weight.items()}
weight.pop('n_averaged', None)
model.load_state_dict(weight)
model.eval()

# Inference
invention = model.getInvention(batch=32, level=level)  # (32, 3, 64, 64)

# Save output images
os.makedirs(output_path, exist_ok=True)
# images = (invention.clamp(-1, 1) + 1.0) / 2.0        # [-1,1] → [0,1]
# for index, image in enumerate(images):
#     path = os.path.join(output_path, f'{index:03d}.png')
#     torchvision.utils.save_image(image, path)
#     continue
# print(f"Saved {len(images)} images to {output_path}/")

batch = (invention.clamp(-1, 1) + 1.0) / 2.0        # [-1,1] → [0,1]
path = os.path.join(output_path, f'batch.png')
image = torchvision.utils.make_grid(
    batch, normalize=True, value_range=(0, 1)
)
torchvision.utils.save_image(image, path)