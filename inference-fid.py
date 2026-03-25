import os
import torchvision
import waterhen
import torch

version_dir = 'log/exp/version'
output_root = './output-fid'
total       = 1000
level       = 50
device      = 'cuda'

checkpoints = [
    'log/exp/version/500000.pt',
    'log/exp/version/1000000.pt',
    'log/exp/version/1500000.pt',
    'log/exp/version/2000000.pt',
]

model = waterhen.Model(device=device)
model.activateLayer()
model.eval()

for weight_path in checkpoints:
    name = os.path.splitext(os.path.basename(weight_path))[0]
    output_path = os.path.join(output_root, name)
    os.makedirs(output_path, exist_ok=True)

    weight = torch.load(weight_path)
    weight = {k.replace('module.', '', 1): v for k, v in weight.items()}
    weight.pop('n_averaged', None)
    model.load_state_dict(weight)

    print(f'Generating {total} images for checkpoint: {name}')
    for index in range(total):
        invention = model.getInvention(batch=1, level=level)  # (1, 3, 64, 64)
        image = (invention.clamp(-1, 1) + 1.0) / 2.0
        path = os.path.join(output_path, f'{index:04d}.png')
        torchvision.utils.save_image(image, path)
        continue

    print(f'Saved {total} images to {output_path}/')
    continue

# python -m pytorch_fid output-fid/100000 ../Data\ Bucket/IoAF/other/


# 500000: FID:  219.36525708504107
# 1000000: FID:  217.49352725217588
# 1500000: FID:  215.7979603395022
# 2000000: FID: 
