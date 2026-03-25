import waterhen
import torch
import torch.optim

condition = torch.randn(
    1,
    3,
    64,
    64
)
chaos = torch.randn(
    1,   # batch size
    26,  # number of frame
    3, # channel
    64,  # height
    64   # width
)
step = torch.randint(0, 1000, [1])
model = waterhen.Model(device='cuda')
model.activateLayer()
reception = model.getReception(condition, chaos, step)


