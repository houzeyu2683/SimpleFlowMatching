import material
import ovum
import torch
import torch.distributed
import os

torch.distributed.init_process_group(backend='nccl')
index = int(os.environ['LOCAL_RANK'])
device = f'cuda:{index}'

hub = material.Hub()
data = hub.getData(
    number=256
)
validation = hub.getValidation(
    number=64,
    reproducibility=False
)

model = ovum.Model(device)
model.activateLayer()
model = torch.nn.parallel.DistributedDataParallel(
    model, device_ids=[index]
)

# model.module.loadWeight("./log/exp-0/checkpoint/5000.pt")

history = './log/exp-5'
framework = ovum.Framework(model, device, history)

snapshot = 1000000
total = -1
accumulation = 1

framework.fitWeight(data, snapshot, total, accumulation, validation)
