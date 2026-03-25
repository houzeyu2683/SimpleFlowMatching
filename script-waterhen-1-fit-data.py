import material
import waterhen

hub = material.Hub()
data = hub.getData(
    number=8
)
validation = hub.getValidation(
    number=1,
    reproducibility=False
)

device = 'cuda'
model = waterhen.Model(device)
model.activateLayer()
# model.loadWeight("./log/exp-0/checkpoint/5000.pt")

history = './log/exp'
framework = waterhen.Framework(model, device, history)
# framework.saveCheckpoint('./testcpt.pt')
# model.loadWeight("./testcpt.pt")


snapshot = 100000
total = -1
accumulation = 4
# schema = {
#     'cycle': 488766,
#     'ratio': 1.2,
#     'boundary': (1e-4, 5*1e-4),
#     'decay': 1.0,
#     'acceleration': 488766
# }
framework.fitWeight(data, snapshot, total, accumulation, validation)
