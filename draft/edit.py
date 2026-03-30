import torch
import waterhen
import perception
import tensordict

device = 'cuda'
model = waterhen.Model(device)
model.activateLayer()
# model
weight = torch.load("./600000.pt")

response = model.load_state_dict(weight, strict=False)
response.missing_keys
response.unexpected_keys
print(response)

batch = tensordict.TensorDict(
    {
        'condition': torch.randn(1, 3, 64, 64), # 參考的 frame
        'chaos': torch.randn(1, 25, 3, 64, 64), # 加入噪音的 frames
        'step': torch.randint(0, 1000, (1,))
    },
    device=device
)


recv2 = model.getReceptionV2(batch['condition'], batch['chaos'], batch['step'])

'''
frame: (B, L, C, H, W)
chunk: (B*L, C, H, W)
activation: (B, L, C, H', W')

activation: (B, L, C, H', W')
sequence: (B*H*W', L, C)
activation: (B*L, C, H'*W')
activation: (B, L, C, H', W')
'''

