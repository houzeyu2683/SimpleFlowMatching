import torch
import tensordict

class Module(torch.nn.Module):

    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device
        return

    def activateLayer(
        self,
        schema: dict,
    ) -> bool:
        channel, embedding = schema['embedding']
        head = schema['head']
        layer = torch.nn.ModuleDict(
            {
                '(1) norm': torch.nn.GroupNorm(
                    32, channel, eps=1e-6, affine=True
                ),
                '(1) projection': torch.nn.Linear(channel, embedding),
                '(2) attention': torch.nn.TransformerEncoderLayer(
                    d_model=embedding,
                    nhead=head,
                    dim_feedforward=embedding*4,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                ),
                '(2) projection': torch.nn.Linear(embedding, channel),
            }
        )
        self.layer = layer.to(self.device)
        return(True)

    def getState(
        self,
        activation: torch.Tensor
    ) -> torch.Tensor:
        activation = activation.to(non_blocking=True, device=self.device)
        #
        identity = activation
        value = self.layer['(1) norm'](activation)
        value = torch.permute(value, (0, 2, 3, 1))
        shape = value.shape
        value = value.flatten(1, 2)
        value = self.layer['(1) projection'](value) # (B, H*W, C)
        value = self.layer['(2) attention'](value)
        value = self.layer['(2) projection'](value) # (B, H*W, C)
        value = torch.cat([value])
        value = value.reshape(shape)
        value = torch.permute(value, (0, 3, 1, 2))
        state = (identity + value) / 2**0.5
        return(state)

    forward = getState


class Bottleneck:

    def __init__(self, device: str) -> None:
        self.device = device
        return

    def getModule(
        self, schema: dict
    ) -> Module:
        module = Module(device=self.device)
        module.activateLayer(schema)
        return(module)

    pass

mode = '!Debug'
if(mode == 'Debug'):
    chaos = torch.randn(
        3,   # batch size
        128, # channel
        16,  # height
        16   # width
    )
    schema = {
        'embedding': (128, 64),
        'head': 1
    }
    bottleneck = Bottleneck(device='cuda')
    module = bottleneck.getModule(schema)
    state = module(chaos)
    print(state.shape)  # (2, 8, 256, 16, 16)
    print("No Error")
    pass
