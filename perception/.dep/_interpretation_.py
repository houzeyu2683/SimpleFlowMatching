import torch
import tensordict

class Module(torch.nn.Module):

    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device
        return

    def activateLayer(
        self, schema: dict
    ) -> bool:
        channel = schema['channel']
        embedding = schema['embedding']
        layer = {
            '(0) skipped-connection': torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=channel[0],
                    out_channels=channel[1],
                    kernel_size=(1, 1)
                )
            ),
            "(1) convolution": torch.nn.Sequential(
                torch.nn.GroupNorm(
                    num_groups=32, 
                    num_channels=channel[0], 
                    eps=1e-05, 
                    affine=True
                ),
                torch.nn.SiLU(),
                torch.nn.Conv2d(
                    in_channels=channel[0], 
                    out_channels=channel[1], 
                    kernel_size=(3, 3), 
                    stride=(1, 1), 
                    padding=(1, 1)
                ),
                torch.nn.GroupNorm(
                    num_groups=32, 
                    num_channels=channel[1], 
                    eps=1e-05, 
                    affine=True
                )
            ),
            "(1) full-connection": torch.nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Linear(
                    in_features=embedding, 
                    out_features=channel[1]*2
                )
            ),
            "(2) convolution": torch.nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Dropout(p=0.1),
                torch.nn.Conv2d(
                    in_channels=channel[1], 
                    out_channels=channel[1], 
                    kernel_size=(3, 3), 
                    stride=(1, 1), 
                    padding=(1, 1)
                )
            )
        }
        self.layer = torch.nn.ModuleDict(layer).to(self.device)
        return(True)

    def getState(
        self,
        activation: tensordict.TensorDict,
        # chaos: torch.Tensor,
        # step: torch.Tensor,
        # condition: torch.Tensor
    ) -> torch.Tensor:
        activation = activation.to(non_blocking=True, device=self.device)
        chaos = activation['chaos']
        step = activation['step']
        #
        # shape = list(chaos.shape)
        # value = chaos.flatten(0, 1)
        value = self.layer['(0) skipped-connection'](chaos)
        value = value#torch.cat([])
        # shape[-3:] = value.shape[-3:]
        # value = value.reshape(shape)
        identity = value
        #
        feature = {}
        #
        # shape = list(chaos.shape)
        # value = chaos.flatten(0, 1)
        value = self.layer['(1) convolution'](chaos)
        value = value #torch.cat([value])
        # shape[-3:] = value.shape[-3:]
        # value = value.reshape(shape)
        feature['map'] = value
        #
        value = self.layer['(1) full-connection'](step) # B, 2C
        value = value[..., None, None]
        # value = value[:, None, ...]
        feature['term'] = torch.chunk(value, chunks=2, dim=1)
        #
        scale, shift = feature['term']
        value = feature['map'] * (1 + scale) + shift
        value = self.layer['(2) convolution'](value)
        state = (identity + value) / 2**0.5
        return(state)

    forward = getState
    pass

class Interpretation:

    def __init__(self, device: str) -> None:
        self.device = device
        return

    def getModule(
        self, 
        schema: dict
    ) -> Module:
        module = Module(device=self.device)
        module.activateLayer(schema)
        return(module)

    pass

mode = '!Debug'
if(mode=='Debug'):
    chaos = torch.randn(
        2,   # batch size
        512, # channel
        64,  # height
        64   # width
    )
    step = torch.randn(
        2,  # batch size
        512 # dimension of time embedding 
    )
    activation = tensordict.TensorDict(
        {"chaos": chaos, 'step': step}
    )
    schema = {
        'channel': (512, 128),
        'embedding': 512
    }
    interpretation = Interpretation(device='cuda')
    module = interpretation.getModule(
        schema
    )

    state = module(activation)
    print("No Error")
    pass


