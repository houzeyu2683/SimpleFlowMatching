import torch

class Module(torch.nn.Module):

    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device
        return

    def activateLayer(
        self,
        schema: dict,
    ) -> bool:
        channel = schema['channel']
        layer = {
            'upsampling': torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode='nearest'),
                torch.nn.Conv2d(
                    in_channels=channel,
                    out_channels=channel,
                    kernel_size=(1, 1)
                )
            )
        }
        self.layer = torch.nn.ModuleDict(layer).to(self.device)
        return(True)

    def getState(
        self,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        activation = activation.to(non_blocking=True, device=self.device)
        state = self.layer['upsampling'](activation)
        return(state)

    forward = getState
    pass


class Interpolation:

    def __init__(self, device: str) -> None:
        self.device = device
        return

    def getModule(
        self,
        schema: dict,
    ) -> Module:
        module = Module(device=self.device)
        module.activateLayer(schema)
        return(module)

    pass

mode = '!Debug'
if(mode=='Debug'):
    chaos = torch.randn(
        2,   # batch size
        128, # channel
        32,  # height
        32   # width
    )
    schema = {'channel': 128}
    interpolation = Interpolation(device='cuda')
    module = interpolation.getModule(schema)
    state = module(chaos)
    print(state.shape)  # (2, 25, 128, 64, 64)
    print("No Error")
    pass
