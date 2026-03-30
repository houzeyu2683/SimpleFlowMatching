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
        channel = schema['channel']
        layer = {
            'downsampling': torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=channel,
                    out_channels=channel,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1)
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
        state = self.layer['downsampling'](activation)
        return(state)

    forward = getState
    pass


class Decimation:

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
        64,  # height
        64   # width
    )
    schema = {'channel': 128}
    decimation = Decimation(device='cuda')
    module = decimation.getModule(schema)
    state = module(chaos)
    print("No Error")
    pass