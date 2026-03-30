import torch
import tensordict

class Contraction(torch.nn.Module):

    def __init__(self, schema: dict, device: str) -> None:
        super().__init__()
        channel = schema['channel']
        layer = torch.nn.ModuleDict(
            {
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
        ).to(device)
        self.schema = schema
        self.device = device
        self.layer = layer
        return

    def getState(
        self,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        activation = activation.to(non_blocking=True, device=self.device)
        state = self.layer['downsampling'](activation)
        return(state)

    forward = getState
    pass


# class Contraction:

#     def __init__(self, device: str) -> None:
#         self.device = device
#         return

#     def getModule(
#         self,
#         schema: dict,
#     ) -> Module:
#         module = Module(device=self.device)
#         module.activateLayer(schema)
#         return(module)

#     pass

# mode = '!Debug'
# if(mode=='Debug'):
#     chaos = torch.randn(
#         2,   # batch size
#         128, # channel
#         64,  # height
#         64   # width
#     )
#     schema = {'channel': 128}
#     decimation = Decimation(device='cuda')
#     module = decimation.getModule(schema)
#     state = module(chaos)
#     print("No Error")
#     pass