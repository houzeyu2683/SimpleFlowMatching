import torch

class Dilation(torch.nn.Module):

    def __init__(self, schema: dict, device: str) -> None:
        super().__init__()
        channel = schema['channel']
        layer = torch.nn.ModuleDict(
            {
                'upsampling': torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode='nearest'),
                    torch.nn.Conv2d(
                        in_channels=channel,
                        out_channels=channel,
                        kernel_size=(1, 1)
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
        state = self.layer['upsampling'](activation)
        return(state)

    forward = getState
    pass


# class Dilation:

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
#         32,  # height
#         32   # width
#     )
#     schema = {'channel': 128}
#     interpolation = Interpolation(device='cuda')
#     module = interpolation.getModule(schema)
#     state = module(chaos)
#     print(state.shape)  # (2, 25, 128, 64, 64)
#     print("No Error")
#     pass
