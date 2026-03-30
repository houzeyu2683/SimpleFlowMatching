import torch
import tensordict

class Dilation(torch.nn.Module):

    def __init__(self, schema: dict, device: str) -> None:
        super().__init__()

        channel = schema['channel']
        embedding = schema['embedding']
        layer = torch.nn.ModuleDict(
            {
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
                    )
                ),
                "(1) norm": torch.nn.GroupNorm(
                    num_groups=32,
                    num_channels=channel[1],
                    eps=1e-05,
                    affine=True
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
        ).to(device)
        self.schema = schema
        self.device = device
        self.layer = layer
        return

    def getSession(self) -> dict:
        session = {}
        return(session)

    def getState(
        self,
        activation: tensordict.TensorDict,
    ) -> torch.Tensor:
        activation = activation.to(non_blocking=True, device=self.device)
        session = self.getSession()
        #
        session['image'] = activation['image']
        session['route'] = activation['route']
        #
        session['identity'] = self.layer['(0) skipped-connection'](
            session['image']
        )
        #
        session['map'] = self.layer['(1) convolution'](
            session['image']
        )
        session['map'] = self.layer['(1) norm'](
            session['map']
        )
        #
        value = self.layer['(1) full-connection'](
            session['route']
        )[..., None, None] # B, 2C
        scale, shift = torch.chunk(value, chunks=2, dim=1)
        session['map'] = self.layer['(2) convolution'](
            session['map'] * (1 + scale) + shift
        )
        state = (session['identity'] + session['map']) / (2**0.5)
        return(state)

    forward = getState
    pass

# class Dilation:

#     def __init__(self, device: str) -> None:
#         self.device = device
#         return

#     def getModule(
#         self,
#         schema: dict
#     ) -> Module:
#         module = Module(device=self.device)
#         module.activateLayer(schema)
#         return(module)

#     pass

# mode = '!Debug'
# if(mode=='Debug'):
#     chaos = torch.randn(
#         2,   # batch size
#         512, # channel
#         64,  # height
#         64   # width
#     )
#     step = torch.randn(
#         2,  # batch size
#         512 # dimension of time embedding
#     )
#     activation = tensordict.TensorDict(
#         {"chaos": chaos, 'step': step}
#     )
#     schema = {
#         'channel': (512, 128),
#         'embedding': 512
#     }
#     dilation = Dilation(device='cuda')
#     module = dilation.getModule(
#         schema
#     )

#     state = module(activation)
#     print("No Error")
#     pass


