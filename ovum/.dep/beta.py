import torch
import diffusers
import tensordict
import safetensors.torch
import os
import requests

class Model(torch.nn.Module):

    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device
        return

    def loadWeight(self, path: str) -> bool:
        weight = safetensors.torch.load_file(path)
        self.load_state_dict(weight)
        return(True)

    def activateLayer(self) -> bool:
        layer = {
            '(0) stepwise projection': torch.nn.Sequential(
                torch.nn.Linear(128, 768),
                torch.nn.SiLU(),
                torch.nn.Linear(768, 768),
            ), # decoding
            '(1) convolution': torch.nn.Conv2d(
                in_channels=3, 
                out_channels=128, 
                kernel_size=(3, 3), 
                stride=(1, 1), 
                padding=(1, 1)
            ),
            '(2) residual encoding': torch.nn.ModuleDict(
                {
                    '0': torch.nn.Sequential(
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=128, 
                            eps=1e-05, 
                            affine=True
                        ),
                        torch.nn.SiLU(),
                        torch.nn.Conv2d(
                            in_channels=128, 
                            out_channels=128, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        ),
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=128, 
                            eps=1e-05, 
                            affine=True
                        )
                    ),
                    '1': torch.nn.Sequential(
                      torch.nn.SiLU(),
                      torch.nn.Linear(
                          in_features=768, 
                          out_features=128*2
                      )
                    ),
                    '2': torch.nn.Sequential(
                        torch.nn.SiLU(),
                        torch.nn.Dropout(p=0.1),
                        torch.nn.Conv2d(
                            in_channels=128, 
                            out_channels=128, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        )
                    )
                }
            ),
            '(3) compressive encoding': torch.nn.AvgPool2d(
                kernel_size=2, 
                stride=2, 
                padding=0
            ),
            '(4) convolution': torch.nn.Conv2d(
                in_channels=128, 
                out_channels=256, 
                kernel_size=(3, 3), 
                stride=(1, 1), 
                padding=(1, 1)
            ),
            '(5) residual encoding': torch.nn.ModuleDict(
                {
                    '0': torch.nn.Sequential(
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=256, 
                            eps=1e-05, 
                            affine=True
                        ),
                        torch.nn.SiLU(),
                        torch.nn.Conv2d(
                            in_channels=256, 
                            out_channels=256, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        ),
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=256, 
                            eps=1e-05, 
                            affine=True
                        )
                    ),
                    '1': torch.nn.Sequential(
                      torch.nn.SiLU(),
                      torch.nn.Linear(
                          in_features=768, 
                          out_features=256*2
                      )
                    ),
                    '2': torch.nn.Sequential(
                        torch.nn.SiLU(),
                        torch.nn.Dropout(p=0.1),
                        torch.nn.Conv2d(
                            in_channels=256, 
                            out_channels=256, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        )
                    )
                }
            ),
            '(6) compressive encoding': torch.nn.AvgPool2d(
                kernel_size=2, 
                stride=2, 
                padding=0
            ),
            '(7) convolution': torch.nn.Conv2d(
                in_channels=256, 
                out_channels=512, 
                kernel_size=(3, 3), 
                stride=(1, 1), 
                padding=(1, 1)
            ),
            '(8) residual encoding': torch.nn.ModuleDict(
                {
                    '0': torch.nn.Sequential(
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=512, 
                            eps=1e-05, 
                            affine=True
                        ),
                        torch.nn.SiLU(),
                        torch.nn.Conv2d(
                            in_channels=512, 
                            out_channels=512, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        ),
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=512, 
                            eps=1e-05, 
                            affine=True
                        )
                    ),
                    '1': torch.nn.Sequential(
                      torch.nn.SiLU(),
                      torch.nn.Linear(
                          in_features=768, 
                          out_features=512*2
                      )
                    ),
                    '2': torch.nn.Sequential(
                        torch.nn.SiLU(),
                        torch.nn.Dropout(p=0.1),
                        torch.nn.Conv2d(
                            in_channels=512, 
                            out_channels=512, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        )
                    )
                }
            ),
            '(9) compressive encoding': torch.nn.AvgPool2d(
                kernel_size=2, 
                stride=2, 
                padding=0
            ),
            '(10) residual bottleneck': torch.nn.ModuleDict(
                {
                    '0': torch.nn.Sequential(
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=512, 
                            eps=1e-05, 
                            affine=True
                        ),
                        torch.nn.SiLU(),
                        torch.nn.Conv2d(
                            in_channels=512, 
                            out_channels=512, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        ),
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=512, 
                            eps=1e-05, 
                            affine=True
                        )
                    ),
                    '1': torch.nn.Sequential(
                        torch.nn.SiLU(),
                        torch.nn.Linear(
                            in_features=768, 
                            out_features=512*2
                        )
                    ),
                    '2': torch.nn.Sequential(
                        torch.nn.SiLU(),
                        torch.nn.Dropout(p=0.1),
                        torch.nn.Conv2d(
                            in_channels=512, 
                            out_channels=512, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        )
                    )
                }
            ),
            '(11) residual attention': torch.nn.ModuleDict(
                {
                    '0': torch.nn.GroupNorm(
                        num_groups=32, 
                        num_channels=512, 
                        eps=1e-05, 
                        affine=True
                    ),
                    '1': torch.nn.MultiheadAttention(
                        embed_dim=512,
                        num_heads=8,
                        batch_first=True
                    ),
                    '2': torch.nn.Conv1d(
                        in_channels=512, 
                        out_channels=512, 
                        kernel_size=1, 
                        stride=1
                    )
                }
            ),
            '(12) residual bottleneck': torch.nn.ModuleDict(
                {
                    '0': torch.nn.Sequential(
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=512, 
                            eps=1e-05, 
                            affine=True
                        ),
                        torch.nn.SiLU(),
                        torch.nn.Conv2d(
                            in_channels=512, 
                            out_channels=512, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        ),
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=512, 
                            eps=1e-05, 
                            affine=True
                        )                        
                    ),
                    '1': torch.nn.Sequential(
                      torch.nn.SiLU(),
                      torch.nn.Linear(
                          in_features=768, 
                          out_features=512*2
                      )
                    ),
                    '2': torch.nn.Sequential(
                        torch.nn.SiLU(),
                        torch.nn.Dropout(p=0.1),
                        torch.nn.Conv2d(
                            in_channels=512, 
                            out_channels=512, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        )
                    )
                }
            ),
            '(13) residual decoding': torch.nn.ModuleDict(
                {
                    '0': torch.nn.Conv2d(
                        in_channels=1024, 
                        out_channels=512, 
                        kernel_size=(1, 1), 
                        stride=(1, 1)
                    ),
                    '1': torch.nn.Sequential(
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=1024, 
                            eps=1e-05, 
                            affine=True
                        ),
                        torch.nn.SiLU(),
                        torch.nn.Conv2d(
                            in_channels=1024, 
                            out_channels=512, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        )                     
                    ),
                    '2': torch.nn.Sequential(
                      torch.nn.SiLU(),
                      torch.nn.Linear(
                          in_features=768, 
                          out_features=512*2
                      )
                    ),
                    '3': torch.nn.Sequential(
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=512, 
                            eps=1e-05, 
                            affine=True
                        ),                        
                        torch.nn.SiLU(),
                        torch.nn.Dropout(p=0.1),
                        torch.nn.Conv2d(
                            in_channels=512, 
                            out_channels=512, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        )
                    )
                }
            ),
            "(14) residual decoding": torch.nn.ModuleDict(
                {
                    '0': torch.nn.Conv2d(
                        in_channels=1024, 
                        out_channels=512, 
                        kernel_size=(1, 1), 
                        stride=(1, 1)
                    ),
                    '1': torch.nn.Sequential(
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=1024, 
                            eps=1e-05, 
                            affine=True
                        ),
                        torch.nn.SiLU(),
                        torch.nn.Conv2d(
                            in_channels=1024, 
                            out_channels=512, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        )                     
                    ),
                    '2': torch.nn.Sequential(
                      torch.nn.SiLU(),
                      torch.nn.Linear(
                          in_features=768, 
                          out_features=512*2
                      )
                    ),
                    '3': torch.nn.Sequential(
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=512, 
                            eps=1e-05, 
                            affine=True
                        ),                        
                        torch.nn.SiLU(),
                        torch.nn.Dropout(p=0.1),
                        torch.nn.Conv2d(
                            in_channels=512, 
                            out_channels=512, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        )
                    )
                }
            ),
            "(15) convolution": torch.nn.Conv2d(
                in_channels=512, 
                out_channels=256, 
                kernel_size=(3, 3), 
                stride=(1, 1), 
                padding=(1, 1)
            ),
            "(16) residual decoding": torch.nn.ModuleDict(
                {
                    '0': torch.nn.Conv2d(
                        in_channels=512, 
                        out_channels=256, 
                        kernel_size=(1, 1), 
                        stride=(1, 1)
                    ),
                    '1': torch.nn.Sequential(
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=512, 
                            eps=1e-05, 
                            affine=True
                        ),
                        torch.nn.SiLU(),
                        torch.nn.Conv2d(
                            in_channels=512, 
                            out_channels=256, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        )                     
                    ),
                    '2': torch.nn.Sequential(
                      torch.nn.SiLU(),
                      torch.nn.Linear(
                          in_features=768, 
                          out_features=256*2
                      )
                    ),
                    '3': torch.nn.Sequential(
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=256, 
                            eps=1e-05, 
                            affine=True
                        ),                        
                        torch.nn.SiLU(),
                        torch.nn.Dropout(p=0.1),
                        torch.nn.Conv2d(
                            in_channels=256, 
                            out_channels=256, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        )
                    )
                }
            ),
            "(17) convolution": torch.nn.Conv2d(
                in_channels=256, 
                out_channels=128, 
                kernel_size=(3, 3), 
                stride=(1, 1), 
                padding=(1, 1)
            ),
            '(18) residual decoding': torch.nn.ModuleDict(
                {
                    '0': torch.nn.Conv2d(
                        in_channels=256, 
                        out_channels=128, 
                        kernel_size=(1, 1), 
                        stride=(1, 1)
                    ),
                    '1': torch.nn.Sequential(
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=256, 
                            eps=1e-05, 
                            affine=True
                        ),
                        torch.nn.SiLU(),
                        torch.nn.Conv2d(
                            in_channels=256, 
                            out_channels=128, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        )                     
                    ),
                    '2': torch.nn.Sequential(
                      torch.nn.SiLU(),
                      torch.nn.Linear(
                          in_features=768, 
                          out_features=128*2
                      )
                    ),
                    '3': torch.nn.Sequential(
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=128, 
                            eps=1e-05, 
                            affine=True
                        ),                        
                        torch.nn.SiLU(),
                        torch.nn.Dropout(p=0.1),
                        torch.nn.Conv2d(
                            in_channels=128, 
                            out_channels=128, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        )
                    )
                }
            ),
            "(19) residual decoding": torch.nn.ModuleDict(
                {
                    '0': torch.nn.Conv2d(
                        in_channels=256, 
                        out_channels=128, 
                        kernel_size=(1, 1), 
                        stride=(1, 1)
                    ),
                    '1': torch.nn.Sequential(
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=256, 
                            eps=1e-05, 
                            affine=True
                        ),
                        torch.nn.SiLU(),
                        torch.nn.Conv2d(
                            in_channels=256, 
                            out_channels=128, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        )                     
                    ),
                    '2': torch.nn.Sequential(
                      torch.nn.SiLU(),
                      torch.nn.Linear(
                          in_features=768, 
                          out_features=128*2
                      )
                    ),
                    '3': torch.nn.Sequential(
                        torch.nn.GroupNorm(
                            num_groups=32, 
                            num_channels=128, 
                            eps=1e-05, 
                            affine=True
                        ),                        
                        torch.nn.SiLU(),
                        torch.nn.Dropout(p=0.1),
                        torch.nn.Conv2d(
                            in_channels=128, 
                            out_channels=128, 
                            kernel_size=(3, 3), 
                            stride=(1, 1), 
                            padding=(1, 1)
                        )
                    )
                }
            ),
            "(20) reception": torch.nn.Sequential(
                torch.nn.GroupNorm(
                    num_groups=32, 
                    num_channels=128, 
                    eps=1e-05, 
                    affine=True
                ),
                torch.nn.SiLU(),
                torch.nn.Conv2d(
                    in_channels=128, 
                    out_channels=6, 
                    kernel_size=(3, 3), 
                    stride=(1, 1), 
                    padding=(1, 1)
                )
            )
        }
        self.layer = torch.nn.ModuleDict(layer).to(self.device)
        return(True)

    def getReception(
        self, 
        chaos: torch.Tensor,
        step: torch.Tensor
    ) -> torch.Tensor:
        chaos = chaos.to(self.device, non_blocking=True) # (B, 3, 64, 64)
        step = step.to(self.device, non_blocking=True) # (B, 3, 64, 64)
        #
        activation = []
        projection = self.layer['(0) stepwise projection'](
            diffusers.models.embeddings.get_timestep_embedding(step, 128)
        )
        state = self.layer['(1) convolution'](chaos)
        
        # ├─── main branch ──────────────────────────────┐
        # │    h = GroupNorm + SiLU(x)                   │
        # │    h = h_upd(h)  → (B, 192, 32, 32)          │
        # │    h = Conv2d(h) → (B, 192, 32, 32)          │
        # │    h = GroupNorm(h)                          │
        # |     h = h * (1+scale) + shift                │
        # │    h = SiLU + Dropout + Conv2d(h)            │
        # │                                              ↓
        # └─── residual branch ──────────────────────── h + x → output
        # ---------------------------------------------
        name = '(2) residual encoding'
        identity = state
        state = self.layer[name]['0'](state)
        term = torch.chunk(
            self.layer[name]['1'](projection)[..., None, None], 
            chunks=2, 
            dim=1
        )
        state = state * (1 + term[0]) + term[1]
        state = self.layer[name]['2'](state)
        state = state + identity 
        activation += [state]    # 128, 64, 64
        # ---------------------------------------------
        name = '(3) compressive encoding'
        state = self.layer[name](state) 
        activation += [state]    # 128, 32, 32
        # ---------------------------------------------
        state = self.layer['(4) convolution'](state) 
        # ---------------------------------------------
        name = '(5) residual encoding'
        identity = state
        state = self.layer[name]['0'](state)
        term = torch.chunk(
            self.layer[name]['1'](projection)[..., None, None], 
            chunks=2, 
            dim=1
        )
        state = state * (1 + term[0]) + term[1]
        state = self.layer[name]['2'](state)
        state = state + identity 
        activation += [state]    # 256, 32, 32
        # ---------------------------------------------
        name = '(6) compressive encoding'
        state = self.layer[name](state) # 256, 16, 16
        # ---------------------------------------------
        state = self.layer['(7) convolution'](state) # 512, 16, 16
        # ---------------------------------------------
        identity = state
        name = '(8) residual encoding'
        state = self.layer[name]['0'](state)
        term = torch.chunk(
            self.layer[name]['1'](projection)[..., None, None], 
            chunks=2, 
            dim=1
        )
        state = state * (1 + term[0]) + term[1]
        state = self.layer[name]['2'](state)
        state = state + identity # 512, 16, 16
        activation += [state]    # 512, 16, 16
        # ---------------------------------------------
        name = '(9) compressive encoding'
        state = self.layer[name](state) # 512, 8, 8
        # ---------------------------------------------
        name = '(10) residual bottleneck'
        identity = state
        state = self.layer[name]['0'](state)
        term = torch.chunk(
            self.layer[name]['1'](projection)[..., None, None], 
            chunks=2, 
            dim=1
        )
        state = state * (1 + term[0]) + term[1]
        state = self.layer[name]['2'](state)
        state = state + identity
        activation += [state]    # 512, 8, 8
        # ---------------------------------------------
        name = '(11) residual attention'
        state = self.layer[name]['0'](state) # group norm
        shape = torch.cat([state]).shape
        patch = torch.flatten(state, 2, -1).transpose(1, 2)
        patch , _ = self.layer[name]['1'](patch, patch, patch)
        patch = self.layer[name]['2'](torch.transpose(patch, 1, 2))
        state = torch.reshape(patch, shape)
        # ---------------------------------------------
        name = '(12) residual bottleneck'
        identity = state
        state = self.layer[name]['0'](state)
        term = torch.chunk(
            self.layer[name]['1'](projection)[..., None, None], 
            chunks=2, 
            dim=1
        )
        state = state * (1 + term[0]) + term[1]
        state = self.layer[name]['2'](state)
        state = state + identity # 512, 8, 8
        # ---------------------------------------------
        name = '(13) residual decoding'
        state = torch.cat([state, activation[-1]], dim=1)
        _ = activation.pop()
        identity = self.layer[name]['0'](state)
        state = self.layer[name]['1'](state)
        term = torch.chunk(
            self.layer[name]['2'](projection)[..., None, None], 
            chunks=2, 
            dim=1
        )
        state = state * (1 + term[0]) + term[1]
        state = self.layer[name]['3'](state)
        state = state + identity # 512, 8, 8
        # ---------------------------------------------
        name = '(14) residual decoding'
        state = torch.nn.functional.interpolate(
            state, scale_factor=2, mode="nearest"
        )
        state = torch.cat([state, activation[-1]], dim=1) # 1024, 16, 16
        _ = activation.pop()
        identity = self.layer[name]['0'](state) # 512, 16, 16
        state = self.layer[name]['1'](state)
        term = torch.chunk(
            self.layer[name]['2'](projection)[..., None, None], 
            chunks=2, 
            dim=1
        )
        state = state * (1 + term[0]) + term[1]
        state = self.layer[name]['3'](state)
        state = state + identity # 512, 16, 16
        # ---------------------------------------------
        name = '(15) convolution'
        state = self.layer[name](state) # 256, 16, 16
        # ---------------------------------------------
        name = '(16) residual decoding'
        state = torch.nn.functional.interpolate(
            state, scale_factor=2, mode="nearest"
        )        
        state = torch.cat([state, activation[-1]], dim=1)
        _ = activation.pop()
        identity = self.layer[name]['0'](state)
        state = self.layer[name]['1'](state)
        term = torch.chunk(
            self.layer[name]['2'](projection)[..., None, None], 
            chunks=2, 
            dim=1
        )
        state = state * (1 + term[0]) + term[1]
        state = self.layer[name]['3'](state)
        state = state + identity # 512, 8, 8
        # ---------------------------------------------
        name = '(17) convolution'
        state = self.layer[name](state)
        # ---------------------------------------------
        name = '(18) residual decoding'
        state = torch.cat([state, activation[-1]], dim=1)
        _ = activation.pop()
        identity = self.layer[name]['0'](state)
        state = self.layer[name]['1'](state)
        term = torch.chunk(
            self.layer[name]['2'](projection)[..., None, None], 
            chunks=2, 
            dim=1
        )
        state = state * (1 + term[0]) + term[1]
        state = self.layer[name]['3'](state)
        state = state + identity # 128, 32, 32
        # ---------------------------------------------
        name = '(19) residual decoding'
        state = torch.nn.functional.interpolate(
            state, scale_factor=2, mode="nearest"
        )      
        state = torch.cat([state, activation[-1]], dim=1)
        _ = activation.pop()
        identity = self.layer[name]['0'](state)
        state = self.layer[name]['1'](state)
        term = torch.chunk(
            self.layer[name]['2'](projection)[..., None, None], 
            chunks=2, 
            dim=1
        )
        state = state * (1 + term[0]) + term[1]
        state = self.layer[name]['3'](state)
        state = state + identity # 128, 64, 64
        # ---------------------------------------------
        name = '(20) reception'
        reception = self.layer[name](state)
        return(reception)
    
    pass



# # main branch
# h = GroupNorm(1536) + SiLU(x)          # (B, 1536, 8×8)
# h = Identity(h)                         # 不變
# h = Conv2d(1536→768)(h)                 # (B, 768, 8×8)  ← channel 壓縮

# scale, shift = Linear(768→1536)(emb)    # 各 768-dim
# h = GroupNorm(768) * (1+scale) + shift  # scale-shift
# h = SiLU + Dropout + Conv2d(768→768)(h) # (B, 768, 8×8)

# # residual branch
# x = Conv2d(1536→768, 1×1)(x)           # (B, 768, 8×8)  ← 1×1 conv 對齊 channel

# return h + x                            # (B, 768, 8×8)


        # ---------------------------------------------

        # value = compression
        # name = '(8) residual bottleneck'
        # term = torch.chunk(
        #     self.layer[name]['stepwise embedding'](step)[..., None, None], 
        #     chunks=2, 
        #     dim=1
        # )
        # identity = self.layer[name]['convolution'](value)
        # digit = self.layer[name]['0'](identity)
        # digit = self.layer[name]['1'](digit)
        # digit = digit * (1 + term[0]) + term[1]
        # digit = digit + identity # 512, 16, 16
        # activation += [digit]

        # criteria = tensordict.TensorDict(device=self.device)
        # # criteria.set("divergence", divergence)
        # criteria.set("pixel", pixel)
        # criteria.set("total", total)
    #     return

    # # forward = getCriteria
    # pass

device = 'cuda'
chaos = torch.randn((1, 3, 64, 64))
step = torch.randint(0, 1000, [1])

model = Model(device)
model.activateLayer()
model.getReception(chaos, step)