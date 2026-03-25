import torch
import diffusers
import tensordict
import tqdm
import perception

class Model(torch.nn.Module):

    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device
        return

    def loadWeight(self, path: str) -> bool:
        weight = torch.load(path)
        self.layer.load_state_dict(weight)
        return(True)

    def activateLayer(self) -> bool:
        conversion = perception.Conversion(device=self.device)
        decimation = perception.Decimation(device=self.device)
        interpretation = perception.Interpretation(device=self.device)
        interpolation = perception.Interpolation(device=self.device)
        bottleneck = perception.Bottleneck(device=self.device)
        space = perception.Space(device=self.device)
        tunnel = 128
        layer = {
            # Initial projection: 3 → C
            '(0) basic-convolution': torch.nn.Conv2d(3, tunnel, kernel_size=(1, 1)),  # chaos (B*L, C, 64, 64)
            # Encoder
            '(1) residual-encoding':  conversion.getModule(
                schema={"channel": (tunnel, tunnel), "embedding": 512}
            ),  # (B,tunnel,64,64)
            '(1) downsamping': decimation.getModule(
                schema={"channel": tunnel}
            ),  # (B,tunnel,32,32)
            '(2) residual-encoding': conversion.getModule(
                schema={'channel': (tunnel, 2*tunnel), "embedding": 512}
            ),  # (B,2*tunnel,32,32)
            '(2) downsamping': decimation.getModule(
                schema={"channel": 2*tunnel}
            ), # (B,2*tunnel,16,16)
            '(3) residual-encoding': conversion.getModule(
                schema={'channel': (2*tunnel, 3*tunnel), 'embedding': 512}
            ),  # (B,3*tunnel,16,16)
            '(3) attentive-encoding': space.getModule(
                schema={'embedding': (3*tunnel, tunnel), 'head': 2}
            ),  # (B,3*tunnel,16,16)     
            '(3) downsamping': decimation.getModule(
                schema={"channel": 3*tunnel}
            ), # (B,3*tunnel,8,8)
            '(4) bottleneck':  bottleneck.getModule(
                schema={'embedding': (3*tunnel, tunnel), 'head': 2}
            ), # (B,3*tunnel,8,8)
            '(5) upsamping': interpolation.getModule(
                schema={"channel": 3*tunnel}
            ), # (B,3*tunnel,16,16)
            '(5) attentive-encoding': space.getModule(
                schema={'embedding': (3*tunnel, tunnel), 'head': 2}
            ),  # (B,3*tunnel,16,16)
            "(5) residual-decoding": interpretation.getModule(
                schema={'channel': (3*tunnel*2, 2*tunnel), 'embedding': 512}
            ), # (B,2*tunnel,16,16)
            '(6) upsamping': interpolation.getModule(
                schema={"channel": 2*tunnel}
            ), # (B,2*tunnel, 32,32)
            "(6) residual-decoding": interpretation.getModule(
                schema={'channel': (2*tunnel*2, tunnel), 'embedding': 512}
            ),
            '(7) upsamping': interpolation.getModule(
                schema={"channel": tunnel}
            ), # (B,tunnel, 64,64)
            "(7) residual-decoding": interpretation.getModule(
                schema={'channel': (tunnel*2, tunnel), 'embedding': 512}
            ),
            '(8) basic-convolution': torch.nn.Conv2d(tunnel, 3, kernel_size=(1, 1)), 

        }
        self.layer = torch.nn.ModuleDict(layer).to(self.device)
        return(True)

    @torch.compile
    def getReception(
        self,
        chaos: torch.Tensor,
        step: torch.Tensor      # (B,) float in [0, 1]
    ) -> torch.Tensor:
        chaos = chaos.to(non_blocking=True, device=self.device)
        step = step.to(non_blocking=True, device=self.device)
        #
        degree = diffusers.models.embeddings.get_timestep_embedding(
            step, 512
        )
        #
        # Initial projection
        connection = []
        value = self.layer['(0) basic-convolution'](chaos)
        activation = tensordict.TensorDict({"chaos": value, 'step': degree})
        value = self.layer['(1) residual-encoding'](activation)
        connection += [value]
        activation = value
        value = self.layer['(1) downsamping'](activation)
        #
        activation = tensordict.TensorDict({"chaos": value, 'step': degree})
        value = self.layer['(2) residual-encoding'](activation)
        connection += [value]
        activation = value
        value = self.layer['(2) downsamping'](activation)
        #
        activation = tensordict.TensorDict({"chaos": value, 'step': degree})
        value = self.layer['(3) residual-encoding'](activation)
        value = self.layer['(3) attentive-encoding'](value)
        connection += [value]
        activation = value
        value = self.layer['(3) downsamping'](activation)
        #
        activation = value
        value = self.layer['(4) bottleneck'](activation)
        #
        activation = value
        value = self.layer['(5) upsamping'](activation)
        value = self.layer['(5) attentive-encoding'](value)
        value = torch.cat([value, connection.pop()], dim=1)
        activation = tensordict.TensorDict({"chaos": value, 'step': degree})
        value = self.layer['(5) residual-decoding'](activation)
        #
        activation = value
        value = self.layer['(6) upsamping'](activation)
        value = torch.cat([value, connection.pop()], dim=1)
        activation = tensordict.TensorDict({"chaos": value, 'step': degree})
        value = self.layer['(6) residual-decoding'](activation)
        #
        activation = value
        value = self.layer['(7) upsamping'](activation)
        value = torch.cat([value, connection.pop()], dim=1)
        activation = tensordict.TensorDict({"chaos": value, 'step': degree})
        value = self.layer['(7) residual-decoding'](activation)
        #
        activation = value
        reception = self.layer['(8) basic-convolution'](activation)
        return(reception)

    def getChaos(self, frame: torch.Tensor, noise: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        frame = frame.to(self.device, non_blocking=True)
        noise = noise.to(self.device, non_blocking=True)
        step = step.to(self.device, non_blocking=True)
        chaos = (1 - step) * frame + step * noise  # (B, 25, 3, 64, 64)
        return(chaos)

    def getCriteria(
        self,
        batch: tensordict.TensorDict
    ) -> tensordict.TensorDict:
        batch = batch.to(self.device, non_blocking=True)
        frame = batch['frame']   # (B, 3, 64, 64) clean image x0
        noise = batch['noise']   # (B, 3, 64, 64) Gaussian noise x1

        # Sample t ~ Logit-Normal (concentrated around 0.5)
        step = torch.sigmoid(
            torch.randn(len(batch))
        )
        # t_b = t

        # Linear interpolation: x_t = (1-t)*x0 + t*x1
        # x_t = (1 - step) * frame + step * noise  # (B, 25, 3, 64, 64)
        chaos = self.getChaos(frame, noise, step[:, None, None, None])

        # Velocity target: v* = x1 - x0
        target = noise - frame

        # Predict velocity
        velocity = self.getReception(chaos, step)  # (B, 25, 3, 64, 64)

        # MSE loss
        tolerance = torch.pow(target - velocity, 2).mean([1, 2, 3])
        total = tolerance.mean()

        criteria = tensordict.TensorDict(device=self.device)
        criteria.set('tolerance', tolerance.mean())
        criteria.set('total', total)
        return(criteria)

    @torch.no_grad()
    def getInvention(
        self,
        batch: int,  # number of videos to generate
        level: int   # number of Euler steps (20-50 is sufficient)
    ) -> torch.Tensor:
        x_t = torch.randn(batch, 3, 64, 64, device=self.device)

        dt = 1.0 / level
        timesteps = torch.linspace(1.0, dt, level)

        for t_val in tqdm.tqdm(timesteps):
            t = torch.full((batch,), t_val, device=self.device)
            velocity = self.getReception(x_t, t)
            x_t = x_t - dt * velocity

        invention = x_t  # (B, 3, 64, 64)
        return(invention)

    forward = getCriteria
    pass


class Discriminator(torch.nn.Module):
    """Window PatchGAN: 判斷 k 連續幀是真實還是生成的（unconditional）

    k=4: input = (B, 4*3, 64, 64) = (B, 12, 64, 64)
    """

    window = 4  # number of consecutive frames per window

    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device
        in_channels = self.window * 3   # k frames stacked
        layer = torch.nn.Sequential(
            # (B, 12, 64, 64)
            torch.nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # → (B, 64, 32, 32)
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # → (B, 128, 16, 16)
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.InstanceNorm2d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # → (B, 256, 8, 8)
            torch.nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
            # → (B, 1, 7, 7) patch scores
        )
        self.layer = layer.to(self.device)
        return

    def forward(
        self,
        window: torch.Tensor  # (B, k, 3, 64, 64) consecutive frames
    ) -> torch.Tensor:
        window = window.to(non_blocking=True, device=self.device)
        x = window.flatten(1, 2)  # (B, k*3, 64, 64)
        score = self.layer(x)     # (B, 1, 7, 7)
        return(score)

    pass
