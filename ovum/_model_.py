import torch
import diffusers
import tensordict
import tqdm
import torchvision
import perception

class Model(torch.nn.Module):

    def __init__(self, device: str) -> None:
        super().__init__()
        self.device = device
        return

    def loadWeight(self, path: str) -> bool:
        weight = torch.load(path)
        self.load_state_dict(weight)
        return(True)

    def activateLayer(self) -> bool:
        convolution = perception.convolution
        sampling = perception.sampling
        attention = perception.attention
        tunnel = 128
        layer = {
            '(0) basic-convolution': torch.nn.Conv2d(
                3, tunnel, kernel_size=(1, 1)
            ),  
            '(1) residual-encoding': convolution.Contraction(
                schema={"channel": (tunnel, 2*tunnel), "embedding": 512},
                device=self.device
            ),
            # (B, 2*tunnel, 64, 64)
            "(1) attentive-patch": attention.Patch(
                schema={'embedding': (2*tunnel, tunnel), 'head': 4},
                device=self.device
            ),
            # (B, 2*tunnel, 64, 64)
            '(1) downsamping': sampling.Contraction(
                schema={"channel": 2*tunnel},
                device=self.device
            ),
            # (B, 2*tunnel, 32, 32)
            '(2) residual-encoding': convolution.Contraction(
                schema={'channel': (2*tunnel, 3*tunnel), "embedding": 512},
                device=self.device
            ),
            # (B, 3*tunnel, 32, 32)
            "(2) attentive-patch": attention.Patch(
                schema={'embedding': (3*tunnel, tunnel), 'head': 4},
                device=self.device
            ),
            # (B, 3*tunnel, 32, 32)
            '(2) downsamping': sampling.Contraction(
                schema={"channel": 3*tunnel},
                device=self.device
            ),
            # (B, 3*tunnel, 16, 16)
            '(3) residual-encoding': convolution.Contraction(
                schema={'channel': (3*tunnel, 4*tunnel), "embedding": 512},
                device=self.device
            ),
            # (B, 4*tunnel, 16, 16)
            "(3) attentive-patch": attention.Patch(
                schema={'embedding': (4*tunnel, tunnel), 'head': 4},
                device=self.device
            ),
            # (B, 4*tunnel, 16, 16)
            '(3) downsamping': sampling.Contraction(
                schema={"channel": 4*tunnel},
                device=self.device
            ),
            # (B, 4*tunnel, 8, 8)
            "(4) bottleneck": attention.Patch(
                schema={'embedding': (4*tunnel, tunnel), 'head': 4},
                device=self.device
            ),
            # (B, 4*tunnel, 8, 8)
            '(5) upsamping': sampling.Dilation(
                schema={"channel": 4*tunnel},
                device=self.device
            ),
            # (B, 4*tunnel, 16, 16)
            "(5) attentive-patch": attention.Patch(
                schema={'embedding': (4*tunnel, tunnel), 'head': 4},
                device=self.device
            ),
            # (B, 4*tunnel, 16, 16)
            '(5) residual-decoding': convolution.Dilation(
                schema={'channel': (8*tunnel, 3*tunnel), "embedding": 512},
                device=self.device
            ),
            # (B, 3*tunnel, 16, 16)
            '(6) upsamping': sampling.Dilation(
                schema={"channel": 3*tunnel},
                device=self.device
            ),
            # (B, 3*tunnel, 32, 32)
            "(6) attentive-patch": attention.Patch(
                schema={'embedding': (3*tunnel, tunnel), 'head': 4},
                device=self.device
            ),
            # (B, 3*tunnel, 32, 32)
            '(6) residual-decoding': convolution.Dilation(
                schema={'channel': (6*tunnel, 2*tunnel), "embedding": 512},
                device=self.device
            ),
            # (B, 2*tunnel, 32, 32)
            '(7) upsamping': sampling.Dilation(
                schema={"channel": 2*tunnel},
                device=self.device
            ),
            # (B, 2*tunnel, 64, 64)
            "(7) attentive-patch": attention.Patch(
                schema={'embedding': (2*tunnel, tunnel), 'head': 4},
                device=self.device
            ),
            # (B, 2*tunnel, 64, 64)
            '(7) residual-decoding': convolution.Dilation(
                schema={'channel': (4*tunnel, tunnel), "embedding": 512},
                device=self.device
            ),
            # (B, tunnel, 64, 64)            
            '(8) basic-convolution': torch.nn.Conv2d(
                tunnel, 3, kernel_size=(1, 1)
            )
        }
        self.layer = torch.nn.ModuleDict(layer).to(self.device)
        return(True)

    def getSession(self) -> dict:
        session = {}
        return(session)

    # @torch.compile
    def getVelocity(
        self,
        image: torch.Tensor,
        step: torch.Tensor      # (B,) float in [0, 1]
    ) -> torch.Tensor:
        image = image.to(non_blocking=True, device=self.device)
        step = step.to(non_blocking=True, device=self.device)
        session = self.getSession()
        #
        power = 1000
        route = diffusers.models.embeddings.get_timestep_embedding(
            step*power, 512
        )
        activation = tensordict.TensorDict(
            {
                # "image": self.layer['(0) basic-convolution*'](image), 
                "image": self.layer['(0) basic-convolution'](image), 
                'route': route
            }
        )
        value = self.layer['(1) attentive-patch'](
            self.layer['(1) residual-encoding'](activation)
        )
        session['1'] = value
        #
        activation = tensordict.TensorDict(
            {
                "image": self.layer['(1) downsamping'](value), 
                'route': route
            }
        )
        value = self.layer['(2) attentive-patch'](
            self.layer['(2) residual-encoding'](activation)
        )
        session['2'] = value
        #
        activation = tensordict.TensorDict(
            {
                "image": self.layer['(2) downsamping'](value), 
                'route': route
            }
        )
        value = self.layer['(3) attentive-patch'](
            self.layer['(3) residual-encoding'](activation)
        )
        session['3'] = value
        #
        value = self.layer['(3) downsamping'](value)
        #
        value = self.layer['(4) bottleneck'](value)
        #
        value = self.layer['(5) attentive-patch'](
            self.layer['(5) upsamping'](value)
        )
        activation = tensordict.TensorDict(
            {
                "image": torch.cat([value, session['3']], dim=1), 
                'route': route
            }
        )
        value = self.layer['(5) residual-decoding'](activation)
        #
        value = self.layer['(6) attentive-patch'](
            self.layer['(6) upsamping'](value)
        )
        activation = tensordict.TensorDict(
            {
                "image": torch.cat([value, session['2']], dim=1), 
                'route': route
            }
        )
        value = self.layer['(6) residual-decoding'](activation)
        
        value = self.layer['(7) attentive-patch'](
            self.layer['(7) upsamping'](value)
        )
        activation = tensordict.TensorDict(
            {
                "image": torch.cat([value, session['1']], dim=1), 
                'route': route
            }
        )
        value = self.layer['(7) residual-decoding'](activation)
        
        velocity = self.layer['(8) basic-convolution'](value)
        # imagination = self.layer['(8) basic-convolution*'](value)
        return(velocity)

    def getImage(
        self, target: torch.Tensor, noise: torch.Tensor, step: torch.Tensor
    ) -> torch.Tensor:
        target = target.to(self.device, non_blocking=True)
        noise = noise.to(self.device, non_blocking=True)
        step = step.to(self.device, non_blocking=True)
        image = (1 - step) * target + step * noise  # (B, 25, 3, 64, 64)
        return(image)

    def getCriteria(
        self,
        batch: tensordict.TensorDict
    ) -> tensordict.TensorDict:
        batch = batch.to(self.device, non_blocking=True)
        target = batch['target']   # (B, 3, 64, 64) clean image x0
        noise = batch['noise']   # (B, 3, 64, 64) Gaussian noise x1

        # Sample t ~ Logit-Normal (concentrated around 0.5)
        step = torch.sigmoid(
            torch.randn(len(batch))
        )
        # t_b = t
        # step = torch.rand(len(batch))

        # Linear interpolation: x_t = (1-t)*x0 + t*x1
        # x_t = (1 - step) * frame + step * noise  # (B, 25, 3, 64, 64)
        image = self.getImage(target, noise, step[:, None, None, None])

        # Velocity target: v* = x1 - x0
        truth = noise - target

        # Predict velocity
        velocity = self.getVelocity(image, step)  # (B, 25, 3, 64, 64)

        # MSE loss
        tolerance = torch.pow(truth - velocity, 2).mean([1, 2, 3])
        total = tolerance.mean()

        criteria = tensordict.TensorDict(device=self.device)
        criteria.set('tolerance', tolerance.mean())
        criteria.set('total', total)
        return(criteria)

    @torch.no_grad()
    def getInvention(
        self,
        number: int, # number of videos to generate
        level: int   # number of Euler steps (20-50 is sufficient)
    ) -> torch.Tensor:
        noise = torch.randn(number, 3, 64, 64, device=self.device)

        delta = 1.0 / level
        iteration = tqdm.tqdm(torch.linspace(1.0, delta, level))

        for item in iteration:
            shape = torch.Size([number])
            step = torch.full(shape, item, device=self.device)
            velocity = self.getVelocity(noise, step)
            noise = noise - (delta * velocity)
            continue
        _ = iteration
        invention = (noise.clamp(-1, 1) + 1.0) / 2.0  # (B, 3, 64, 64)
        invention = torchvision.utils.make_grid(
            invention, normalize=True, value_range=(0, 1)
        )
        return(invention)

    forward = getCriteria
    pass


