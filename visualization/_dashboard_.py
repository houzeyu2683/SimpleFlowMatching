import os
import torch
import typing
import torchvision
import torch.utils.tensorboard

class Dashboard:

    def __init__(self, directory: str) -> None:
        self.directory = directory
        return

    def openSession(self) -> bool:
        os.makedirs(self.directory, exist_ok=True)
        self.session = torch.utils.tensorboard.SummaryWriter(self.directory)
        return(True)

    def closeSession(self) -> bool:
        self.session.close()
        return(True)

    def insertStatistic(
        self, 
        tag: str, 
        element: dict, 
        number: int
    ) -> bool:
        self.session.add_scalars(tag, element, number)
        self.session.flush()
        return(True)
    
    def insertPicture(
        self, tag: str, image: torch.Tensor, number: int
    ) -> bool:
        grid = torchvision.utils.make_grid(
            image, 
            normalize=True,
            value_range=(-1, 1)
        )
        self.session.add_image(tag, grid, number)
        return(True)

    pass
