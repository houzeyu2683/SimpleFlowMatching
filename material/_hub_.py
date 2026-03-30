import torch
import os
import torchvision
import torchcodec
import functools
import PIL.Image
import tensordict
import safetensors.torch
import cv2
import application
import pathlib
import torchvision.transforms.functional
import random
import diffusers

def getCollation(queue: list) -> tensordict.TensorDict:
    bundle = {
        'target': [],
        'noise': []
    }
    iteration = enumerate(queue)
    for _, item in iteration:
        path = item
        image = PIL.Image.open(path).resize((64, 64))
        # assert torch.is_tensor(video)
        # video = video.float()
        # video = video / 255
        getDigit = torchvision.transforms.Normalize(
            [0.5, 0.5, 0.5], 
            [0.5, 0.5, 0.5]
        )
        target = getDigit(
            torchvision.transforms.ToTensor()(image)
        )
        # index = torch.randperm(len(video)-1-25)[0]
        # frame = video[index:index+1+8]   # (1, 3, 64, 64)
        noise = torch.randn_like(target)
        bundle['target'] += [target]
        bundle['noise'] += [noise]
        continue
    _ = iteration
    target = torch.stack(bundle['target'])
    noise = torch.stack(bundle['noise'])
    source = {
        'target': target,
        'noise': noise,
    }
    size = len(queue)
    collation = tensordict.TensorDict(
        source=source,
        batch_size=size
    ).detach()
    return(collation)

class Unit(torch.utils.data.Dataset):

    def __init__(self, queue: list) -> None:
        self.queue = queue
        return
    
    def getLength(self) -> int:
        length = len(self.queue)
        return(length)

    def getItem(self, index: int) -> tuple:
        item = self.queue[index]
        return(item)

    __len__ = getLength
    __getitem__ = getItem
    pass

class Document:

    def __init__(self, path: str) -> None:
        self.path = path
        return

    def getQueue(self) -> list:
        folder = os.path.dirname(self.path)
        paper = open(self.path, 'r')
        queue = []
        iteration = paper.readlines()
        for item in iteration:
            path = item.replace("\n", "")
            queue += [os.path.join(folder, path)]
            continue
        _ = iteration
        paper.close()
        return(queue)

    pass

class Hub:

    def __init__(self) -> None:
        return

    def getData(self, number: int) -> torch.utils.data.DataLoader:
        name = 'data.txt'
        path = os.path.join(self.bucket, self.folder, name)
        queue = Document(path).getQueue()
        unit = Unit(queue)
        sampler = torch.utils.data.distributed.DistributedSampler(unit)
        data = torch.utils.data.DataLoader(
            dataset=unit,
            batch_size=number,
            sampler=sampler,
            collate_fn=getCollation,
            drop_last=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )
        return(data)

    def getValidation(
        self, 
        number: int, 
        reproducibility: bool
    ) -> torch.utils.data.DataLoader:
        name = 'validation.txt'
        path = os.path.join(self.bucket, self.folder, name)
        queue = Document(path).getQueue()
        unit = Unit(queue)
        validation = torch.utils.data.DataLoader(
            dataset=unit,
            batch_size=number,
            shuffle=not reproducibility,
            collate_fn=getCollation, 
            drop_last=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        return(validation)
    
    def getTest(
        self, 
        number: int,
        reproducibility: bool
    ) -> torch.utils.data.DataLoader:
        name = 'test.txt'
        path = os.path.join(self.bucket, self.folder, name)
        queue = Document(path).getQueue()
        unit = Unit(queue)
        test = torch.utils.data.DataLoader(
            dataset=unit,
            batch_size=number,
            shuffle=not reproducibility,
            collate_fn=getCollation, 
            drop_last=False
        )
        return(test)
    
    def getBatch(self, number: int) -> tensordict.TensorDict:
        name = 'data.txt'
        path = os.path.join(self.bucket, self.folder, name)
        queue = Document(path).getQueue()
        unit = Unit(queue)
        data = torch.utils.data.DataLoader(
            dataset=unit,
            batch_size=number,
            shuffle=True,
            collate_fn=getCollation
        )
        batch = next(iter(data))
        return(batch)
    
    bucket = '/kaggle/input/datasets/houzeyu2683/ioaf-set'
    folder = 'IoAF/'
    # bucket = os.path.join(pathlib.Path.home(), 'Documents', 'Data Bucket')
    # folder = 'IoAF/'
    pass

