import torch.distributed
import os
import torch

class Distribution:

    def __init__(self, model: torch.nn.Module, device: str) -> None:
        self.model = model
        self.device = device
        return
    
    def getDepartment(self) -> torch.nn.parallel.DistributedDataParallel:
        torch.distributed.init_process_group(backend='nccl')
        index = int(os.environ['LOCAL_RANK'])
        device = f'{self.device}:{index}'
        department = torch.nn.parallel.DistributedDataParallel(
            self.model, device_ids=[index]
        )
        return(department)

    pass





