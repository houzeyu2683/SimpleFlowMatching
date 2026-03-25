import torch
import os
import onnxruntime
import logging

class Luggage:

    def __init__(self, folder: str) -> None:
        self.folder = folder
        return

    def exportModule(
        self,
        model: torch.nn.Module, 
        method: str,
        data: list,
        key: list,
        # elasticity: dict,
        archive: str
    ) -> bool:
        function = getattr(model, method)
        module = torch.nn.Module()
        setattr(module, 'forward', function)
        assert len(data)==len(key)
        data = tuple(data)
        os.makedirs(self.folder, exist_ok=True)
        path = os.path.join(self.folder, archive)
        # warnings.filterwarnings("ignore", module="torch.onnx")
        logging.getLogger("torch.onnx").setLevel(logging.ERROR)
        torch.onnx.export(
            module,
            data,
            path,
            input_names=key,
            external_data=False,
            # dynamic_axes=elasticity
        )
        return(True)

    pass