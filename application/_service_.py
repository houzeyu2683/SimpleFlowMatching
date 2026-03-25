import torch
import os
import onnxruntime
import logging
import requests

class Service:

    def __init__(
        self, 
        version: str, 
        archive: str
    ) -> None:
        self.version = version
        self.archive = archive
        return
    
    # https://github.com/houzeyu2683/Ae/releases/download/sparrow-v1.0.0/getCompression.onnx
    def loadSession(self) -> bool:
        # link = 'https://github.com/houzeyu2683/VAe/releases/download/'
        link = os.path.join(self.head, self.version, self.archive)
        path = os.path.join(self.root, self.version, self.archive)
        # root = '.hub/model/'
        # archive = 'weight.pt'
        # access = os.path.join(link, tag, archive)
        # folder = os.path.join(root, tag)
        # archive = os.path.basename(access)
        # path = os.path.join(folder, archive)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        here = os.path.isfile(path)
        if(not here):
            response = requests.get(link, stream=True)
            paper = open(path, 'wb')
            # with open(path, 'wb') as paper:
            for chunk in response.iter_content(chunk_size=8192):
                paper.write(chunk)
                continue
            paper.close()
            pass
        # weight = safetensors.torch.load_file(path)
        # self.load_state_dict(weight)
        session = onnxruntime.InferenceSession(
            path, 
            providers=[
                'CUDAExecutionProvider', 
                'CPUExecutionProvider'
            ]
        )
        self.session = session
        return(True)

    # def makeSession(self, archive: str) -> bool:
    #     path = os.path.join(self.folder, archive)
    #     session = onnxruntime.InferenceSession(
    #         path, 
    #         providers=[
    #             'CUDAExecutionProvider', 
    #             'CPUExecutionProvider'
    #         ]
    #     )
    #     self.session = session
    #     return(True)

    def getResponse(self, request: dict) -> list:
        response = self.session.run(None, request)
        # self.response = response
        return(response)
    
    __call__ = getResponse
    head = 'https://github.com/houzeyu2683/Ae/releases/download'
    root = '.hub/model/'
    pass