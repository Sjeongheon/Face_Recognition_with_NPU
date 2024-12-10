import os
import sys

import torch
import torch.nn as nn

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from models.experimental import attempt_load
from utils.general import check_img_size
import models
from utils.activations import Hardswish, SiLU
import onnx

class Yolov5Face(object):
    def __init__(self, model_file=None):
        """
        Initialize the Detector class.

        :param model_path: Path to the YOLOv5 model file (default is yolov5n-0.5.pt)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.model = attempt_load(model_file, map_location=device)

    def export_to_onnx(self, onnx_file_path, input_size=(128, 128)):
        """
        Export the PyTorch model to ONNX format.

        :param onnx_file_path: Path where the ONNX model will be saved.
        :param input_size: The input size for the model (e.g., 128x128).
        """
        # delete anchor grid
        delattr(self.model.model[-1], 'anchor_grid')
        self.model.model[-1].anchor_grid=[torch.zeros(1)] * 3 # nl=3 number of detection layers
        self.model.model[-1].export_cat = True
        self.model.eval()
        
        # Checks
        gs = int(max(self.model.stride))  # grid size (max stride)
        input_size = [check_img_size(x, gs) for x in input_size]  # verify img_size are gs-multiples
        
        # Create a dummy input tensor (batch size 1, 3 color channels, and input size)
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(self.device)

         # Update model
        for k, m in self.model.named_modules():
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
            if isinstance(m, models.common.Conv):  # assign export-friendly activations
                if isinstance(m.act, nn.Hardswish):
                    m.act = Hardswish()
                elif isinstance(m.act, nn.SiLU):
                    m.act = SiLU()
            # elif isinstance(m, models.yolo.Detect):
            #     m.forward = m.forward_export  # assign forward (optional)
            if isinstance(m, models.common.ShuffleV2Block):#shufflenet block nn.SiLU
                for i in range(len(m.branch1)):
                    if isinstance(m.branch1[i], nn.SiLU):
                        m.branch1[i] = SiLU()
                for i in range(len(m.branch2)):
                    if isinstance(m.branch2[i], nn.SiLU):
                        m.branch2[i] = SiLU()
        y = self.model(dummy_input)
        
        # ONNX export
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        self.model.fuse()  # only for ONNX
        input_names=['input']
        output_names=['output']
        torch.onnx.export(self.model, dummy_input, onnx_file_path, verbose=False, opset_version=12, 
            input_names=input_names,
            output_names=output_names,
            dynamic_axes = {'input': {0: 'batch'},
                            'output': {0: 'batch'}
                            })
        
        # Checks
        onnx_model = onnx.load(onnx_file_path)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print(f"Model has been successfully exported to {onnx_file_path}")

model = Yolov5Face(model_file="face_detection\yolov5_face\weights\yolov5m-face.pt")
model.export_to_onnx("yolov5m.onnx", input_size=(128, 128))