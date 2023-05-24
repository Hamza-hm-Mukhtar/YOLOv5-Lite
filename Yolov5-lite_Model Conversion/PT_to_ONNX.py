import torch
from models.experimental import attempt_load


weights_path = 'racket_best_yslite.pt'
model = attempt_load(weights_path, map_location=torch.device('cpu'))


onnx_path = 'yolov5s_lite.onnx'
model.model[-1].export = torch.onnx.export(model.model[-1], torch.randn(1, 3, 640, 640), onnx_path, verbose=False, opset_version=12, input_names=['input'], output_names=['output'])

