# RKNN optimization for exporting model

## Source
Base on https://github.com/ultralytics/ultralytics with commit id as c9be1f3cce89778f79fb462797b8ca0300e3813d




## What different
With inference result values unchanged, the following optimizations were applied:
- Change output node, remove post-process from the model. (post-process block in model is unfriendly for quantization)
- Remove dfl structure at the end of the model. (which slowdown the inference speed on NPU device)
- Add a score-sum output branch to speedup post-process.

All the removed operation will be done on CPU. (the CPU post-process could be found in **RKNN_Model_Zoo**)




## Export ONNX model

After meeting the environment requirements specified in "./requirements.txt," execute the following command to export the model (support detect/segment model):

```
# Adjust the model file path in "./ultralytics/yolo/cfg/default.yaml" (default is yolov8n.pt). If you trained your own model, please provide the corresponding path. 
# For example, filled with yolov8n.pt for detection model.
# Filling with yolov8n-seg.pt for segmentation model.

export PYTHONPATH=./
python ./ultralytics/engine/exporter.py

# Upon completion, the ".onnx" model will be generated. If the original model is "yolov8n.pt," the generated model will be "yolov8n.onnx"
```



## Convert to RKNN model, Python demo, C demo

Please refer to https://github.com/airockchip/rknn_model_zoo.