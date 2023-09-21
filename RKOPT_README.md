## RKNN optimization for exporting model

## 1. Model structure optimization

The current optimization removes the dfl structure at the end of the model and the post-processing structure.



## 2. Adjustments

- The dfl structure performs poorly in NPU processing performance. Assuming there are 6000 candidate boxes, if the original model places the dfl structure before the "box confidence filtering," then all 6000 candidate boxes need to undergo dfl calculations. However, if the dfl structure is placed after the "box confidence filtering" and there are, for example, 100 candidate boxes, the computational load of the dfl part reduces to 100.

  Therefore, even though the dfl structure is processed by the CPU and does not benefit from NPU acceleration, the reduced computational load is still significant.

- Assuming there are 6000 candidate boxes and 80 classes of detection targets, there are approximately 4.8*10^5 confidence values that need to be checked. This occupies a significant amount of time. Therefore, when exporting the model, an additional summation operation for the 80 classes of detection targets was introduced to quickly filter the confidences. This structure is effective for certain scenarios.

  To disable this optimization, you can comment out the following code located at lines 470-478 in "./ultralytics/nn/modules.py":

```
cls_sum = torch.clamp(y[-1].sum(1, keepdim=True), 0, 1)
y.append(cls_sum)
```




## 3. Model Export Operation

After meeting the environment requirements specified in "./requirements.txt," execute the following command to export the model( support detect/ segment model):

```
# Adjust the model file path in "./ultralytics/yolo/cfg/default.yaml" (default is yolov8n.pt). If you trained your own model, please provide the corresponding path.

export PYTHONPATH=./
python ./ultralytics/engine/exporter.py

# Upon completion, the "_rknnopt.torchscript" model will be generated. If the original model is "yolov8n.pt," the generated model will be "yolov8n_rknnopt.torchscript."
```



## 4. Convert to RKNN model, Python demo, C demo

Please refer to https://github.com/airockchip/rknn_model_zoo/tree/main/models/CV/object_detection/yolo for converting to an RKNN model, Python demo, and C demo.