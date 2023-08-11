<<<<<<< HEAD
## RKNN 导出模型说明

### 1.调整部分

- 由于 dfl 结构在 npu 处理性能不佳。假设有6000个候选框，原模型将 dfl 结构放置于 ''框置信度过滤" 前，则 6000 个候选框都需要计算经过 dfl 计算；而将 dfl 结构放置于 ''框置信度过滤" 后，假设过程成 100 个候选框，则dfl部分计算量减少至 100 个。

  故将 dfl 结构使用 cpu 处理的耗时，虽然享受不到 npu 加速，但是本来带来的计算量较少也是很可观的。



- 假设存在 6000 个候选框，存在 80 类检测目标，则阈值需要检索的置信度有 6000* 80 ～= 4.8*10^5 个，占据了较多耗时，故导出模型时，在模型中额外新增了对 80 类检测目标进行求和操作，用于快速过滤置信度，该结构在部分情况下对模型有效。

  可以在 ./ultralytics/nn/modules.py  470行～478行的位置，注释掉这部分优化，对应的代码是:

  ```
  cls_sum = torch.clamp(y[-1].sum(1, keepdim=True), 0, 1)
  y.append(cls_sum)
  ```

  



### 2.导出模型操作

在满足 ./requirements.txt 的环境要求后，执行以下语句导出模型

```
# 调整 ./ultralytics/yolo/cfg/default.yaml 中 model 文件路径，默认为 yolov8n.pt，若自己训练模型，请调接至对应的路径
=======
## Description - export optimized model for RKNPU

### 1. Model structure Adjustment

- The dfl structure has poor performance on NPU processing, moved outside the model.

   Assuming that there are 6000 candidate frames, the original model places the dfl structure before the "box confidence filter", then the 6000 candidate frames need to be calculated through dfl calculation. If the dfl structure is placed after the "box confidence filter", Assuming that there are 100 candidate boxes left after filtering, the calculation amount of the dfl part is reduced to 100, which greatly reduces the occupancy of computing resources and bandwidth resources.



- Assuming that there are 6000 candidate boxes and the detection category is 80, the threshold retrieval operation needs to be repeated 6000* 80 ~= 4.8*10^5 times, which takes a lot of time. Therefore, when exporting the model, an additional summation operation for 80 types of detection targets is added to the model to quickly filter the confidence. (This structure is effective in some cases, related to the training results of the model)

  You can comment out this part of the optimization at line 470 to line 478 of ./ultralytics/nn/modules.py, and the corresponding code is:

```
cls_sum = torch.clamp(y[-1].sum(1, keepdim=True), 0, 1)
y.append(cls_sum)
```




- (optional) In fact, if the user refers to the structure of yolov5, the output of 80 categories is adjusted to 80+1 category, and the newly added category 1 is used as the confidence level of the control box, which acts as a filter. In this way, the post-processing can reduce the number of logical judgments by 10 to 40 times when the CPU executes the threshold judgment.



### 2. Export model operation

After meeting the environmental requirements of ./requirements.txt, execute the following statement to export the model

```
# Adjust the model file path in ./ultralytics/yolo/cfg/default.yaml, the default is yolov8n.pt, if you train the model yourself, please transfer to the corresponding path
>>>>>>> 73dee0dffa77fe0e1d63c4a8e369ebaf28ff906d

export PYTHONPATH=./
python ./ultralytics/yolo/engine/exporter.py

<<<<<<< HEAD
执行完毕后，会生成 _rknnopt.torchscript 模型。假如原始模型为 yolov8n.pt，则生成 yolov8n_rknnopt.torchscript 模型。
=======
After execution, the _rknnopt.torchscript model will be generated. If the original model is yolov8n.pt, generate the yolov8n_rknnopt.torchscript model.
>>>>>>> 73dee0dffa77fe0e1d63c4a8e369ebaf28ff906d
```



<<<<<<< HEAD


### 3.转RKNN模型、Python demo、C demo

请参考 https://github.com/airockchip/rknn_model_zoo/tree/main/models/CV/object_detection/yolo 
=======
Export Code Changes Explained

- In ultralytics/yolo/cfg/default.yaml, there is a parameter **format** for exporting the model format, and the support for 'rknn' has been added
- When the model is inferred to Detect Head, format=='rknn' takes effect, dfl and post-processing are skipped,
- It should be noted that this repository has not tested the optimization method of pose head and segment head, which is currently not supported. You can try to change it yourself if needed.



### 3. Transfer to RKNN model, Python demo, C demo

Please refer to https://github.com/airockchip/rknn_model_zoo/tree/main/models/CV/object_detection/yolo
>>>>>>> 73dee0dffa77fe0e1d63c4a8e369ebaf28ff906d

