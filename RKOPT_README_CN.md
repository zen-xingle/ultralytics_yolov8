## RKNN optimization for exporting model

## 1.Model structure optimize

目前的优化方式将模型尾部的 dfl 结构、以及将 0~1的坐标值还原回 0～640(图片输入尺寸) 的后处理结构移除。



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
# 调整 ./ultralytics/yolo/cfg/default.yaml 中 model 文件路径，默认为 yolov8n.pt，若自己训练模型，请调接至对应的路径。支持检测、分割模型。

export PYTHONPATH=./
python ./ultralytics/yolo/engine/exporter.py

# 执行完毕后，会生成 _rknnopt.torchscript 模型。假如原始模型为 yolov8n.pt，则生成 yolov8n_rknnopt.torchscript 模型。
```





### 3.转RKNN模型、Python demo、C demo

请参考 https://github.com/airockchip/rknn_model_zoo/tree/main/models/CV/object_detection/yolo 

