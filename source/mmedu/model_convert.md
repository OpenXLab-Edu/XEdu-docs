# AI模型的转换

## 1. 常见的AI模型类型

### 1.1 ONNX

ONNX 的全称是“Open Neural Network Exchange”，即“开放的神经网络切换”，旨在实现不同神经网络开发框架之间的互通互用。ONNXRuntime是微软推出的一款推理框架，支持多种运行后端包括CPU，GPU，TensorRT，DML等，是对ONNX模型最原生的支持。

### 1.2 NCNN

腾讯公司开发的移动端平台部署工具，一个为手机端极致优化的高性能神经网络前向计算框架。NCNN仅用于推理，不支持学习。

## 2. MMEdu的模型转换

### 2.1 典型代码

```
from MMEdu import MMClassification as cls
model = cls(backbone='MobileNet')
model.num_classes = 2
checkpoint = 'checkpoints/cls_model/CatsDog/best_accuracy_top-1_epoch_2.pth'
out_file="out_file/catdog.onnx"
model.convert(checkpoint=checkpoint, backend="ONNX", out_file=out_file)
```

