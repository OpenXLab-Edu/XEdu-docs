# AI模型的转换

## 1. 常见的AI模型类型

### 1.1 ONNX

ONNX 的全称是“Open Neural Network Exchange”，即“开放的神经网络切换”，旨在实现不同神经网络开发框架之间的互通互用。ONNXRuntime是微软推出的一款推理框架，支持多种运行后端包括CPU，GPU，TensorRT，DML等，是对ONNX模型最原生的支持。

### 1.2 NCNN

腾讯公司开发的移动端平台部署工具，一个为手机端极致优化的高性能神经网络前向计算框架。NCNN仅用于推理，不支持学习。

## 2. MMEdu的模型转换

- 转换准备：

  分类的标签文件、待转换的模型权重文件。

- 需要配置四个信息：

  待转换的模型权重文件（`checkpoint`），图片分类的类别数量（model.num_classes），分类标签信息文件（`class_path`）和输出的文件（`out_file`）。

### 2.1 典型代码

```
from MMEdu import MMClassification as cls
model = cls(backbone='MobileNet')
model.num_classes = 2
checkpoint = 'checkpoints/cls_model/CatsDog/best_accuracy_top-1_epoch_2.pth'
out_file="out_file/catdog.onnx"
model.convert(checkpoint=checkpoint, backend="ONNX", out_file=out_file, class_path=class_path)
```

这段代码是完成分类模型的转换，接下来对为您`model.convert`函数的各个参数：

`checkpoint`：选择想要进行模型转换的权重文件，以.pth为后缀。

`backend`：模型转换的后端推理框架，目前支持ONNX，后续将陆续支持NCNN、TensorRT、OpenVINO等。

`out_file`：模型转换后的输出文件路径。

`class_path`：模型输入的类别文件路径。

目标检测模型转换的示例代码如下：

```
from MMEdu import MMDetection as det
model = det(backbone='SSD_Lite')
model.num_classes = 80
checkpoint = 'checkpoints/COCO-80/ssdlite.pth'
out_file="out_file/COCO-80.onnx"
model.convert(checkpoint=checkpoint, backend="ONNX", out_file=out_file, class_path=class_path)
```

### 2.1 部署到硬件

- 需上传到硬件的文件：

  1）out_file文件夹（内含模型转换生成的两个文件）。
  
  2）BaseData.py，用于数据预处理。
  
  新建一个代码文件，将out_file文件夹中的py文件中的代码稍作修改用于代码运行。
