# 模型转换和应用

## 一、简介

用XEdu系列工具训练的模型，是否只能运行在安装了XEdu环境的电脑上？如何将训练好的AI模型方便地部署到不同的硬件设备上？这在实际应用中非常重要。XEdu提供了帮助模型转换和应用的工具。

## 二、基本概念

1.模型转换（Model Convert ）：为了让训练好的模型能在不同框架间流转，通常需要将模型从训练框架转换为推理框架。这样可以在各种硬件设备上部署模型，提高模型的通用性和实用性。

2.模型应用（Model Applying ）：在实际问题中使用训练好的模型进行预测和分析。这通常涉及到数据预处理、模型输入、模型输出解释等步骤。模型应用的目标是将深度学习技术与实际业务场景相结合，以解决实际问题，提高工作效率和准确性。

3.模型部署（Model Deploying ）：将训练好的模型应用到实际场景中，如手机、开发板等。模型部署需要解决环境配置、运行效率等问题。部署过程中，可能需要对模型进行优化，以适应特定的硬件和软件环境，确保模型在实际应用中的性能和稳定性。

4.深度学习推理框架：一种让深度学习算法在实时处理环境中提高性能的框架。常见的有<a href="https://github.com/microsoft/onnxruntime">ONNXRuntime</a>、<a href="https://github.com/Tencent/ncnn">NCNN</a>、<a href="https://github.com/NVIDIA/TensorRT">TensorRT</a>、<a href="https://github.com/openvinotoolkit/openvino">OpenVINO</a>等。ONNXRuntime是微软推出的一款推理框架，支持多种运行后端包括CPU，GPU，TensorRT，DML等，是对ONNX模型最原生的支持。NCNN是腾讯公司开发的移动端平台部署工具，一个为手机端极致优化的高性能神经网络前向计算框架。NCNN仅用于推理，不支持学习。

### 为什么要进行模型转换？

模型转换的目的是让训练好的模型能在不同框架间流转。在实际应用中，模型转换主要用于工业部署，负责将模型从训练框架迁移到推理框架。这是因为随着深度学习应用和技术的发展，训练框架和推理框架的职能已经逐渐分化。训练框架主要关注易用性和研究员的需求，而推理框架关注硬件平台的优化加速，以实现更快的模型执行。由于它们的职能和侧重点不同，没有一个深度学习框架能完全满足训练和推理的需求，因此模型转换变得非常重要。

**概括：** 训练框架大，塞不进两三百块钱买的硬件设备中，推理框架小，能在硬件设备上安装。要把训练出的模型翻译成推理框架能读懂的语言，才能在硬件设备上运行

## 三、如何进行模型转换？

我们可以直接使用MMEdu、BaseNN的convert函数进行一键式模型转换。

### 1.MMEdu模型转换

MMEdu内置了一个`convert`函数，来实现了一键式模型转换，转换前先了解一下转换要做的事情吧。

- 转换准备：

  待转换的模型权重文件（用MMEdu训练）。

- 需要配置两个信息：

  待转换的模型权重文件（`checkpoint`）和输出的文件（`out_file`）。

- 模型转换的典型代码：

```
from MMEdu import MMClassification as cls
model = cls(backbone='MobileNet')
checkpoint = 'checkpoints/cls_model/CatsDog/best_accuracy_top-1_epoch_2.pth'
out_file="catdog.onnx"
model.convert(checkpoint=checkpoint, out_file=out_file)
```

这段代码是完成分类模型的转换，接下来对为您`model.convert`函数的各个参数：

`checkpoint`：选择想要进行模型转换的权重文件，以.pth为后缀。

`out_file`：模型转换后的输出文件路径。


类似的，目标检测模型转换的示例代码如下：

```
from MMEdu import MMDetection as det
model = det(backbone='SSD_Lite')
checkpoint = 'checkpoints/COCO-80/ssdlite.pth'
out_file="COCO-80.onnx"
model.convert(checkpoint=checkpoint, out_file=out_file)
```

参考项目：<a href="https://www.openinnolab.org.cn/pjlab/project?id=645110943c0e930cb55e859b&sc=62f34141bf4f550f3e926e0e#public">MMEdu模型转换
</a>

## 2.BaseNN模型转换

BaseNN内置了一个`convert`函数，来实现了一键式模型转换，转换前先了解一下转换要做的事情吧。

- 转换准备：

  待转换的模型权重文件（用BaseNN训练）。

- 需要配置两个信息：

  待转换的模型权重文件（`checkpoint`）和输出的文件（`out_file`）。

- 模型转换的典型代码：

```python
from BaseNN import nn
model = nn()
model.convert(checkppint="basenn_cd.pth",out_file="basenn_cd.onnx")
```

`model.convert()`参数信息：

`checkpoint`: 指定要转换的pth模型文件路径

`out_file`: 指定转换出的onnx模型文件路径

## 四、如何快速进行模型应用？

将转换后的模型应用于实际问题时，一般需要编写代码来加载模型、输入数据、执行预测并处理输出。这可能涉及到将输入数据转换为模型所需的格式，以及将模型的输出转换为可理解的结果。例如，在图像分类任务中，你可能需要将图像转换为张量，然后将其输入到模型中，最后将模型的输出转换为类别标签。

使用XEdu工具转换后生成的示例代码和ONNX模型，您可以轻松地完成模型的快速应用。

### MMEdu模型转换后示例代码

敬请期待。

### BaseNN模型转换后示例代码

敬请期待。

## 五、模型应用和部署

模型应用和部署是将训练好的模型应用于实际场景的过程。这通常包括以下几个步骤：

1. 选择硬件和软件环境：根据实际应用需求，选择合适的硬件（如CPU、GPU、FPGA等）和软件环境（如操作系统、编程语言、库等）。
2. 模型优化：为了提高模型在实际应用中的性能，可能需要对模型进行优化，例如量化、剪枝、蒸馏等。这些优化方法可以减小模型大小、降低计算复杂度，从而提高运行效率。
3. 接口设计：为了方便其他开发者或系统调用和使用模型，需要设计合适的接口。这可能包括API设计、封装模型为库或服务等。
4. 监控和维护：在模型部署后，需要对其进行监控和维护，确保模型在实际应用中的稳定性和准确性。这可能包括日志记录、性能监控、模型更新等。
5. 安全性：在部署模型时，需要考虑数据安全和隐私保护。确保数据在传输和存储过程中的安全性，以及模型在处理敏感信息时的合规性。

通过遵循这些步骤，您可以将模型成功部署到实际应用场景中，实现模型的价值。在下面的示例代码中，我们将展示如何将转换后的模型应用到实际问题中。

### 1.连接摄像头

```
import cv2
from XEdu.hub import Workflow as wf
mmcls = wf(task='mmedu',checkpoint='cats_dogs.onnx')
cap = cv2.VideoCapture(0)
ret, img = cap.read()
result=  mmcls.inference(data=img)
format_result = mmcls.format_output(lang="zh")
cap.release()
```