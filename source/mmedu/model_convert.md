# AI模型的转换

## 1. 模型转换的意义

模型转换是为了模型能在不同框架间流转。在实际应用时，模型转换几乎都用于工业部署，负责模型从训练框架到部署侧推理框架的连接。 这是因为随着深度学习应用和技术的演进，训练框架和推理框架的职能已经逐渐分化。 分布式、自动求导、混合精度……训练框架往往围绕着易用性，面向设计算法的研究员，以研究员能更快地生产高性能模型为目标。 硬件指令集、预编译优化、量化算法……推理框架往往围绕着硬件平台的极致优化加速，面向工业落地，以模型能更快执行为目标。由于职能和侧重点不同，没有一个深度学习框架能面面俱到，完全一统训练侧和推理侧，而模型在各个框架内部的表示方式又千差万别，所以模型转换就被广泛需要了。

## 2. 常见的AI模型类型

### 2.1 ONNX

ONNX 的全称是“Open Neural Network Exchange”，即“开放的神经网络切换”，旨在实现不同神经网络开发框架之间的互通互用。ONNXRuntime是微软推出的一款推理框架，支持多种运行后端包括CPU，GPU，TensorRT，DML等，是对ONNX模型最原生的支持。

### 2.2 NCNN

腾讯公司开发的移动端平台部署工具，一个为手机端极致优化的高性能神经网络前向计算框架。NCNN仅用于推理，不支持学习。

## 3. MMEdu的模型转换

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

## 4. 模型部署

### 4.1 部署到硬件（可以运行python的硬件）

- 硬件上需安装的库：

  onnxruntime
  
- 需上传到硬件的文件：

  1）out_file文件夹（内含模型转换生成的两个文件）。
  
  2）BaseData.py，用于数据预处理。
  
  新建一个代码文件，将out_file文件夹中的py文件中的代码稍作修改用于代码运行。

示例代码：

```
import onnxruntime as rt
import BaseData
import numpy as np
tag = ['cat', 'dog']
sess = rt.InferenceSession('out_file/catdog.onnx', None)

input_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name

dt = BaseData.ImageData('/data/TC4V0D/CatsDogsSample/test_set/cat/cat26.jpg', backbone='MobileNet')

input_data = dt.to_tensor()
pred_onx = sess.run([out_name], {input_name: input_data})
ort_output = pred_onx[0]
idx = np.argmax(ort_output, axis=1)[0]

if tag[idx] == 'dog':
    print('这是小狗，汪汪汪！')
else:
    print('这是小猫，喵喵喵！')
```

## 5. 模型部署示例

猫狗分类小助手：https://www.openinnolab.org.cn/pjlab/project?id=63c3f52a1dd9517dffa1f513&sc=62f34141bf4f550f3e926e0e#public

千物识别小助手：https://www.openinnolab.org.cn/pjlab/project?id=63c4106c2e26ff0a30cb440f&sc=62f34141bf4f550f3e926e0e#public

有无人检测小助手：https://www.openinnolab.org.cn/pjlab/project?id=63c4b6d22e26ff0a30f26ebc&sc=62f34141bf4f550f3e926e0e#public

行空板上温州话识别：https://www.openinnolab.org.cn/pjlab/project?id=63b7c66e5e089d71e61d19a0&sc=62f34141bf4f550f3e926e0e#public
