# “最后一公里”—XEdu助力AI模型部署

得益与`OpenMMLab`系列工具的不断进步与发展。MMEdu通过集成OpenMMLab开源的`模型部署工具箱MMDeploy`和`模型压缩工具包MMRazor`，打通了从算法模型到应用程序这 “最后一公里”！ 今天我们将开启`AI模型部署`入门系列教程，在面向中小学AI教育的开发和学习工具 `XEdu` 的辅助下，介绍以下内容：

- 模型转换
- 模型量化
- 多模态交互

> MMEdu已经可以帮助我diy自己的AI模型了，为什么要多此一举、徒增难度，来学习需要更多编程知识的模型部署模块？

初学新知识，我们总会有困惑与畏难，好奇这项技术能否为我所用、产生效益，担心过程是否困难，付出与收获不成正比。没有关系！请让自己的思考凝聚成三个层面，即`黄金圈法则：Why、How、What`，来从如下的行文中去寻找答案。

希望通过本系列教程，带领大家学会如何把自己使用`MMEdu`训练的计算机视觉任务`SOTA模型`部署到`ONNXRuntime`、`NCNN`等各个推理引擎上。我们默认大家熟悉 Python 语言，除此之外不需要了解任何模型部署的知识。

## Why：为什么
### 为什么要进行模型转换

模型转换是为了模型能在不同框架间流转。在实际应用时，模型转换几乎都用于工业部署，负责模型从训练框架到部署侧推理框架的连接。 这是因为随着深度学习应用和技术的演进，训练框架和推理框架的职能已经逐渐分化。 分布式、自动求导、混合精度……训练框架往往围绕着易用性，面向设计算法的研究员，以研究员能更快地生产高性能模型为目标。 硬件指令集、预编译优化、量化算法……推理框架往往围绕着硬件平台的极致优化加速，面向工业落地，以模型能更快执行为目标。由于职能和侧重点不同，没有一个深度学习框架能面面俱到，完全一统训练侧和推理侧，而模型在各个框架内部的表示方式又千差万别，所以模型转换就被广泛需要了。

> 概括：训练框架大，塞不进两三百块钱买的硬件设备中，推理框架小，能在硬件设备上安装。要把训练出的模型翻译成推理框架能读懂的语言，才能在硬件设备上运行

### 为什么要进行模型量化
模型量化是指将深度学习模型中的参数、激活值等数据转化为更小的数据类型（通常是8位整数或者浮点数），以达到模型大小减小、计算速度加快、内存占用减小等优化目的的技术手段。模型量化有以下几个优点：减小模型大小、加速模型推理、减少内存占用等。因此，模型量化可以帮助提高深度学习模型的效率和性能，在实际应用中具有重要的价值和意义。

> 概括：对模型采用合适的量化，能在对准确率忽略不计的情况下，让模型更小、更快、更轻量。比如原先168 MB的模型量化后大小变为了42.6 MB，推理速度提高了两倍。

### 为什么要进行多模态交互
多模态交互是指利用多个感知通道（例如语音、图像、触觉、姿态等）进行交互的技术。多模态交互在人机交互、智能交通、健康医疗、教育培训等领域都有广泛的应用、在提高交互效率、用户体验、解决单模态限制和实现智能化交互等方面具有重要的作用和价值。
>概括：给你的AI作品加点创客料
>

### 什么是推理框架
深度学习推理框架是一种让深度学习算法在实时处理环境中提高性能的框架。常见的有[ONNXRuntime](https://github.com/microsoft/onnxruntime)、[NCNN](https://github.com/Tencent/ncnn)、[TensorRT](https://github.com/NVIDIA/TensorRT)、[OpenVINO](https://github.com/openvinotoolkit/openvino)等。
> ONNXRuntime是微软推出的一款推理框架，支持多种运行后端包括CPU，GPU，TensorRT，DML等，是对ONNX模型最原生的支持。
> 
> NCNN是腾讯公司开发的移动端平台部署工具，一个为手机端极致优化的高性能神经网络前向计算框架。NCNN仅用于推理，不支持学习。

__值得注意的是，包括Pytorch、Tensorflow，以及国内的百度PaddlePaddle、华为的MindSpore等主流的深度学习框架都开发了工具链来回应这个Why。我们采用业界主流的方法，以更高代码封装度的形式来解决这一问题。接下来，且听我们对利用`XEdu`进行`How怎么做`的流程娓娓道来。__

## How：怎么做
总结一下Why中的回应，在软件工程中，部署指把开发完毕的软件投入使用的过程，包括环境配置、软件安装等步骤。类似地，对于深度学习模型来说，模型部署指让训练好的模型在特定环境中运行的过程。相比于软件部署，模型部署会面临更多的难题：

> 1. 运行模型所需的环境难以配置。深度学习模型通常是由一些框架编写，比如 PyTorch、TensorFlow。由于框架规模、依赖环境的限制，这些框架不适合在手机、开发板等生产环境中安装。 
> 2. 深度学习模型的结构通常比较庞大，需要大量的算力才能满足实时运行的需求。模型的运行效率需要优化。 因为这些难题的存在，模型部署不能靠简单的环境配置与安装完成。`


经过工业界和学术界数年的探索，结合`XEdu`的工具，展示模型部署一条流行的流水线：

<div align="center">
	<img src="../images/model_convert/XEdu模型部署全链路pipeline.JPG" width="75%">
</div>

这一条流水线解决了模型部署中的两大问题：使用对接深度学习框架和推理引擎的中间表示，开发者不必担心如何在新环境中运行各个复杂的框架；通过中间表示的网络结构优化和推理引擎对运算的底层优化，模型的运算效率大幅提升。

### 用MMEdu进行模型转换
MMEdu内置了一个`convert`函数实现了一键式模型转换，转换前先了解一下转换要做的事情吧。

- 转换准备：

  分类的标签文件、待转换的模型权重文件。

- 需要配置四个信息：

  待转换的模型权重文件（`checkpoint`），图片分类的类别数量（model.num_classes），分类标签信息文件（`class_path`）和输出的文件（`out_file`）。

- 模型转换的典型代码：

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

类似的，目标检测模型转换的示例代码如下：

```
from MMEdu import MMDetection as det
model = det(backbone='SSD_Lite')
model.num_classes = 80
checkpoint = 'checkpoints/COCO-80/ssdlite.pth'
out_file="out_file/COCO-80.onnx"
model.convert(checkpoint=checkpoint, backend="ONNX", out_file=out_file, class_path=class_path)
```

现在，让我们从“[从零开始训练猫狗识别模型并完成模型转换](https://www.openinnolab.org.cn/pjlab/project?id=63c756ad2cf359369451a617&sc=635638d69ed68060c638f979#public)”项目入手，见识一下使用MMEdu工具完成从模型训练到模型部署的基本流程吧！

**1.准备数据集**

思考自己想要解决的分类问题后，首先收集数据并整理好数据集，如想要解决猫狗识别问题需准备猫狗数据集。

**2.模型训练**

全新开始训练一个模型，一般要花较长时间。因此我们强烈建议在预训练模型的基础上继续训练，哪怕你要分类的数据集和预训练的数据集并不一样。如下代码使用基于MobileNet网络训练的猫狗识别预训练模型，在这个预训练模型基础上继续训练。基于预训练模型继续训练可起到加速训练的作用，通常会使得模型达到更好的效果。

```
from MMEdu import MMClassification as cls
model = cls(backbone='MobileNet')
model.num_classes = 2
model.load_dataset(path='/data/TC4V0D/CatsDogsSample') 
model.save_fold = 'checkpoints/cls_model/CatsDog1' 
model.train(epochs=5, checkpoint='checkpoints/pretrain_model/mobilenet_v2.pth' ,batch_size=4, lr=0.001, validate=True,device='cuda')
```

**3.推理部署**

使用MMEdu图像分类模块模型推理的示例代码完成模型推理。返回的数据类型是一个字典列表（很多个字典组成的列表）类型的变量，内置的字典表示分类的结果，如“`{'标签': 0, '置信度': 0.9417100548744202, '预测结果': 'cat'}`”，我们可以用字典访问其中的元素。巧用预测结果设置一些输出。如：

```
from MMEdu import MMClassification as cls
model = cls(backbone='MobileNet')
checkpoint = 'checkpoints/cls_model/CatsDog1/best_accuracy_top-1_epoch_1.pth'
class_path = '/data/TC4V0D/CatsDogsSample/classes.txt'
img_path = '/data/TC4V0D/CatsDogsSample/test_set/cat/cat0.jpg'
result = model.inference(image=img_path, show=True, class_path=class_path,checkpoint = checkpoint,device='cuda')
x = model.print_result(result)
print('标签（序号）为：',x[0]['标签'])
if x[0]['标签'] == 0:
    print('这是小猫，喵喵喵！')
else:
    print('这是小猫，喵喵喵！')
```

**4.模型转换**

```
from MMEdu import MMClassification as cls
model = cls(backbone='MobileNet')
checkpoint = 'checkpoints/cls_model/CatsDog1/best_accuracy_top-1_epoch_1.pth'
model.num_classes = 2
class_path = '/data/TC4V0D/CatsDogsSample/classes.txt'
out_file='out_file/cats_dogs.onnx'
model.convert(checkpoint=checkpoint, backend="ONNX", out_file=out_file, class_path=class_path)
```

此时项目文件中的out_file文件夹下便生成了模型转换后生成的两个文件，可打开查看。一个是ONNX模型权重，一个是示例代码，示例代码稍作改动即可运行（需配合BaseData.py的BaseDT库）。

**5.模型部署**

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

## What：什么现象与成果

### 精度测试结果
#### 软硬件环境
- 操作系统：Ubuntu 16.04
- 系统位数：64
- 处理器：Intel i7-11700 @ 2.50GHz * 16
- 显卡：GeForce GTX 1660Ti
- 推理框架：ONNXRuntime == 1.13.1
- 数据处理工具：BaseDT == 0.0.1

#### 配置
- `静态图`导出
- `batch`大小为1
- `BaseDT`内置`ImageData`工具进行数据预处理
#### 精度测试结果汇总
- 图像分类
<table class="tg">
    <thead>
  <tr>
    <th rowspan="2">模型</th>
    <th rowspan="2">数据集</th>
    <th rowspan="1" colspan="2">权重大小</th>
    <th rowspan="1" colspan="2">精度（TOP-1）</th>
    <th rowspan="1" colspan="2">精度（TOP-5）</th>
  </tr>
  <tr>
    <th colspan="1">FP32</th>
    <th colspan="1">INT8</th>
    <th colspan="1">FP32</th>
    <th colspan="1">INT8</th>
    <th colspan="1">FP32</th>
    <th colspan="1">INT8</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td class="tg-zk71">MobileNet</td>
    <td><a href="http://www.image-net.org/challenges/LSVRC/2012/">ImageNet</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/mobilenet/model/mobilenetv2-10.onnx">13.3 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/mobilenet/model/mobilenetv2-12-int8.onnx">3.5 MB</a> </td>
    <td>70.94%</td>
    <td>68.30%</td>
    <td>89.99%</td>
    <td>88.44%</td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">ResNet18</td>
    <td><a href="http://www.image-net.org/challenges/LSVRC/2012/">ImageNet</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet18-v1-7.onnx">44.7 MB</a></td>
    <td></td>
    <td>69.93%</td>
    <td></td>
    <td>89.29%</td>
    <td></td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">ResNet50</td>
    <td><a href="http://www.image-net.org/challenges/LSVRC/2012/">ImageNet</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v1-7.onnx">97.8 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v1-12-int8.onnx">24.6 MB</a></td>
    <td>74.93%</td>
    <td>74.77%</td>
    <td>92.38%</td>
    <td>92.32%</td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">ShuffleNet_v2</td>
    <td><a href="http://www.image-net.org/challenges/LSVRC/2012/">ImageNet</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/shufflenet/model/shufflenet-v2-10.onnx">9.2 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/shufflenet/model/shufflenet-v2-12-int8.onnx">2.28 MB</a></td>
    <td>69.36%</td>
    <td>66.15%</td>
    <td>88.32%</td>
    <td>86.34%</td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">VGG</td>
    <td><a href="http://www.image-net.org/challenges/LSVRC/2012/">ImageNet</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/vgg/model/vgg16-7.onnx">527.8 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/vgg/model/vgg16-12-int8.onnx">101.1 MB</a></td>
    <td>72.62%</td>
    <td>72.32%</td>
    <td>91.14%</td>
    <td>90.97%</td>
  </tr>
</tbody>
</table>

> ImageNet 数据集：ImageNet项目是一个用于视觉对象识别软件研究的大型可视化数据库。ImageNet项目每年举办一次软件比赛，即`ImageNet大规模视觉识别挑战赛`（ILSVRC），软件程序竞相正确分类检测物体和场景。 ImageNet挑战使用了一个“修剪”的1000个非重叠类的列表。2012年在解决ImageNet挑战方面取得了巨大的突破
> 
> 准确度（Top-1）：排名第一的类别与实际结果相符的准确率
> 
> 准确度（Top-5）：排名前五的类别包含实际结果的准确率
> 
- 目标检测
<table class="tg">
    <thead>
  <tr>
    <th rowspan="2">模型</th>
    <th rowspan="2">数据集</th>
    <th rowspan="1" colspan="2">权重大小</th>
    <th rowspan="1" colspan="2">精度（mAP）</th>
  </tr>
  <tr>
    <th colspan="1">FP32</th>
    <th colspan="1">INT8</th>
    <th colspan="1">FP32</th>
    <th colspan="1">INT8</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td class="tg-zk71">SSD_Lite</td>
    <td><a href="https://cocodataset.org/#home">COCO</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx">28.1 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12-int8.onnx">8.5 MB</a> </td>
    <td>0.2303</td>
    <td>0.2285</td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">FasterRCNN</td>
    <td><a href="https://cocodataset.org/#home">COCO</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12.onnx">168.5 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12-int8.onnx">42.6 MB</a></td>
    <td>0.3437</td>
    <td>0.3399</td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">Mask_RCNN</td>
    <td><a href="https://cocodataset.org/#home">COCO</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-12.onnx">169.7 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-12-int8.onnx">45.9 MB</a></td>
    <td>0.3372</td>
    <td>0.3340</td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">Yolov3</td>
    <td><a href="https://cocodataset.org/#home">COCO</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov3/model/yolov3-12.onnx">237 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov3/model/yolov3-12-int8.onnx">61 MB</a></td>
    <td>0.2874</td>
    <td>0.2688</td>
  </tr>
</tbody>
</table>

> COCO 数据集: MS COCO的全称是`Microsoft Common Objects in Context`，起源于微软于2014年出资标注的Microsoft COCO数据集，与ImageNet竞赛一样，被视为是计算机视觉领域最受关注和最权威的比赛之一。 COCO数据集是一个大型的、丰富的物体检测，分割和字幕数据集。这个数据集以scene understanding为目标，目前为止有语义分割的最大数据集，提供的类别有80 类，有超过33 万张图片，其中20 万张有标注，整个数据集中个体的数目超过150 万个。
> 
>AP (average Precision)：平均精度，在不同recall下的最高precision的均值(一般会对各类别分别计算各自的AP)
> 
> mAP（mean AP）:平均精度的均值，各类别的AP的均值
> 

### 边、端设备测试结果

#### 行空板测试
> 行空板, 青少年Python教学用开源硬件，解决Python教学难和使用门槛高的问题，旨在推动Python教学在青少年中的普及。官网：https://www.dfrobot.com.cn/
##### 软硬件环境
- 操作系统：Linux
- 系统位数：64
- 处理器：4核单板AArch64 1.20GHz
- 内存：512MB
- 硬盘：16GB
- 推理框架：ONNXRuntime == 1.13.1
- 数据处理工具：BaseDT == 0.0.1
##### 配置
- `静态图`导出
- `batch`大小为1
- `BaseDT`内置`ImageData`工具进行数据预处理 
- 测试时，计算各个数据集中 10 张图片的平均耗时

下面是我们环境中的测试结果：
- 图像分类
<table class="tg">
    <thead>
  <tr>
    <th rowspan="2">模型</th>
    <th rowspan="2">数据集</th>
    <th rowspan="1" colspan="2">权重大小</th>
    <th rowspan="1" colspan="2">吞吐量 (图片数/每秒) </th>
  </tr>
  <tr>
    <th colspan="1">FP32</th>
    <th colspan="1">INT8</th>
    <th colspan="1">FP32</th>
    <th colspan="1">INT8</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td class="tg-zk71">MobileNet</td>
    <td><a href="http://www.image-net.org/challenges/LSVRC/2012/">ImageNet</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/mobilenet/model/mobilenetv2-10.onnx">13.3 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/mobilenet/model/mobilenetv2-12-int8.onnx">3.5 MB</a> </td>
    <td>1.77</td>
    <td>4.94</td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">ResNet18</td>
    <td><a href="http://www.image-net.org/challenges/LSVRC/2012/">ImageNet</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet18-v1-7.onnx">44.7 MB</a></td>
    <td></td>
    <td>0.46</td>
    <td></td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">ResNet50</td>
    <td><a href="http://www.image-net.org/challenges/LSVRC/2012/">ImageNet</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v1-7.onnx">97.8 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v1-12-int8.onnx">24.6 MB</a></td>
    <td>0.22</td>
    <td>0.58</td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">ShuffleNet_v2</td>
    <td><a href="http://www.image-net.org/challenges/LSVRC/2012/">ImageNet</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/shufflenet/model/shufflenet-v2-10.onnx">9.2 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/shufflenet/model/shufflenet-v2-12-int8.onnx">2.28 MB</a></td>
    <td>3.97</td>
    <td>8.51</td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">VGG</td>
    <td><a href="http://www.image-net.org/challenges/LSVRC/2012/">ImageNet</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/vgg/model/vgg16-7.onnx">527.8 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/vgg/model/vgg16-12-int8.onnx">101.1 MB</a></td>
    <td>*</td>
    <td>*</td>
  </tr>
</tbody>
</table>

​    吞吐量 (图片数/每秒)：表示每秒模型能够识别的图片总数，常用来评估模型的表现

> *：不建议部署，单张图片推理的时间超过30s

- 目标检测
<table class="tg">
    <thead>
   <tr>
    <th rowspan="2">模型</th>
    <th rowspan="2">数据集</th>
    <th rowspan="1" colspan="2">权重大小</th>
    <th rowspan="1" colspan="2">吞吐量 (图片数/每秒) </th>
  </tr>
  <tr>
    <th colspan="1">FP32</th>
    <th colspan="1">INT8</th>
    <th colspan="1">FP32</th>
    <th colspan="1">INT8</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td class="tg-zk71">SSD_Lite<sup>*</sup></td>
    <td><a href="https://cocodataset.org/#home">COCO</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx">28.1 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12-int8.onnx">8.5 MB</a> </td>
    <td>0.55</td>
    <td>1.30</td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">SSD_Lite<sup>**</sup></td>
    <td><a href="https://cocodataset.org/#home">COCO</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx">28.1 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12-int8.onnx">8.5 MB</a> </td>
    <td></td>
    <td></td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">FasterRCNN</td>
    <td><a href="https://cocodataset.org/#home">COCO</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12.onnx">168.5 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12-int8.onnx">42.6 MB</a></td>
    <td></td>
    <td></td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">Mask_RCNN</td>
    <td><a href="https://cocodataset.org/#home">COCO</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-12.onnx">169.7 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-12-int8.onnx">45.9 MB</a></td>
    <td></td>
    <td></td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">Yolov3</td>
    <td><a href="https://cocodataset.org/#home">COCO</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov3/model/yolov3-12.onnx">237 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov3/model/yolov3-12-int8.onnx">61 MB</a></td>
    <td>0.026</td>
    <td>0.066</td>
  </tr>
</tbody>
</table>

>*：后端支持网络为MobileNetv1，性能弱于以MobileNetv2为后端推理框架的版本
> 
> **：后端支持网络为MobileNetv2，即MMEdu中SSD_Lite选用的版本，可从参数对比中得出其精度、准确度、模型大小均优于以MobileNetv1为后端推理框架的SSD_Lite

#### 树莓派（4b）测试
> Raspberry Pi。中文名为“树莓派”,简写为RPi，或者RasPi/RPi)是为学生计算机编程教育而设计，卡片式电脑，其系统基于Linux。
##### 软硬件环境
- 操作系统：Linux
- 系统位数：32
- 处理器：BCM2711 四核 Cortex-A72(ARM v8) @1.5GHz
- 内存：4G
- 硬盘：16G
- 推理框架：ONNXRuntime == 1.13.1
- 数据处理工具：BaseDT == 0.0.1
##### 配置
- `静态图`导出
- `batch`大小为1
- `BaseDT`内置`ImageData`工具进行数据预处理 
- 测试时，计算各个数据集中 10 张图片的平均耗时

下面是我们环境中的测试结果：
- 图像分类
<table class="tg">
    <thead>
  <tr>
    <th rowspan="2">模型</th>
    <th rowspan="2">数据集</th>
    <th rowspan="1" colspan="2">权重大小</th>
    <th rowspan="1" colspan="2">吞吐量 (图片数/每秒) </th>
  </tr>
  <tr>
    <th colspan="1">FP32</th>
    <th colspan="1">INT8</th>
    <th colspan="1">FP32</th>
    <th colspan="1">INT8</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td class="tg-zk71">MobileNet</td>
    <td><a href="http://www.image-net.org/challenges/LSVRC/2012/">ImageNet</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/mobilenet/model/mobilenetv2-10.onnx">13.3 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/mobilenet/model/mobilenetv2-12-int8.onnx">3.5 MB</a> </td>
    <td>6.45</td>
    <td></td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">ResNet18</td>
    <td><a href="http://www.image-net.org/challenges/LSVRC/2012/">ImageNet</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet18-v1-7.onnx">44.7 MB</a></td>
    <td></td>
    <td>3.20</td>
    <td></td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">ResNet50</td>
    <td><a href="http://www.image-net.org/challenges/LSVRC/2012/">ImageNet</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v1-7.onnx">97.8 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v1-12-int8.onnx">24.6 MB</a></td>
    <td>1.48</td>
    <td>2.91</td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">ShuffleNet_v2</td>
    <td><a href="http://www.image-net.org/challenges/LSVRC/2012/">ImageNet</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/shufflenet/model/shufflenet-v2-10.onnx">9.2 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/shufflenet/model/shufflenet-v2-12-int8.onnx">2.28 MB</a></td>
    <td>19.11</td>
    <td>10.85<cup>*</cup></td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">VGG</td>
    <td><a href="http://www.image-net.org/challenges/LSVRC/2012/">ImageNet</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/vgg/model/vgg16-7.onnx">527.8 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/classification/vgg/model/vgg16-12-int8.onnx">101.1 MB</a></td>
    <td>0.43</td>
    <td>0.44</td>
  </tr>
</tbody>
</table>

> 吞吐量 (图片数/每秒)：表示每秒模型能够识别的图片总数，常用来评估模型的表现
> 
> *：量化后在树莓派上推理速度变慢

- 目标检测
<table class="tg">
    <thead>
   <tr>
    <th rowspan="2">模型</th>
    <th rowspan="2">数据集</th>
    <th rowspan="1" colspan="2">权重大小</th>
    <th rowspan="1" colspan="2">吞吐量 (图片数/每秒) </th>
  </tr>
  <tr>
    <th colspan="1">FP32</th>
    <th colspan="1">INT8</th>
    <th colspan="1">FP32</th>
    <th colspan="1">INT8</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td class="tg-zk71">SSD_Lite<sup>*</sup></td>
    <td><a href="https://cocodataset.org/#home">COCO</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx">28.1 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12-int8.onnx">8.5 MB</a> </td>
    <td>2.55</td>
    <td></td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">SSD_Lite<sup>**</sup></td>
    <td><a href="https://cocodataset.org/#home">COCO</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx"></a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12-int8.onnx"></a></td>
    <td></td>
    <td></td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">FasterRCNN</td>
    <td><a href="https://cocodataset.org/#home">COCO</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12.onnx">168.5 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12-int8.onnx">42.6 MB</a></td>
    <td></td>
    <td></td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">Mask_RCNN</td>
    <td><a href="https://cocodataset.org/#home">COCO</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-12.onnx">169.7 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-12-int8.onnx">45.9 MB</a></td>
    <td></td>
    <td></td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td class="tg-zk71">Yolov3</td>
    <td><a href="https://cocodataset.org/#home">COCO</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov3/model/yolov3-12.onnx">237 MB</a></td>
    <td><a href="https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/yolov3/model/yolov3-12-int8.onnx">61 MB</a></td>
    <td>0.21</td>
    <td>0.34</td>
  </tr>
</tbody>
</table>

>*：后端支持网络为MobileNetv1，性能弱于以MobileNetv2为后端推理框架的版本
> 
> **：后端支持网络为MobileNetv2，即MMEdu中SSD_Lite选用的版本，可从参数对比中得出其精度、准确度、模型大小均优于以MobileNetv1为后端推理框架的SSD_Lite

__注：硬件测试模块持续更新中，如有更多硬件测试需求，请[联系我们](https://github.com/OpenXLab-Edu/XEdu-docs/issues)__

## 多模态交互
回顾用AI解决真实问题的流程图，我们已经介绍了收集数据、训练模型、模型推理和应用部署。结合项目设计，我们还会去思考如何通过摄像头获得图像，如何控制灯光发亮，如何操纵舵机，如何设计显示界面UI等需要使用输入设备和输出设备等来实现的交互设计，即对`多模态交互`的考量。

<div align="center">
	<img src="../images/model_convert/用AI解决真实问题.JPG" width="80%">
</div>

更多传感器、执行器使用教程参见：[DFRobot](https://wiki.dfrobot.com.cn/)


## 更多模型部署项目

猫狗分类小助手：https://www.openinnolab.org.cn/pjlab/project?id=63c3f52a1dd9517dffa1f513&sc=62f34141bf4f550f3e926e0e#public

千物识别小助手：https://www.openinnolab.org.cn/pjlab/project?id=63c4106c2e26ff0a30cb440f&sc=62f34141bf4f550f3e926e0e#public

有无人检测小助手：https://www.openinnolab.org.cn/pjlab/project?id=63c4b6d22e26ff0a30f26ebc&sc=62f34141bf4f550f3e926e0e#public

行空板上温州话识别：https://www.openinnolab.org.cn/pjlab/project?id=63b7c66e5e089d71e61d19a0&sc=62f34141bf4f550f3e926e0e#public

树莓派与MMEdu：https://www.openinnolab.org.cn/pjlab/project?id=63bb8be4c437c904d8a90350&backpath=/pjlab/projects/list?backpath=/pjlab/ai/projects#public

MMEdu模型在线转换：https://www.openinnolab.org.cn/pjlab/project?id=63a1a47e5e089d71e6c6f068&backpath=/pjlab/projects/list?backpath=/pjlab/ai/projects#public

