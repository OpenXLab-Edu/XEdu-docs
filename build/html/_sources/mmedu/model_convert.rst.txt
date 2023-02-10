揭秘AI模型的转换
================

为什么要进行模型转换？
----------------------

模型转换是为了模型能在不同框架间流转。在实际应用时，模型转换几乎都用于工业部署，负责模型从训练框架到部署侧推理框架的连接。
这是因为随着深度学习应用和技术的演进，训练框架和推理框架的职能已经逐渐分化。
分布式、自动求导、混合精度……训练框架往往围绕着易用性，面向设计算法的研究员，以研究员能更快地生产高性能模型为目标。
硬件指令集、预编译优化、量化算法……推理框架往往围绕着硬件平台的极致优化加速，面向工业落地，以模型能更快执行为目标。由于职能和侧重点不同，没有一个深度学习框架能面面俱到，完全一统训练侧和推理侧，而模型在各个框架内部的表示方式又千差万别，所以模型转换就被广泛需要了。

快速认识常见的AI模型类型
------------------------

1. ONNX
~~~~~~~

ONNX 的全称是“Open Neural Network
Exchange”，即“开放的神经网络切换”，旨在实现不同神经网络开发框架之间的互通互用。ONNXRuntime是微软推出的一款推理框架，支持多种运行后端包括CPU，GPU，TensorRT，DML等，是对ONNX模型最原生的支持。

2. NCNN
~~~~~~~

腾讯公司开发的移动端平台部署工具，一个为手机端极致优化的高性能神经网络前向计算框架。NCNN仅用于推理，不支持学习。

用MMEdu就能做模型转换！
-----------------------

MMEdu内置了一个\ ``convert``\ 函数实现了一键式模型转换，转换前先了解一下转换要做的事情吧。

-  转换准备：

   分类的标签文件、待转换的模型权重文件。

-  需要配置四个信息：

   待转换的模型权重文件（\ ``checkpoint``\ ），图片分类的类别数量（model.num_classes），分类标签信息文件（\ ``class_path``\ ）和输出的文件（\ ``out_file``\ ）。

-  模型转换的典型代码：

::

   from MMEdu import MMClassification as cls
   model = cls(backbone='MobileNet')
   model.num_classes = 2
   checkpoint = 'checkpoints/cls_model/CatsDog/best_accuracy_top-1_epoch_2.pth'
   out_file="out_file/catdog.onnx"
   model.convert(checkpoint=checkpoint, backend="ONNX", out_file=out_file, class_path=class_path)

这段代码是完成分类模型的转换，接下来对为您\ ``model.convert``\ 函数的各个参数：

``checkpoint``\ ：选择想要进行模型转换的权重文件，以.pth为后缀。

``backend``\ ：模型转换的后端推理框架，目前支持ONNX，后续将陆续支持NCNN、TensorRT、OpenVINO等。

``out_file``\ ：模型转换后的输出文件路径。

``class_path``\ ：模型输入的类别文件路径。

目标检测模型转换的示例代码如下：

::

   from MMEdu import MMDetection as det
   model = det(backbone='SSD_Lite')
   model.num_classes = 80
   checkpoint = 'checkpoints/COCO-80/ssdlite.pth'
   out_file="out_file/COCO-80.onnx"
   model.convert(checkpoint=checkpoint, backend="ONNX", out_file=out_file, class_path=class_path)

从零开始训练猫狗识别模型并完成模型转换
--------------------------------------

现在我们一起体验以猫狗识别为例，理解并掌握使用MMEdu工具完成从模型训练到模型部署的基本流程。

可参考的项目：https://www.openinnolab.org.cn/pjlab/project?id=63c756ad2cf359369451a617&sc=635638d69ed68060c638f979#public

（用谷歌浏览器打开）

**1.准备数据集**

思考自己想要解决的分类问题后，首先收集数据并整理好数据集，如想要解决猫狗识别问题需准备猫狗数据集。

**2.模型训练**

全新开始训练一个模型，一般要花较长时间。因此我们强烈建议在预训练模型的基础上继续训练，哪怕你要分类的数据集和预训练的数据集并不一样。如下代码使用基于MobileNet网络训练的猫狗识别预训练模型，在这个预训练模型基础上继续训练。基于预训练模型继续训练可起到加速训练的作用，通常会使得模型达到更好的效果。

::

   from MMEdu import MMClassification as cls
   model = cls(backbone='MobileNet')
   model.num_classes = 2
   model.load_dataset(path='/data/TC4V0D/CatsDogsSample') 
   model.save_fold = 'checkpoints/cls_model/CatsDog1' 
   model.train(epochs=5, checkpoint='checkpoints/pretrain_model/mobilenet_v2.pth' ,batch_size=4, lr=0.001, validate=True,device='cuda')

**3.推理部署**

使用MMEdu图像分类模块模型推理的示例代码完成模型推理。返回的数据类型是一个字典列表（很多个字典组成的列表）类型的变量，内置的字典表示分类的结果，如“``{'标签': 0, '置信度': 0.9417100548744202, '预测结果': 'cat'}``”，我们可以用字典访问其中的元素。巧用预测结果设置一些输出。如：

::

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

**4.模型转换和部署**

::

   from MMEdu import MMClassification as cls
   model = cls(backbone='MobileNet')
   checkpoint = 'checkpoints/cls_model/CatsDog1/best_accuracy_top-1_epoch_1.pth'
   model.num_classes = 2
   class_path = '/data/TC4V0D/CatsDogsSample/classes.txt'
   out_file='out_file/cats_dogs.onnx'
   model.convert(checkpoint=checkpoint, backend="ONNX", out_file=out_file, class_path=class_path)

此时项目文件中的out_file文件夹下便生成了模型转换后生成的两个文件，可打开查看。一个是ONNX模型权重，一个是示例代码，示例代码稍作改动即可运行（需配合BaseData.py的BaseDT库）。

挑战模型部署到硬件
------------------

准备工作
~~~~~~~~

-  硬件上需安装的库：

   onnxruntime

-  需上传到硬件的文件：

   1）out_file文件夹（内含模型转换生成的两个文件）。

   2）BaseData.py，用于数据预处理。

   新建一个代码文件，将out_file文件夹中的py文件中的代码稍作修改用于代码运行。

示例代码：

::

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

更多模型部署项目
----------------

猫狗分类小助手：https://www.openinnolab.org.cn/pjlab/project?id=63c3f52a1dd9517dffa1f513&sc=62f34141bf4f550f3e926e0e#public

千物识别小助手：https://www.openinnolab.org.cn/pjlab/project?id=63c4106c2e26ff0a30cb440f&sc=62f34141bf4f550f3e926e0e#public

有无人检测小助手：https://www.openinnolab.org.cn/pjlab/project?id=63c4b6d22e26ff0a30f26ebc&sc=62f34141bf4f550f3e926e0e#public

行空板上温州话识别：https://www.openinnolab.org.cn/pjlab/project?id=63b7c66e5e089d71e61d19a0&sc=62f34141bf4f550f3e926e0e#public
