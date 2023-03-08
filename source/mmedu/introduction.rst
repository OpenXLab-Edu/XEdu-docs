MMEdu基本功能
=============

MMEdu功能简介
-------------

MMEdu是一个计算机视觉方向的深度学习开发工具，是一个用来训练AI模型的工具。

MMEdu和常见AI框架的比较
-----------------------

1）MMEdu和OpenCV的比较
~~~~~~~~~~~~~~~~~~~~~~

OpenCV是一个开源的计算机视觉框架，MMEdu的核心模块MMCV基于OpenCV，二者联系紧密。

OpenCV虽然是一个很常用的工具，但是普通用户很难在OpenCV的基础上训练自己的分类器。MMEdu则是一个入门门槛很低的深度学习开发工具，借助MMEdu和经典的网络模型，只要拥有一定数量的数据，连小学生都能训练出自己的个性化模型。

2）MMEdu和MediaPipe的比较
~~~~~~~~~~~~~~~~~~~~~~~~~

MediaPipe 是一款由 Google Research
开发并开源的多媒体机器学习模型应用框架，支持人脸识别、手势识别和表情识别等，功能非常强大。MMEdu中的MMPose模块关注的重点也是手势识别，功能类似。但MediaPipe是应用框架，而不是开发框架。换句话说，用MediaPipe只能完成其提供的AI识别功能，没办法训练自己的个性化模型。

3）MMEdu和Keras的比较
~~~~~~~~~~~~~~~~~~~~~

Keras是一个高层神经网络API，是对Tensorflow、Theano以及CNTK的进一步封装。OpenMMLab和Keras一样，都是为支持快速实验而生。MMEdu则源于OpenMMLab，其语法设计借鉴过Keras。

相当而言，MMEdu的语法比Keras更加简洁，对中小学生来说也更友好。目前MMEdu的底层框架是Pytorch，而Keras的底层是TensorFlow（虽然也有基于Pytorch的Keras）。

4）MMEdu和FastAI的比较
~~~~~~~~~~~~~~~~~~~~~~

FastAI（Fast.ai）最受学生欢迎的MOOC课程平台，也是一个PyTorch的顶层框架。和OpenMMLab的做法一样，为了让新手快速实施深度学习，FastAI团队将知名的SOTA模型封装好供学习者使用。

FastAI同样基于Pytorch，但是和OpenMMLab不同的是，FastAI只能支持GPU。考虑到中小学的基础教育中很难拥有GPU环境，MMEdu特意将OpenMMLab中支持CPU训练的工具筛选出来，供中小学生使用。

MMEdu基于OpenMMLab的基础上开发，因为面向中小学，优先选择支持CPU训练的模块。

MMEdu的内置模块概述
-------------------

================ ====== ================
模块名称         简称   功能
================ ====== ================
MMClassification MMCLS  图片分类
MMDetection      MMDET  图片中的物体检测
MMGeneration     MMGEN  GAN，风格化
MMPose           MMPOSE 骨架
MMEditing              
MMSegmentation          像素级识别
================ ====== ================

MMEdu的内置SOTA模型
-------------------

MMEdu内置了常见的SOTA模型，我们还在不断更新中。如需查看所有支持的SOTA模型，可使用\ ``model.sota()``\ 代码进行查看。

================ =============================================================================================================================================================================================================================================================================================================================================== ================
模块名称         内置模型                                                                                                                                                                                                                                                                                                                                        功能
================ =============================================================================================================================================================================================================================================================================================================================================== ================
MMClassification `LeNet <https://xedu.readthedocs.io/zh/latest/dl_library/net/lenet5.html>`__\ 、\ `ResNet18 <https://xedu.readthedocs.io/zh/latest/dl_library/net/ResNet.html>`__\ 、\ `ResNet50 <https://xedu.readthedocs.io/zh/latest/dl_library/net/ResNet.html>`__\ 、\ `MobileNet <https://xedu.readthedocs.io/zh/latest/dl_library/net/mobilenet.html>`__ 图片分类
MMDetection      `FastRCNN <https://xedu.readthedocs.io/zh/latest/dl_library/net/FasterRCNN.html>`__\ 、\ `SSD_Lite <https://xedu.readthedocs.io/zh/latest/dl_library/net/SSD_Lite.html>`__\ 、\ `Yolov3 <https://xedu.readthedocs.io/zh/latest/dl_library/net/Yolov3.html>`__                                                                                   图片中的物体检测
================ =============================================================================================================================================================================================================================================================================================================================================== ================

注：对于这些SOTA模型的作用，请参考本文档的“深度学习知识库”部分的“经典网络模型介绍”。当然，通过“AI模型
+ 关键词”的形式，你在很多搜索引擎中都能找到资料。

数据集支持
----------

MMEdu系列提供了包括分类、检测等任务的若干数据集，存储在dataset文件夹下。

1）ImageNet
~~~~~~~~~~~

ImageNet是斯坦福大学提出的一个用于视觉对象识别软件研究的大型可视化数据库，目前大部分模型的性能基准测试都在ImageNet上完成。MMEdu的MMClassification支持的数据集类型是ImageNet，如需训练自己创建的数据集，数据集需整理成ImageNet格式。

ImageNet格式数据集文件夹结构如下所示，包含三个文件夹和三个文本文件，图像数据文件夹内，不同类别图片按照文件夹分门别类排好，通过trainning_set、val_set、test_set区分训练集、验证集和测试集。文本文件classes.txt说明类别名称与序号的对应关系，val.txt说明验证集图片路径与类别序号的对应关系，test.txt说明测试集图片路径与类别序号的对应关系。

.. code:: plain

   imagenet
   ├── training_set
   │   ├── class_0
   │   │   ├── filesname_0.JPEG
   │   │   ├── filesname_1.JPEG
   │   │   ├── ...
   │   ├── ...
   │   ├── class_n
   │   │   ├── filesname_0.JPEG
   │   │   ├── filesname_1.JPEG
   │   │   ├── ...
   ├── classes.txt
   ├── val_set
   │   ├── ...
   ├── val.txt
   ├── test_set
   │   ├── ...
   ├── test.txt

如上所示训练数据根据图片的类别，存放至不同子目录下，子目录名称为类别名称。

classes.txt包含数据集类别标签信息，每行包含一个类别名称，按照字母顺序排列。

.. code:: plain

   class_0
   class_1
   ...
   class_n

为了验证和测试，我们建议划分训练集、验证集和测试集，因此另外包含“val.txt”和“test.txt”这两个标签文件，要求是每一行都包含一个文件名和其相应的真实标签。格式如下所示：

.. code:: plain

   filesname_0.jpg 0
   filesname_1.jpg 0
   ...
   filesname_a.jpg n
   filesname_b.jpg n

注：真实标签的值应该位于\ ``[0,类别数目-1]``\ 之间。

如果您觉得整理规范格式数据集有点困难，您只需收集完图片按照类别存放，然后完成训练集（trainning_set）、验证集（val_set）和测试集（test_set）等的拆分，整理在一个大的文件夹下作为你的数据集。此时指定数据集路径后同样可以训练模型，因为XEdu拥有检查数据集的功能，如您的数据集缺失txt文件，会自动帮您生成“classes.txt”，“val.txt”等（如存在对应的数据文件夹）开始训练。这些txt文件会生成在您指定的数据集路径下，即帮您补齐数据集。完整的从零开始制作一个ImageNet格式的数据集的步骤详见\ `深度学习知识库 <https://xedu.readthedocs.io/zh/latest/dl_library/dataset_introduction.html#imagenet>`__\ 。

2）COCO
~~~~~~~

COCO数据集是微软于2014年提出的一个大型的、丰富的检测、分割和字幕数据集，包含33万张图像，针对目标检测和实例分割提供了80个类别的物体的标注，一共标注了150万个物体。MMEdu的MMDetection支持的数据集类型是COCO，如需训练自己创建的数据集，数据集需转换成COCO格式。

MMEdu的MMDetection设计的COCO格式数据集文件夹结构如下所示，“annotations”文件夹存储标注文件，“images”文件夹存储用于训练、验证、测试的图片。

.. code:: plain

   coco
   ├── annotations
   │   ├── train.json
   │   ├── ...
   ├── images
   │   ├── train
   │   │   ├── filesname_0.JPEG
   │   │   ├── filesname_1.JPEG
   │   │   ├── ...
   │   ├── ...

如果您的文件夹结构和上方不同，则需要在“Detection_Edu.py”文件中修改\ ``load_dataset``\ 方法中的数据集和标签加载路径。

COCO数据集的标注信息存储在“annotations”文件夹中的\ ``json``\ 文件中，需满足COCO标注格式，基本数据结构如下所示。

.. code:: plain

   # 全局信息
   {
       "images": [image],
       "annotations": [annotation],
       "categories": [category]
   }

   # 图像信息标注，每个图像一个字典
   image {
       "id": int,  # 图像id编号，可从0开始
       "width": int, # 图像的宽
       "height": int,  # 图像的高
       "file_name": str, # 文件名
   }

   # 检测框标注，图像中所有物体及边界框的标注，每个物体一个字典
   annotation {
       "id": int,  # 注释id编号
       "image_id": int,  # 图像id编号
       "category_id": int,   # 类别id编号
       "segmentation": RLE or [polygon],  # 分割具体数据，用于实例分割
       "area": float,  # 目标检测的区域大小
       "bbox": [x,y,width,height],  # 目标检测框的坐标详细位置信息
       "iscrowd": 0 or 1,  # 目标是否被遮盖，默认为0
   }

   # 类别标注
   categories [{
       "id": int, # 类别id编号
       "name": str, # 类别名称
       "supercategory": str, # 类别所属的大类，如哈巴狗和狐狸犬都属于犬科这个大类
   }]

为了验证和测试，我们建议划分训练集、验证集和测试集，需要生成验证集valid和标注文件valid.json，测试集test和标注文件test.json，json文件的基本数据结构依然是COCO格式。制作一个COCO格式的数据集的步骤详见\ `深度学习知识库 <https://xedu.readthedocs.io/zh/latest/dl_library/dataset_introduction.html#coco>`__\ 。
