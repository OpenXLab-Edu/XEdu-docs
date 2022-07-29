MMEdu模块应用详解
=================

1.MMEdu是什么？
---------------

MMEdu源于国产人工智能视觉（CV）算法集成框架OpenMMLab，是一个“开箱即用”的深度学习开发工具。在继承OpenMMLab强大功能的同时，MMEdu简化了神经网络模型搭建和训练的参数，降低了编程的难度，并实现一键部署编程环境，让初学者通过简洁的代码完成各种SOTA模型（state-of-the-art，指在该项研究任务中目前最好/最先进的模型）的训练，并能够快速搭建出AI应用系统。

GitHub：https://github.com/OpenXLab-Edu/OpenMMLab-Edu

国内镜像：https://gitee.com/openxlab-edu/OpenMMLab-Edu

2.MMEdu和常见AI框架的比较
-------------------------

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

3.模块概述
----------

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

4.内置模型
----------

================ ==================================== ================
模块名称         内置模型                             功能
================ ==================================== ================
MMClassification LeNet、ResNet18、ResNet50、MobileNet 图片分类
MMDetection      FastRCNN                             图片中的物体检测
================ ==================================== ================

5.数据集支持
------------

MMEdu系列提供了包括分类、检测等任务的若干数据集，存储在dataset文件夹下。

1.ImageNet
~~~~~~~~~~

ImageNet是斯坦福大学提出的一个用于视觉对象识别软件研究的大型可视化数据库，目前大部分模型的性能基准测试都在ImageNet上完成。MMEdu的MMClassification支持的数据集类型是ImageNet，如需训练自己创建的数据集，数据集需转换成ImageNet格式。

ImageNet格式数据集文件夹结构如下所示，图像数据文件夹和标签文件放在同级目录下。

.. code:: plain

   imagenet
   ├── ...
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
   ├── ...

如上所示训练数据根据图片的类别，存放至不同子目录下，子目录名称为类别名称。

classes.txt包含数据集类别标签信息，每行包含一个类别名称，按照字母顺序排列。

.. code:: plain

   class_0
   class_1
   ...
   class_n

为了验证和测试，我们建议划分训练集、验证集和测试集，此时需另外生成“val.txt”和“test.txt”这两个标签文件，要求是每一行都包含一个文件名和其相应的真实标签。格式如下所示：

.. code:: plain

   filesname_0.jpg 0
   filesname_1.jpg 0
   ...
   filesname_a.jpg n
   filesname_b.jpg n

注：真实标签的值应该位于\ ``[0,类别数目-1]``\ 之间。

这里，为您提供一段用Python代码完成标签文件的程序如下所示，程序中设计了“val.txt”和“test.txt”这两个标签文件每行会包含类别名称、文件名和真实标签。

.. code:: plain

   import os
   # 列出指定目录下的所有文件名，确定类别名称
   classes = os.listdir('D:\测试数据集\EX_dataset\\training_set')
   # 打开指定文件，并写入类别名称
   with open('D:\测试数据集\EX_dataset/classes.txt','w') as f:
       for line in classes:
           str_line = line +'\n'
           f.write(str_line) # 文件写入str_line，即类别名称

   test_dir = 'D:\测试数据集\EX_dataset\\test_set/' # 指定测试集文件路径
   # 打开指定文件，写入标签信息
   with open('D:\测试数据集\EX_dataset/test.txt','w') as f:
       for cnt in range(len(classes)):
           t_dir = test_dir + classes[cnt]  # 指定测试集某个分类的文件目录
           files = os.listdir(t_dir) # 列出当前类别的文件目录下的所有文件名
           # print(files)
           for line in files:
               str_line = classes[cnt] + '/' + line + ' '+str(cnt) +'\n' 
               f.write(str_line) 

   val_dir = 'D:\测试数据集\EX_dataset\\val_set/'  # 指定文件路径
   # 打开指定文件，写入标签信息
   with open('D:\测试数据集\EX_dataset/val.txt', 'w') as f:
       for cnt in range(len(classes)):
           t_dir = val_dir + classes[cnt]  # 指定验证集某个分类的文件目录
           files = os.listdir(t_dir)  # 列出当前类别的文件目录下的所有文件名
           # print(files)
           for line in files:
               str_line = classes[cnt] + '/' + line + ' ' + str(cnt) + '\n'
               f.write(str_line)  # 文件写入str_line，即标注信息

至于如何从零开始制作一个ImageNet格式的数据集，可参考如下步骤。

第一步：整理图片
^^^^^^^^^^^^^^^^

您可以用任何设备拍摄图像，也可以从视频中抽取帧图像，需要注意，这些图像可以被划分为多个类别。每个类别建立一个文件夹，文件夹名称为类别名称，将图片放在其中。

接下来需要对图片进行尺寸、保存格式等的统一，可使用如下代码：

.. code:: plain

   from PIL import Image
   from torchvision import transforms
   import os

   def makeDir(folder_path):
       if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
           os.makedirs(folder_path)

   classes = os.listdir('D:\测试数据集\自定义数据集')
   read_dir = 'D:\测试数据集\自定义数据集/' # 指定原始图片路径
   new_dir = 'D:\测试数据集\自定义数据集new/'
   for cnt in range(len(classes)):
       r_dir = read_dir + classes[cnt] + '/'
       files = os.listdir(r_dir)
       for index,file in enumerate(files):
           img_path = r_dir + file
           img = Image.open(img_path)   # 读取图片
           resize = transforms.Resize([224, 224])
           IMG = resize(img)
           w_dir = new_dir + classes[cnt] + '/'
           makeDir(w_dir)
           save_path = w_dir + str(index)+'.jpg'
           IMG = IMG.convert('RGB')
           IMG.save(save_path)

第二步：划分训练集、验证集和测试集
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

根据整理的数据集大小，按照一定比例拆分训练集、验证集和测试集，可使用如下代码将原始数据集按照“6:2:2”的比例拆分。

.. code:: plain

   import os
   import shutil
   # 列出指定目录下的所有文件名，确定类别名称
   classes = os.listdir('D:\测试数据集\自定义表情数据集')

   # 定义创建目录的方法
   def makeDir(folder_path):
       if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
           os.makedirs(folder_path)

   # 指定文件目录
   read_dir = 'D:\测试数据集\自定义表情数据集/' # 指定原始图片路径
   train_dir = 'D:\测试数据集\自制\EX_dataset\\training_set/' # 指定训练集路径
   test_dir = 'D:\测试数据集\自制\EX_dataset\\test_set/' # 指定测试集路径
   val_dir = 'D:\测试数据集\自制\EX_dataset\\val_set/' # 指定验证集路径

   for cnt in range(len(classes)):
       r_dir = read_dir + classes[cnt] + '/'  # 指定原始数据某个分类的文件目录
       files = os.listdir(r_dir)  # 列出某个分类的文件目录下的所有文件名
       files = files[:1000]
       # 按照6:2:2拆分文件名
       offset1 = int(len(files) * 0.6)
       offset2 = int(len(files) * 0.8)
       training_data = files[:offset1]
       val_data = files[offset1:offset2]
       test_data = files[offset2:]

       # 根据拆分好的文件名新建文件目录放入图片
       for index,fileName in enumerate(training_data):
           w_dir = train_dir + classes[cnt] + '/'  # 指定训练集某个分类的文件目录
           makeDir(w_dir)
           shutil.copy(r_dir + fileName,w_dir + classes[cnt] + str(index)+'.jpg')
       for index,fileName in enumerate(test_data):
           w_dir = test_dir + classes[cnt] + '/'  # 指定测试集某个分类的文件目录
           makeDir(w_dir)
           shutil.copy(r_dir + fileName, w_dir + classes[cnt] + str(index) + '.jpg')
       for index,fileName in enumerate(val_data):
           w_dir = val_dir + classes[cnt] + '/'  # 指定验证集某个分类的文件目录
           makeDir(w_dir)
           shutil.copy(r_dir + fileName, w_dir + classes[cnt] + str(index) + '.jpg')

第三步：生成标签文件
^^^^^^^^^^^^^^^^^^^^

划分完训练集、验证集和测试集，我们需要生成“classes.txt”，“val.txt”和“test.txt”，使用上文介绍的Python代码完成标签文件的程序生成标签文件。

第四步：给数据集命名
^^^^^^^^^^^^^^^^^^^^

最后，我们将这些文件放在一个文件夹中，命名为数据集的名称。这样，在训练的时候，只要通过\ ``model.load_dataset``\ 指定数据集的路径就可以了。

2.COCO
~~~~~~

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

​ 这里，为您提供一种自己制作COCO格式数据集的方法。

.. _第一步整理图片-1:

第一步、整理图片
^^^^^^^^^^^^^^^^

根据需求按照自己喜欢的方式收集图片，图片中包含需要检测的信息即可，可以使用ImageNet格式数据集整理图片的方式对收集的图片进行预处理。

第二步、标注图片
^^^^^^^^^^^^^^^^

可使用LabelMe批量打开图片文件夹的图片，进行标注并保存为json文件。

-  LabelMe：格式为LabelMe，提供了转VOC、COCO格式的脚本，可以标注矩形、圆形、线段、点。标注语义分割、实例分割数据集尤其推荐。
-  LabelMe安装与打开方式：\ ``pip install labelme``\ 安装完成后输入\ ``labelme``\ 即可打开。

第三步、转换成COCO标注格式
^^^^^^^^^^^^^^^^^^^^^^^^^^

将LabelMe格式的标注文件转换成COCO标注格式，可以使用如下代码：

.. code:: plain

   import json
   import numpy as np
   import glob
   import PIL.Image
   from PIL import ImageDraw
   from shapely.geometry import Polygon

   class labelme2coco(object):
       def __init__(self, labelme_json=[], save_json_path='./new.json'):
           '''
           :param labelme_json: 所有labelme的json文件路径组成的列表
           :param save_json_path: json保存位置
           '''
           self.labelme_json = labelme_json
           self.save_json_path = save_json_path
           self.annotations = []
           self.images = []
           self.categories = [{'supercategory': None, 'id': 1, 'name': 'cat'},{'supercategory': None, 'id': 2, 'name': 'dog'}] # 指定标注的类别
           self.label = []
           self.annID = 1
           self.height = 0
           self.width = 0
           self.save_json()

       # 定义读取图像标注信息的方法
       def image(self, data, num):
           image = {}
           height = data['imageHeight']
           width = data['imageWidth']
           image['height'] = height
           image['width'] = width
           image['id'] = num + 1
           image['file_name'] = data['imagePath'].split('/')[-1]
           self.height = height
           self.width = width
           return image

       # 定义数据转换方法
       def data_transfer(self):
           for num, json_file in enumerate(self.labelme_json):
               with open(json_file, 'r') as fp:
                   data = json.load(fp)  # 加载json文件
                   self.images.append(self.image(data, num)) # 读取所有图像标注信息并加入images数组
                   for shapes in data['shapes']:
                       label = shapes['label']
                       points = shapes['points']
                       shape_type = shapes['shape_type']
                       if shape_type == 'rectangle':
                           points = [points[0],[points[0][0],points[1][1]],points[1],[points[1][0],points[0][1]]]     
                       self.annotations.append(self.annotation(points, label, num)) # 读取所有检测框标注信息并加入annotations数组
                       self.annID += 1
           print(self.annotations)

       # 定义读取检测框标注信息的方法
       def annotation(self, points, label, num):
           annotation = {}
           annotation['segmentation'] = [list(np.asarray(points).flatten())]
           poly = Polygon(points)
           area_ = round(poly.area, 6)
           annotation['area'] = area_
           annotation['iscrowd'] = 0
           annotation['image_id'] = num + 1
           annotation['bbox'] = list(map(float, self.getbbox(points)))
           annotation['category_id'] = self.getcatid(label)
           annotation['id'] = self.annID
           return annotation

       # 定义读取检测框的类别信息的方法
       def getcatid(self, label):
           for categorie in self.categories:
               if label == categorie['name']:
                   return categorie['id']
           return -1

       def getbbox(self, points):
           polygons = points
           mask = self.polygons_to_mask([self.height, self.width], polygons)
           return self.mask2box(mask)

       def mask2box(self, mask):
           '''从mask反算出其边框
           mask：[h,w]  0、1组成的图片
           1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
           '''
           # np.where(mask==1)
           index = np.argwhere(mask == 1)
           rows = index[:, 0]
           clos = index[:, 1]
           # 解析左上角行列号
           left_top_r = np.min(rows)  # y
           left_top_c = np.min(clos)  # x

           # 解析右下角行列号
           right_bottom_r = np.max(rows)
           right_bottom_c = np.max(clos)

           return [left_top_c, left_top_r, right_bottom_c - left_top_c,
                   right_bottom_r - left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

       def polygons_to_mask(self, img_shape, polygons):
           mask = np.zeros(img_shape, dtype=np.uint8)
           mask = PIL.Image.fromarray(mask)
           xy = list(map(tuple, polygons))
           PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
           mask = np.array(mask, dtype=bool)
           return mask

       def data2coco(self):
           data_coco = {}
           data_coco['images'] = self.images
           data_coco['categories'] = self.categories
           data_coco['annotations'] = self.annotations
           return data_coco

       def save_json(self):
           self.data_transfer()
           self.data_coco = self.data2coco()
           # 保存json文件
           json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4)  # 写入指定路径的json文件，indent=4 更加美观显示

   labelme_json = glob.glob('picture/*.json')  # 获取指定目录下的json格式的文件
   labelme2coco(labelme_json, 'picture/new.json') # 指定生成文件路径

第四步、按照目录结构整理文件
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

创建两个文件夹“images”和“annotations”，分别用于存放图片以及标注信息。按照要求的目录结构，整理好文件夹的文件，最后将文件夹重新命名，在训练的时候，只要通过\ ``model.load_dataset``\ 指定数据集的路径就可以了。

6.一键安装包目录详解
--------------------

MMEdu一键安装版是一个压缩包，解压后即可使用。

MMEdu的根目录结构如下：

.. code:: plain

   OpenMMLab-Edu
   ├── MMEdu
   ├── checkpoints
   ├── dataset
   ├── demo
   ├── HowToStart
   ├── tools（github)
   ├── visualization（github)
   ├── setup.bat
   ├── pyzo.exe
   ├── run_jupyter.bat

接下来对每层子目录进行介绍。

MMEdu目录：
~~~~~~~~~~~

存放各个模块的底层代码、算法模型文件夹“models”和封装环境文件夹“mmedu”。“models”文件夹中提供了各个模块常见的网络模型，内置模型配置文件和说明文档，说明文档提供了模型简介、特点、预训练模型下载链接和适用领域等。“mmedu”文件夹打包了MMEdu各模块运行所需的环境和中小学课程常用的库。

checkpoints目录：
~~~~~~~~~~~~~~~~~

存放各个模块的预训练模型的权重文件，分别放在以模块名称命名的文件夹下，如“cls_model”。

dataset目录：
~~~~~~~~~~~~~

存放为各个模块任务准备的数据集，分别放在以模块名称命名的文件夹下，如“cls”。同时github上此目录下还存放了各个模块自定义数据集的说明文档，如“pose-dataset.md”，文档提供了每个模块对应的数据集格式、下载链接、使用说明、自制数据集流程。

demo目录：
~~~~~~~~~~

存放各个模块的测试程序，如“cls_demo.py”，并提供了测试图片。测试程序包括\ ``py``\ 文件和\ ``ipynb``\ 文件，可支持各种“Python
IDE”和“jupyter
notebook”运行，可运行根目录的“pyzo.exe”和“run_jupyter.bat”后打开测试程序。

HowToStart目录：
~~~~~~~~~~~~~~~~

存放各个模块的使用教程文档，如“MMClassfication使用教程.md”，文档提供了代码详细说明、参数说明与使用等。同时github上此目录下还存放了OpenMMLab各个模块的开发文档供感兴趣的老师和同学参考，如“OpenMMLab_MMClassification.md”，提供了模块介绍、不同函数使用、深度魔改、添加网络等。

tools目录：
~~~~~~~~~~~

存放数据集格式的转换、不同框架的部署等通用工具。后续会陆续开发数据集查看工具、数据集标注工具等工具。

visualization目录：
~~~~~~~~~~~~~~~~~~~

存放可视化界面。
