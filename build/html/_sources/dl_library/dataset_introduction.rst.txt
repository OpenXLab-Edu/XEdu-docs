经典数据集
==========

ImageNet
--------

ImageNet
是目前世界上图像识别最大的数据库，有超过1500万张图片，约2.2万种类别，权威、可靠。

斯坦福大学教授李飞飞为了解决机器学习中过拟合和泛化的问题而牵头构建的数据集。该数据集从2007年开始手机建立，直到2009年作为论文的形式在CVPR
2009上面发布。直到目前，该数据集仍然是深度学习领域中图像分类、检测、定位的最常用数据集之一。

基于ImageNet有一个比赛，从2010年开始举行，到2017年最后一届结束。该比赛称为ILSVRC，全称是ImageNet
Large-Scale Visual Recognition
Challenge，每年举办一次，每次从ImageNet数据集中抽取部分样本作为比赛的数据集。ILSVRC比赛包括：图像分类、目标定位、目标检测、视频目标检测、场景分类。在该比赛的历年优胜者中，诞生了AlexNet（2012）、VGG（2014）、GoogLeNet（2014）、ResNet（2015）等耳熟能详的深度学习网络模型。“ILSVRC”一词有时候也用来特指该比赛使用的数据集，即ImageNet的一个子集，其中最常用的是2012年的数据集，记为ILSVRC2012。因此有时候提到ImageNet，很可能是指ImageNet中用于ILSVRC2012的这个子集。ILSVRC2012数据集拥有1000个分类（这意味着面向ImageNet图片识别的神经网络的输出是1000个），每个分类约有1000张图片。这些用于训练的图片总数约为120万张，此外还有一些图片作为验证集和测试集。ILSVRC2012含有5万张图片作为验证集，10万张图片作为测试集（测试集没有标签，验证集的标签通过另外的文档给出）。

ImageNet不仅是一个数据集、一项比赛，也是一种典型的数据集格式。分类任务中最经典的数据集类型就是\ `ImageNet格式 <https://xedu.readthedocs.io/zh/latest/mmedu/introduction.html#imagenet>`__\ 。

XEdu中MMEdu的图像分类模块数据集类型是\ `ImageNet <https://xedu.readthedocs.io/zh/latest/mmedu/introduction.html#imagenet>`__\ ，包含三个文件夹和三个文本文件，文件夹内，不同类别图片按照文件夹分门别类排好，通过trainning_set、val_set、test_set区分训练集、验证集和测试集。文本文件classes.txt说明类别名称与序号的对应关系，val.txt说明验证集图片路径与类别序号的对应关系，test.txt说明测试集图片路径与类别序号的对应关系。如需训练自己创建的数据集，数据集需转换成\ `ImageNet格式 <https://xedu.readthedocs.io/zh/latest/mmedu/introduction.html#imagenet>`__\ 。这里，为您提供几种自己制作\ `ImageNet格式 <https://xedu.readthedocs.io/zh/latest/mmedu/introduction.html#coco>`__\ 数据集的方法。

从零开始制作一个ImageNet格式数据集
----------------------------------

(1）巧用BaseDT的make_dataset功能制作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

第一步：整理图片
^^^^^^^^^^^^^^^^

首先新建一个images文件夹用于存放图片，然后开始采集图片，您可以用任何设备拍摄图像，也可以从视频中抽取帧图像，需要注意，这些图像可以被划分为多个类别。每个类别建立一个文件夹，文件夹名称为类别名称，将图片放在其中。

第二步：制作类别说明文件
^^^^^^^^^^^^^^^^^^^^^^^^

在images文件夹同级目录下新建一个文本文件classes.txt，将类别名称写入，要求符合\ `ImageNet格式 <https://xedu.readthedocs.io/zh/latest/mmedu/introduction.html#imagenet>`__\ 。

参考示例如下：

::

   cat
   dog

此时前两步整理的文件夹应是如下格式：

::

   |---images
       |---class1
             |----xxx.jpg/png/....
       |---class2
             |----xxx.jpg/png/....
       |---class3
             |----xxx.jpg/png/....
       |---classN
             |----xxx.jpg/png/....
   classes.txt

第三步：生成数据集
^^^^^^^^^^^^^^^^^^

使用BaseDT库完成数据集制作。如需了解更多BaseDT库数据集处理的功能，详见\ `BaseDT的数据集格式转换 <https://xedu.readthedocs.io/zh/latest/basedt/introduction.html#id7>`__\ 部分。

::

   from BaseDT.dataset import DataSet
   ds = DataSet(r"my_dataset_catdog2") # 指定为生成数据集的路径
   # 默认比例为train_ratio = 0.7, test_ratio = 0.1, val_ratio = 0.2
   ds.make_dataset(r"catdog2", src_format="IMAGENET，",train_ratio = 0.8, test_ratio = 0.1, val_ratio = 0.1)# 指定原始数据集的路径，数据集格式选择IMAGENET

第四步、检查数据集格式
^^^^^^^^^^^^^^^^^^^^^^

最后检查数据集格式转换是否已完成，将文件夹重新命名，在训练的时候，只要通过\ ``model.load_dataset``\ 指定数据集的路径就可以了。

注：网上下载的图像分类数据集也可使用上述方法完成数据集处理。

(2）按照标准方式制作
~~~~~~~~~~~~~~~~~~~~

.. _第一步整理图片-1:

第一步：整理图片
^^^^^^^^^^^^^^^^

您可以用任何设备拍摄图像，也可以从视频中抽取帧图像，需要注意，这些图像可以被划分为多个类别。每个类别建立一个文件夹，文件夹名称为类别名称，将图片放在其中。

接下来需要对图片进行尺寸、保存格式等的统一，简单情况下的参考代码如下：

.. code:: plain

   from PIL import Image
   from torchvision import transforms
   import os

   def makeDir(folder_path):
       if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
           os.makedirs(folder_path)

   classes = os.listdir('./my_dataset/training_set')
   read_dir = './my_dataset/training_set/' # 指定原始图片路径
   new_dir = './my_dataset/newtraining_set'
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

根据整理的数据集大小，按照一定比例拆分训练集、验证集和测试集，可手动也可以使用如下代码将原始数据集按照“6:2:2”的比例拆分。

.. code:: plain

   import os
   import shutil
   # 列出指定目录下的所有文件名，确定分类信息
   classes = os.listdir('./my_photo')

   # 定义创建目录的方法
   def makeDir(folder_path):
       if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
           os.makedirs(folder_path)

   # 指定文件目录
   read_dir = './my_photo/' # 指定原始图片路径
   train_dir = './my_dataset/training_set/' # 指定训练集路径
   test_dir = './my_dataset/test_set/'# 指定测试集路径
   val_dir = './my_dataset/val_set/'# 指定验证集路径

   for cnt in range(len(classes)):
       r_dir = read_dir + classes[cnt] + '/'  # 指定原始数据某个分类的文件目录
       files = os.listdir(r_dir)  # 列出某个分类的文件目录下的所有文件名
       # files = files[:4000]
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
           # shutil.copy(r_dir + fileName,w_dir + classes[cnt] + str(index)+'.jpg')
           shutil.copy(r_dir + fileName, w_dir + str(index) + '.jpg')
       for index,fileName in enumerate(val_data):
           w_dir = val_dir + classes[cnt] + '/'  # 指定测试集某个分类的文件目录
           makeDir(w_dir)
           # shutil.copy(r_dir + fileName, w_dir + classes[cnt] + str(index) + '.jpg')
           shutil.copy(r_dir + fileName, w_dir + str(index) + '.jpg')
       for index,fileName in enumerate(test_data):
           w_dir = test_dir + classes[cnt] + '/'  # 指定验证集某个分类的文件目录
           makeDir(w_dir)
           # shutil.copy(r_dir + fileName, w_dir + classes[cnt] + str(index) + '.jpg')
           shutil.copy(r_dir + fileName, w_dir + str(index) + '.jpg')

第三步：生成标签文件
^^^^^^^^^^^^^^^^^^^^

划分完训练集、验证集和测试集，我们需要生成“classes.txt”，“val.txt”和“test.txt”。其中classes.txt包含数据集类别标签信息，每行包含一个类别名称，按照字母顺序排列。“val.txt”和“test.txt”这两个标签文件的要求是每一行都包含一个文件名和其相应的真实标签。

可以手动完成，这里也为您提供一段用Python代码完成标签文件的程序如下所示，程序中设计了“val.txt”和“test.txt”这两个标签文件每行会包含类别名称、文件名和真实标签。

.. code:: plain

   # 在windows测试通过
   import os
   # 列出指定目录下的所有文件名，确定类别名称
   classes = os.listdir('./my_dataset/training_set')
   # 打开指定文件，并写入类别名称
   with open('./my_dataset/classes.txt','w') as f:
       for line in classes:
           str_line = line +'\n'
           f.write(str_line) # 文件写入str_line，即类别名称

   test_dir = './my_dataset/test_set/' # 指定测试集文件路径
   # 打开指定文件，写入标签信息
   with open('./my_dataset/test.txt','w') as f:
       for cnt in range(len(classes)):
           t_dir = test_dir + classes[cnt]  # 指定测试集某个分类的文件目录
           files = os.listdir(t_dir) # 列出当前类别的文件目录下的所有文件名
           # print(files)
           for line in files:
               str_line = classes[cnt] + '/' + line + ' '+str(cnt) +'\n' 
               f.write(str_line) 

   val_dir = './my_dataset/val_set/'  # 指定文件路径
   # 打开指定文件，写入标签信息
   with open('./my_dataset/val.txt', 'w') as f:
       for cnt in range(len(classes)):
           t_dir = val_dir + classes[cnt]  # 指定验证集某个分类的文件目录
           files = os.listdir(t_dir)  # 列出当前类别的文件目录下的所有文件名
           # print(files)
           for line in files:
               str_line = classes[cnt] + '/' + line + ' ' + str(cnt) + '\n'
               f.write(str_line)  # 文件写入str_line，即标注信息

如果您使用的是Mac系统，可以使用下面的代码。

.. code:: plain

   # 本文件可以放在数据集的根目录下运行
   import os
   # 如果不是在数据集根目录下，可以指定路径
   set_path = './' 

   templist = os.listdir(set_path +'training_set')
   # 处理mac的特殊文件夹
   classes = []
   for line in templist:
       if line[0] !='.':
           classes.append(line)
       
   with open(set_path +'classes.txt','w') as f:
       for line in classes: 
           str_line = line +'\n'
           f.write(str_line) # 文件分行写入，即类别名称

   val_dir = set_path +'val_set/'  # 指定验证集文件路径
   # 打开指定文件，写入标签信息
   with open(set_path +'val.txt', 'w') as f:
       for cnt in range(len(classes)):
           t_dir = val_dir + classes[cnt]  # 指定验证集某个分类的文件目录
           files = os.listdir(t_dir)  # 列出当前类别的文件目录下的所有文件名
           # print(files)
           for line in files:
               str_line = classes[cnt] + '/' + line + ' ' + str(cnt) + '\n'
               f.write(str_line)  # 文件写入str_line，即标注信息

   test_dir = set_path +'test_set/' # 指定测试集文件路径
   # 打开指定文件，写入标签信息
   with open(set_path +'test.txt','w') as f:
       for cnt in range(len(classes)):
           t_dir = test_dir + classes[cnt]  # 指定测试集某个分类的文件目录
           files = os.listdir(t_dir) # 列出当前类别的文件目录下的所有文件名
           # print(files)
           for line in files:
               str_line = classes[cnt] + '/' + line + ' '+str(cnt) +'\n'
               f.write(str_line)

第四步：给数据集命名
^^^^^^^^^^^^^^^^^^^^

最后，我们将这些文件放在一个文件夹中，命名为数据集的名称。这样，在训练的时候，只要通过\ ``model.load_dataset``\ 指定数据集的路径就可以了。

(3）巧用XEdu自动补齐功能快速制作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果您觉得整理规范格式数据集有点困难，其实您只收集了图片按照类别存放，然后完成训练集（trainning_set）、验证集（val_set）和测试集（test_set）等的拆分，最后整理在一个大的文件夹下作为您的数据集也可以符合要求。此时指定数据集路径后同样可以训练模型，因为XEdu拥有检测数据集的功能，如您的数据集缺失txt文件，会自动帮您生成“classes.txt”，“val.txt”等（如存在对应的数据文件夹）开始训练。这些txt文件会生成在您指定的数据集路径下，即帮您补齐数据集。

COCO
----

MS COCO的全称是Microsoft Common Objects in
Context，起源于微软于2014年出资标注的Microsoft
COCO数据集，与ImageNet竞赛一样，被视为是计算机视觉领域最受关注和最权威的比赛之一。

COCO数据集是一个大型的、丰富的物体检测，分割和字幕数据集。这个数据集以scene
understanding为目标，主要从复杂的日常场景中截取，图像中的目标通过精确的segmentation进行位置的标定。图像包括91类目标，328,000影像和2,500,000个label。目前为止有语义分割的最大数据集，提供的类别有80
类，有超过33 万张图片，其中20 万张有标注，整个数据集中个体的数目超过150
万个。

XEdu中MMEdu的MMDetection模块支持的数据集类型是COCO，如需训练自己创建的数据集，数据集需转换成\ `COCO格式 <https://xedu.readthedocs.io/zh/latest/mmedu/introduction.html#coco>`__\ 。这里，为您提供几种自己制作\ `COCO格式 <https://xedu.readthedocs.io/zh/latest/mmedu/introduction.html#coco>`__\ 数据集的方法。

从零开始制作一个COCO格式数据集
------------------------------

(1）OpenInnoLab版
~~~~~~~~~~~~~~~~~

.. _第一步整理图片-2:

第一步、整理图片
^^^^^^^^^^^^^^^^

新建一个images文件夹用于存放图片
，根据需求按照自己喜欢的方式收集图片，图片中包含需要检测的信息即可。

第二步、标注图片
^^^^^^^^^^^^^^^^

使用熟悉的标注方式标注图片，如可进入平台的在线工具-人工智能工坊-数据标注完成数据标注。跳转链接：https://www.openinnolab.org.cn/pjlab/projects/channel

第三步、转换成COCO格式
^^^^^^^^^^^^^^^^^^^^^^

使用BaseDT库将平台标注格式的数据集转换成COCO格式，可以使用如下代码：

.. code:: plain

   from BaseDT.dataset import DataSet
   ds = DataSet(r"my_dataset") # 指定目标数据集
   ds.make_dataset(r"/data/HZQV42", src_format="INNOLAB",train_ratio = 0.8, test_ratio = 0.1, val_ratio = 0.1) # 仅需修改为待转格式的原始数据集路径（注意是整个数据集）

.. _第四步检查数据集格式-1:

第四步、检查数据集格式
^^^^^^^^^^^^^^^^^^^^^^

最后检查数据集格式转换是否已完成，将文件夹重新命名，在训练的时候，只要通过\ ``model.load_dataset``\ 指定数据集的路径就可以了。

参考项目：https://www.openinnolab.org.cn/pjlab/project?id=63c4ad101dd9517dffdff539&sc=635638d69ed68060c638f979#public

(2）LabelMe版
~~~~~~~~~~~~~

.. _第一步整理图片-3:

第一步、整理图片
^^^^^^^^^^^^^^^^

根据需求按照自己喜欢的方式收集图片，图片中包含需要检测的信息即可，可以使用ImageNet格式数据集整理图片的方式对收集的图片进行预处理。

.. _第二步标注图片-1:

第二步、标注图片
^^^^^^^^^^^^^^^^

使用熟悉的标注方式标注图片，如可使用LabelMe批量打开图片文件夹的图片，进行标注并保存为json文件。

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

(3）改装网上下载的目标检测数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

网上也可以找到一些目标检测数据集，但是网上下载的数据集的格式可能不符合XEdu的需求。那么就需要进行数据集格式转换。

我们可以下载网上的数据集，改装生成我们需要的数据集格式。此时可以选择使用BaseDT的常见数据集格式转换功能。

第一步：整理原始数据集
^^^^^^^^^^^^^^^^^^^^^^

首先新建一个annotations文件夹用于存放所有标注文件（VOC格式的为xml文件、COCO格式的为json格式），然后新建一个images文件夹用于存放所有图片，同时在根目录下新建一个classes.txt，写入类别名称。整理规范如下：

::

   原数据集（目标检测）
   |---annotations
         |----xxx.json/xxx.xml/xxx.txt
   |---images
         |----xxx.jpg/png/....
   classes.txt

第二步：转换为COCO格式
^^^^^^^^^^^^^^^^^^^^^^

使用BaseDT库将平台标注格式的数据集转换成COCO格式，可以使用如下代码。如需了解更多BaseDT库数据集处理的功能，详见\ `BaseDT的数据集格式转换 <https://xedu.readthedocs.io/zh/latest/basedt/introduction.html#id7>`__\ 部分。

.. code:: plain

   from BaseDT.dataset import DataSet
   ds = DataSet(r"my_dataset") # 指定为新数据集路径
   ds.make_dataset(r"G:\\测试数据集\\fruit_voc", src_format="VOC",train_ratio = 0.8, test_ratio = 0.1, val_ratio = 0.1) # 指定待转格式的原始数据集路径，原始数据集格式，划分比例，默认比例为train_ratio = 0.7, test_ratio = 0.1, val_ratio = 0.2

第三步、检查数据集格式
^^^^^^^^^^^^^^^^^^^^^^

最后检查数据集格式转换是否已完成，将文件夹重新命名，在训练的时候，只要通过\ ``model.load_dataset``\ 指定数据集的路径就可以了。
