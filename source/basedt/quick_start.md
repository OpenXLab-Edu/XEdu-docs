# 快速上手BaseDT

## 简介

BaseDT不仅是一个功能强大的数据处理库，也是一个易于使用和扩展的库。它可以让你专注于模型的构建和训练，而不用担心数据处理的问题。

## 图像数据处理

BaseDT提供了一个data模块，它包含了多个子模块和类，分别针对不同类型和格式的数据提供了处理功能。

以图片数据处理为例，一条语句，就完成了图片的剪辑、缩放等操作。

范例1：快速处理图片

```Python
from BaseDT.data import ImageData
#图片路径，修改为对应路径
img = r"D:\PythonProject\OpenDataLab-Edu\dataset\cat2.jpg" 
data = ImageData(img, size=(256, 256), crop_size=(224,224), normalize=True)#
data.show()
```

下图分别是原图片，resize大小至(256, 256)的图片，crop大小至(224,224)的图片和归一化的图片。

![](../images/basedt/pic_case.JPEG)


范例2.：根据特定模型处理图像数据


```Python
from BaseDT.data import ImageData
img = r"D:\PythonProject\OpenDataLab-Edu\dataset\cat2.jpg" #修改为对应路径
data = ImageData(img, backbone = "MobileNet")
tensor_value = data.to_tensor()
```

该代码表示使用"MobileNet"网络模型训练时的数据处理，并将数据转化为网络模型可处理的张量。

通过使用BaseDT库，我们可以大大简化网络模型部署时所需地数据处理过程，并且保证与训练时相同或类似地pipeline一致性。

## 数据集转换

用BaseDT库可以快速将不同格式的数据集之间自由转换。

（1）针对网上下载的数据集，BaseDT可支持常见数据集格式（目前支持IMAGENET、VOC和COCO）的转换。

首先需将网上下载的原始数据集做初步整理，整理规范如下：

```
原数据集(分类)
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

原数据集(目标检测)
|---annotations
      |----xxx.json/xxx.xml
|---images
      |----xxx.jpg/png/....
classes.txt
```

注意这里需要有images文件夹和classes.txt这两个部分，才可以进行格式转换。

** 常见图像分类数据集转ImageNet **

如是图像分类数据集，首先需将所有图片按照类别存放，然后将所有图片文件夹放入images文件夹，同时新建一个classes.txt，写入类别信息。通过此过程，也有助于初步了解网上下载的数据集。

整理完毕即可使用BaseDT数据集格式转换的代码完成数据集转换。需指定新数据集路径、原始数据集路径、原始数据集格式、划分比例（如不设置，则默认比例为训练集:测试集:验证集=7:1:2）。

```python
from BaseDT.dataset import DataSet
ds = DataSet(r"mycls") # 指定为生成数据集的路径
# 默认比例为train_ratio = 0.7, test_ratio = 0.1, val_ratio = 0.2
ds.make_dataset(r"mydataset", src_format="IMAGENET",train_ratio = 0.8, test_ratio = 0.1, val_ratio = 0.1)# 指定原始数据集的路径，数据集格式选择IMAGENET
```


** 常见目标检测数据集转coco **

如是目标检测数据集，需将所有图片存放至images文件夹，所有标注文件（VOC格式的为xml文件、COCO格式的为json格式）存放至annotations文件夹，同时在根目录下新建一个classes.txt，写入类别信息（整理规范见前文）。通过此过程，也有助于初步了解网上下载的数据集。

注：大部分yolo数据集使用的是VOC格式

整理完毕即可使用BaseDT数据集格式转换的代码完成数据集转换。需指定新数据集路径、原始数据集路径、原始数据集格式、划分比例（如不设置，则默认比例为训练集:测试集:验证集=7:1:2）。

```python
from BaseDT.dataset import DataSet
ds = DataSet(r"my_dataset") # 指定为新数据集路径
ds.make_dataset(r"G:\\测试数据集\\fruit_voc", src_format="VOC",train_ratio = 0.8, test_ratio = 0.1, val_ratio = 0.1) # 指定待转格式的原始数据集路径，原始数据集格式，划分比例，默认比例为train_ratio = 0.7, test_ratio = 0.1, val_ratio = 0.2
```

![](../images/basedt/voc2coco.png)

转换成功后，不仅将网上下载的数据集转换为了XEdu支持的数据集格式，并对训练集、测试集、验证集进行了重新划分。

（2）针对OpenInnoLab平台标注创建的数据集，也可以使用BaseDT直接转换为XEdu支持的数据集。

**注：需新建一个images文件夹用于存放图片 ，再将images上传平台进行在线标注。**

```Python
from BaseDT.dataset import DataSet
ds = DataSet(r"my_dataset") # 指定目标数据集
ds.make_dataset(r"/data/HZQV42", src_format="INNOLAB",train_ratio = 0.8, test_ratio = 0.1, val_ratio = 0.1) # 仅需修改为待转格式的原始数据集路径（注意是整个数据集）
```

![](../images/basedt/cocotransok.png)

更多XEdu支持的数据集格式的介绍详见[数据集支持](https://xedu.readthedocs.io/zh-cn/latest/mmedu/introduction.html#id3)部分。

**用BaseDT完成数据集制作和格式检查体验项目**：[https://www.openinnolab.org.cn/pjlab/project?id=6412c95f79f259135f259f45&sc=635638d69ed68060c638f979#public](https://www.openinnolab.org.cn/pjlab/project?id=6412c95f79f259135f259f45&sc=635638d69ed68060c638f979#public)

##  数据集拆分

一般训练前建议先完成数据集划分，即将数据集拆分为训练集和验证集，训练集用于训练模型，验证集用于评估模型的性能。针对CSV格式的数据集，可以使用BaseDT完成拆分。

```python
from BaseDT.dataset import split_tab_dataset
# 指定待拆分的csv数据集
path = "feature_data/mmpose_pose_train_feature.csv"
# 指定特征数据列、标签列、训练集比重，‘normalize=True’表示进行归一化处理
tx,ty,val_x,val_y = split_tab_dataset(path,data_column=range(1,35),label_column=35,train_val_ratio=0.8,normalize=True)
```

![](../images/basedt/csvdataset_split.png)
