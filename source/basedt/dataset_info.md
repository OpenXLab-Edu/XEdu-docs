# 常见数据集格式介绍

为了降低使用难度，XEdu对支持的数据集格式做了限制，图像分类使用ImageNet，目标检测则使用了coco。

## 1）ImageNet

ImageNet是斯坦福大学提出的一个用于视觉对象识别软件研究的大型可视化数据库，目前大部分模型的性能基准测试都在ImageNet上完成。MMEdu的图像分类模块支持的数据集类型是ImageNet，如需训练自己创建的数据集，数据集需整理成ImageNet格式。

ImageNet格式数据集文件夹结构如下所示，包含三个文件夹(trainning_set、val_set、test_set)和三个文本文件(classes.txt、val.txt、test.txt)，图像数据文件夹内，不同类别图片按照文件夹分门别类排好，通过trainning_set、val_set、test_set区分训练集、验证集和测试集。文本文件classes.txt说明类别名称与序号的对应关系，val.txt说明验证集图片路径与类别序号的对应关系，test.txt说明测试集图片路径与类别序号的对应关系。

```plain
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
```

如上所示训练数据根据图片的类别，存放至不同子目录下，子目录名称为类别名称。

classes.txt包含数据集类别标签信息，每行包含一个类别名称，按照字母顺序排列。

```plain
class_0
class_1
...
class_n
```

为了验证和测试，我们建议划分训练集、验证集和测试集，因此另外包含“val.txt”和“test.txt”这两个标签文件，要求是每一行都包含一个文件名和其相应的真实标签。格式如下所示：

```plain
filesname_0.jpg 0
filesname_1.jpg 0
...
filesname_a.jpg n
filesname_b.jpg n
```
注：真实标签的值应该位于`[0,类别数目-1]`之间。

以“MNIST手写体数字”数据集为例。

```plain
0/0.jpg 0
0/1.jpg 0
...
1/0.jpg 1
1/1.jpg 1
```

如果您觉得整理规范格式数据集有点困难，您只需收集完图片按照类别存放，然后完成训练集（trainning_set）、验证集（val_set）和测试集（test_set）等的拆分，整理在一个大的文件夹下作为你的数据集。此时指定数据集路径后同样可以训练模型，因为XEdu拥有检查数据集的功能，如您的数据集缺失txt文件，会自动帮您生成“classes.txt”，“val.txt”等（如存在对应的数据文件夹）开始训练。这些txt文件会生成在您指定的数据集路径下，即帮您补齐数据集。完整的从零开始制作一个ImageNet格式的数据集的步骤详见[深度学习知识库](https://xedu.readthedocs.io/zh-cn/master/how_to_use/dl_library/howtomake_imagenet.html)。

## 2）COCO

COCO数据集是微软于2014年提出的一个大型的、丰富的检测、分割和字幕数据集，包含33万张图像，针对目标检测和实例分割提供了80个类别的物体的标注，一共标注了150万个物体。MMEdu的MMDetection支持的数据集类型是COCO，如需训练自己创建的数据集，数据集需转换成COCO格式。

MMEdu的目标检测模块设计的COCO格式数据集文件夹结构如下所示，“annotations”文件夹存储标注文件，“images”文件夹存储用于训练、验证、测试的图片。

```plain
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
```

如果您的文件夹结构和上方不同，则需要在“Detection_Edu.py”文件中修改`load_dataset`方法中的数据集和标签加载路径。

COCO数据集的标注信息存储在“annotations”文件夹中的`json`文件中，需满足COCO标注格式，基本数据结构如下所示。

```plain
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
```

为了验证和测试，我们建议划分训练集、验证集和测试集，需要生成验证集valid和标注文件valid.json，测试集test和标注文件test.json，json文件的基本数据结构依然是COCO格式。制作一个COCO格式的数据集的步骤详见[深度学习知识库](https://xedu.readthedocs.io/zh-cn/master/how_to_use/dl_library/howtomake_coco.html)。

## 使用示例

文档涉及的部分代码见XEdu帮助文档配套项目集：[https://www.openinnolab.org.cn/pjlab/project?id=64f54348e71e656a521b0cb5&sc=645caab8a8efa334b3f0eb24#public](https://www.openinnolab.org.cn/pjlab/project?id=64f54348e71e656a521b0cb5&sc=645caab8a8efa334b3f0eb24#public)


