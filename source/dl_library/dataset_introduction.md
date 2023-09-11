# 经典数据集介绍

## 常见的数据集

## ImageNet

ImageNet 是目前世界上图像识别最大的数据库，有超过1500万张图片，约2.2万种类别，权威、可靠。 

斯坦福大学教授李飞飞为了解决机器学习中过拟合和泛化的问题而牵头构建的数据集。该数据集从2007年开始手机建立，直到2009年作为论文的形式在CVPR 2009上面发布。直到目前，该数据集仍然是深度学习领域中图像分类、检测、定位的最常用数据集之一。

基于ImageNet有一个比赛，从2010年开始举行，到2017年最后一届结束。该比赛称为ILSVRC，全称是ImageNet Large-Scale Visual Recognition Challenge，每年举办一次，每次从ImageNet数据集中抽取部分样本作为比赛的数据集。ILSVRC比赛包括：图像分类、目标定位、目标检测、视频目标检测、场景分类。在该比赛的历年优胜者中，诞生了AlexNet（2012）、VGG（2014）、GoogLeNet（2014）、ResNet（2015）等耳熟能详的深度学习网络模型。“ILSVRC”一词有时候也用来特指该比赛使用的数据集，即ImageNet的一个子集，其中最常用的是2012年的数据集，记为ILSVRC2012。因此有时候提到ImageNet，很可能是指ImageNet中用于ILSVRC2012的这个子集。ILSVRC2012数据集拥有1000个分类（这意味着面向ImageNet图片识别的神经网络的输出是1000个），每个分类约有1000张图片。这些用于训练的图片总数约为120万张，此外还有一些图片作为验证集和测试集。ILSVRC2012含有5万张图片作为验证集，10万张图片作为测试集（测试集没有标签，验证集的标签通过另外的文档给出）。

ImageNet不仅是一个数据集、一项比赛，也是一种典型的数据集格式。分类任务中最经典的数据集类型就是[ImageNet格式](https://xedu.readthedocs.io/zh/master/mmedu/introduction.html#imagenet)。

XEdu中MMEdu的图像分类模块数据集类型是[ImageNet](https://xedu.readthedocs.io/zh/master/mmedu/introduction.html#imagenet)，包含三个文件夹和三个文本文件，文件夹内，不同类别图片按照文件夹分门别类排好，通过training_set、val_set、test_set区分训练集、验证集和测试集。文本文件classes.txt说明类别名称与序号的对应关系，val.txt说明验证集图片路径与类别序号的对应关系，test.txt说明测试集图片路径与类别序号的对应关系。如需训练自己创建的数据集，数据集需转换成[ImageNet格式](https://xedu.readthedocs.io/zh/master/mmedu/introduction.html#imagenet)。如何制作ImageNet格式数据集详见[后文](https://xedu.readthedocs.io/zh/master/dl_library/howtomake_imagenet.html#imagenet)。


## COCO

MS COCO的全称是Microsoft Common Objects in Context，起源于微软于2014年出资标注的Microsoft COCO数据集，与ImageNet竞赛一样，被视为是计算机视觉领域最受关注和最权威的比赛之一。 

COCO数据集是一个大型的、丰富的物体检测，分割和字幕数据集。这个数据集以scene understanding为目标，主要从复杂的日常场景中截取，图像中的目标通过精确的segmentation进行位置的标定。图像包括91类目标，328,000影像和2,500,000个label。目前为止有语义分割的最大数据集，提供的类别有80 类，有超过33 万张图片，其中20 万张有标注，整个数据集中个体的数目超过150 万个。

XEdu中MMEdu的MMDetection模块支持的数据集类型是[COCO](https://xedu.readthedocs.io/zh/master/mmedu/introduction.html#coco)，如需训练自己创建的数据集，数据集需转换成[COCO格式](https://xedu.readthedocs.io/zh/master/mmedu/introduction.html#coco)。如何制作COCO格式数据集详见[后文](https://xedu.readthedocs.io/zh/master/dl_library/howtomake_coco.html#coco)。
