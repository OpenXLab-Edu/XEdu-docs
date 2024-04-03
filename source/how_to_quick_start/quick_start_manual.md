# 使用说明

欢迎各位小伙伴们来到XEdu的AI天地，要想“畅游”XEdu，入门手册少不了，接下来将会一一介绍XEdu的奇妙AI工具们，带领大家快速了解XEdu各个模块的功能和特点，此外，我们还提供了链接“传送门”，方便小伙伴们轻松玩转AI。

XEdu的使用有两种：网页或者本地

- 网页推荐使用浦育平台，里面有XEdu容器，能够省去环境的配置，并且每周会有一定额度的免费远程算力，上手更加轻松。
- 本地环境的安装推荐选择XEdu一键安装包（CPU版本），它满足大部分机房需求。


## 案例一：用XEduhub执行推理任务（检测任务）

### 简介：

XEduHub针对一些常见任务，提供了现成的优质模型，可以完成目标检测、关键点检测等等，还可以实现自训练模型推理，让初学者能轻松进行AI应用实践。本项目完成了直接调用XEduHub一个内置模型`det_hand`实现检测手的功能，只用7行代码就可实现。

### 链接：

案例详解：[用XEduhub执行推理任务（检测任务）](https://xedu.readthedocs.io/zh/master/how_to_quick_start/how_to_start_hub.html#)
[https://xedu.readthedocs.io/zh/master/how_to_quick_start/how_to_start_hub.html#](https://xedu.readthedocs.io/zh/master/how_to_quick_start/how_to_start_hub.html#)

项目链接：敬请期待……

## 案例二：用BaseML训练机器学习模型（抛物线）

### 简介：

BaseML库提供了众多机器学习训练方法，如线性回归、KNN、SVM等等，可以快速训练和应用模型。本项目使用BaseML中的回归算法，以及其他算法训练投石车落地距离预测模型。投石车落地距离预测是一个典型的抛物线问题，根据投石角度与距离对照表，用机器学习方法预测抛物线函数。

### 链接：

案例详解：[用BaseML训练机器学习模型（抛物线）](https://xedu.readthedocs.io/zh/master/how_to_quick_start/how_to_start_baseml.html#)
[https://xedu.readthedocs.io/zh/master/how_to_quick_start/how_to_start_baseml.html#](https://xedu.readthedocs.io/zh/master/how_to_quick_start/how_to_start_baseml.html#)

项目链接：敬请期待……

## 案例三：用BaseNN训练搭建全连接神经网络（鸢尾花）

### 简介：

BaseNN可以方便地逐层搭建神经网络，深入探究神经网络的原理，训练深度学习模型。本项目核心功能是完成使用经典的鸢尾花数据集完成鸢尾花分类，最后完成了一个简单的鸢尾花分类小应用，输入花萼长度、宽度、花瓣长度、宽度，可以输出预测结果。

### 链接：

案例详解：[用BaseNN训练搭建全连接神经网络（鸢尾花）](https://xedu.readthedocs.io/zh/master/how_to_quick_start/how_to_start_basenn.html#)
[https://xedu.readthedocs.io/zh/master/how_to_quick_start/how_to_start_basenn.html#](https://xedu.readthedocs.io/zh/master/how_to_quick_start/how_to_start_basenn.html#)

项目链接：
[用BaseNN库搭建全连接神经网络训练IRIS鸢尾花分类模型](https://openinnolab.org.cn/pjlab/project?id=641bc2359c0eb14f22fdbbb1&backpath=/pjlab/projects/list#public)
[https://openinnolab.org.cn/pjlab/project?id=641bc2359c0eb14f22fdbbb1&backpath=/pjlab/projects/list#public](https://openinnolab.org.cn/pjlab/project?id=641bc2359c0eb14f22fdbbb1&backpath=/pjlab/projects/list#public)

## 案例四：用MMEdu训练LeNet图像分类模型（手写体）

### 简介：

MMEdu是人工智能视觉算法集成的深度学习开发工具。本项目使用MMEdu的图像分类模块MMClassification，根据经典的手写体ImageNet格式数据集，训练LeNet模型实现手写体识别。此外目前MMClassifiation支持的SOTA模型有LeNet、MobileNet、ResNet18、ResNet50等，支持训练的数据集格式为ImageNet。

### 链接：

案例详解：[用MMEdu训练LeNet图像分类模型（手写体）](https://xedu.readthedocs.io/zh/master/how_to_quick_start/how_to_start_mmcls.html#)
[https://xedu.readthedocs.io/zh/master/how_to_quick_start/how_to_start_mmcls.html#](https://xedu.readthedocs.io/zh/master/how_to_quick_start/how_to_start_mmcls.html#)

项目链接：[用MMEdu实现MNIST手写体数字识别（XEdu官方版）](https://openinnolab.org.cn/pjlab/project?id=64a3c64ed6c5dc7310302853&sc=62f34141bf4f550f3e926e0e#public)
[https://openinnolab.org.cn/pjlab/project?id=64a3c64ed6c5dc7310302853&sc=62f34141bf4f550f3e926e0e#public](https://openinnolab.org.cn/pjlab/project?id=64a3c64ed6c5dc7310302853&sc=62f34141bf4f550f3e926e0e#public)

## 案例五：用MMEdu训练SSD_Lite目标检测模型（猫狗）

### 简介：

MMEdu是人工智能视觉算法集成的深度学习开发工具。本项目使用MMEdu的目标检测模块MMDetection，根据猫狗多目标COCO数据集，训练SSD_Lite模型实现猫狗目标检测。此外此外目前MMClassifiation支持的SOTA模型有LeNet、MobileNet、ResNet18、ResNet50等，支持训练的数据集格式为COCO。

### 链接：

案例详解：[用MMEdu训练SSD_Lite目标检测模型（猫狗）](https://xedu.readthedocs.io/zh/master/how_to_quick_start/how_to_start_mmdet.html)
[https://xedu.readthedocs.io/zh/master/how_to_quick_start/how_to_start_mmdet.html](https://xedu.readthedocs.io/zh/master/how_to_quick_start/how_to_start_mmdet.html)

项目链接：[用MMEdu解决目标检测问题（以猫狗检测为例）](https://openinnolab.org.cn/pjlab/project?id=64055f119c0eb14f22db647c&sc=62f34141bf4f550f3e926e0e#public)
[https://openinnolab.org.cn/pjlab/project?id=64055f119c0eb14f22db647c&sc=62f34141bf4f550f3e926e0e#public](https://openinnolab.org.cn/pjlab/project?id=64055f119c0eb14f22db647c&sc=62f34141bf4f550f3e926e0e#public)

## 案例六：综合项目石头剪刀布的实时识别（XEduhub+BaseNN）

### 简介：

组合XEdu工具集的工具完成一个综合项目非常方便，本项目使用XEduHub提取手势图像的关键点信息，再将这些关键点信息作为特征输入到一个自己搭建的全连接神经网络模型中进行训练，此步骤由BaseNN实现，最后到本地完成模型应用。


### 链接：

案例详解：[综合项目石头剪刀布的实时识别（XEduhub+BaseNN）](https://xedu.readthedocs.io/zh/master/how_to_quick_start/how_to_make_a_small_project.html#)
[https://xedu.readthedocs.io/zh/master/how_to_quick_start/how_to_make_a_small_project.html#](https://xedu.readthedocs.io/zh/master/how_to_quick_start/how_to_make_a_small_project.html#)

项目链接：[用XEduHub和BaseNN完成石头剪刀布手势识别](https://openinnolab.org.cn/pjlab/project?id=66062a39a888634b8a1bf2ca&backpath=/pjedu/userprofile?slideKey=project#public)
[https://openinnolab.org.cn/pjlab/project?id=66062a39a888634b8a1bf2ca&backpath=/pjedu/userprofile?slideKey=project#public](https://openinnolab.org.cn/pjlab/project?id=66062a39a888634b8a1bf2ca&backpath=/pjedu/userprofile?slideKey=project#public)