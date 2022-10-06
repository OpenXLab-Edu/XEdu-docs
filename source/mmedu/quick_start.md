# MMEdu快速入门

## 1.MMEdu是什么？

MMEdu源于国产人工智能视觉（CV）算法集成框架OpenMMLab，是一个“开箱即用”的深度学习开发工具。在继承OpenMMLab强大功能的同时，MMEdu简化了神经网络模型搭建和训练的参数，降低了编程的难度，并实现一键部署编程环境，让初学者通过简洁的代码完成各种SOTA模型（state-of-the-art，指在该项研究任务中目前最好/最先进的模型）的训练，并能够快速搭建出AI应用系统。 

官方地址：[OpenInnoLab](https://www.openinnolab.org.cn/pjEdu/xedu)

GitHub：https://github.com/OpenXLab-Edu/OpenMMLab-Edu 

国内镜像：https://gitee.com/openxlab-edu/OpenMMLab-Edu

## 2.体验MMEdu

MMEdu有多种安装方式，可以通过Pip方式安装，也可以使用一键安装包。体验MMEdu的最快速方式是通过InnoLab平台。

### 2.1访问InnoLab

InnoLab为上海人工智能实验室推出的青少年AI学习平台。在“AI项目工坊 - 人工智能工坊”中，查找”MMEdu“，即可找到所有与MMEdu相关的体验项目。

AI项目工坊：https://www.openinnolab.org.cn/pjLab/projects/channel

![image](../iamges/mmedu/quick_start_01.png)

### 2.2 克隆项目

点击项目即可查看，但是强烈建议你先“克隆”。


### 2.3 加载数据集



### 2.4 训练模型

典型训练：

```python
from MMEdu import MMClassification as cls
model = cls(backbone='LeNet')
model.num_classes = 3
model.load_dataset(path='./dataset')
model.save_fold = './my_model'
model.train(epochs=10, validate=True)
```

继续训练：

```python
from MMEdu import MMClassification as cls
model = cls(backbone='LeNet')
model.num_classes = 3
model.load_dataset(path='./dataset')
model.save_fold = './my_model'
checkpoint = './latest.pth'
model.train(epochs=10, validate=True, checkpoint=checkpoint)
```

### 2.5 模型推理

```python
from MMEdu import MMClassification as cls
img = './img.png'
model = cls(backbone='LeNet')
checkpoint = './latest.pth'
class_path = './classes.txt'
result = model.inference(image=img, show=True, class_path=class_path,checkpoint = checkpoint)
model.print_result(result)
```

## 
