#  GUI库PySimpleGUI

## 1. 简介

`PySimpleGUI` 和 `PySimpleGUIWeb` 都是为 Python 设计的易用的图形用户界面（GUI）库，它们提供了一个简单的方法来创建和显示窗口、按钮、文本和其他 GUI 组件。这两个库的主要区别在于它们的目标平台和底层实现。

### PySimpleGUI

- **底层框架**: 使用的是 Python 的标准 GUI 库 `tkinter` 作为其底层实现。
- **目标平台**: 桌面环境。通过它，你可以轻松地为 Windows、Mac 和 Linux 创建原生应用程序。
- 特点：
  - 简单的 API，允许快速构建应用程序。
  - 提供了大量的小部件，如按钮、文本框、列表框等。
  - 可以轻松地与其他 Python 库集成。

官方GitHub仓库地址：[https://github.com/PySimpleGUI/PySimpleGUI](https://github.com/PySimpleGUI/PySimpleGUI)

### PySimpleGUIWeb

- **底层框架**: 使用 `Remi` 作为其底层实现，它是一个 Python GUI 库，允许你创建的 GUI 在 web 浏览器中运行。
- **目标平台**: Web 浏览器。它旨在为那些想要一个简单的方法来创建 web 应用程序的开发者提供解决方案。
- 特点：
  - 不需要深入了解 web 开发或 HTML/CSS/JavaScript 的知识。
  - 可以在任何支持的浏览器中运行，无需客户端安装。
  - API 与其他 `PySimpleGUI` 版本相似，使得从桌面应用迁移到 web 应用变得简单。

官方GitHub仓库地址：[https://github.com/PySimpleGUI/PySimpleGUI/tree/master/PySimpleGUIWeb](https://github.com/PySimpleGUI/PySimpleGUI/tree/master/PySimpleGUIWeb)

## 2. 安装

均可以采用pip命令安装，具体如下：

```
pip install PySimpleGUIWeb
pip install PySimpleGUI
```

## 3. 如何选择PySimpleGUI和PySimpleGUIWeb

在选择使用哪个版本之前，你应该首先确定你的应用程序的需求。如果你需要一个轻量级的桌面应用程序，`PySimpleGUI` 可能是更好的选择。如果你希望你的应用程序可以在浏览器中运行，那么 `PySimpleGUIWeb` 更适合。

## 4.代码示例

以下是一个基本的 `PySimpleGUIWeb` 示例：

```python
import PySimpleGUIWeb as sg

layout = [[sg.Text('来自 PySimpleGUIWeb 的问候！')],
          [sg.Button('确定')]]

window = sg.Window('示例', layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == '确定':
        break

window.close()
```

上面这段代码一个简单的PySimpleGUIWeb应用程序，它创建了一个包含文本和按钮的窗口。当用户点击"确定"按钮或关闭窗口时，程序将结束。


## 4. 借助PySimpleGUIWeb部署简易AI应用

使用`PySimpleGUIWeb`部署AI应用是一个相对简单且直观的方法，它允许您创建基于Web的图形用户界面(GUI)。只需准备好模型后，使用`PySimpleGUIWeb`创建一个简单的Web应用界面。

### 示例1：带窗体的摄像头实时推理的程序

下面是一段使用PySimpleGUIWeb与OpenCV来显示实时的摄像头图像并对其进行实时推理。在推理过程中，使用的是ONNX模型，推理的代码是借助XEdu团队推出的模型部署工具[BaseDeploy](https://xedu.readthedocs.io/zh/master/basedeploy/introduction.html)，代码较为简洁。关于基于MMEdu训练的模型转换为ONNX的说明可见[最后一步：AI模型转换与部署](https://xedu.readthedocs.io/zh/master/mmedu/model_convert.html#ai)。

```
import remi
import PySimpleGUIWeb as sg
import BaseDeploy as bd
import cv2  #pip install opencv-python
import numpy as np #pip install numpy
from BaseDT.plot import imshow_det_bboxes

model_path = 'det.onnx'
model = bd(model_path)

def my_inf(frame):
    global model,class_name
    res1, img = model.inference(frame,get_img='cv2')
    res2 = model.print_result(res1)
    if len(res2)==0:
        return None,None
    classes=[]
    for res in res2:
        classes.append(res['预测结果'])
    return str(classes),img

#背景色
sg.theme('LightGreen')
#定义窗口布局
layout = [
  [sg.Image(filename='', key='image',size=(600, 400))],
  [sg.Button('关闭', size=(20, 1))],
  [sg.Text('推理结果：',key='res')]
]

#窗口设计
window = sg.Window('OpenCV实时图像处理',layout,size=(600, 500))
#打开内置摄像头
cap = cv2.VideoCapture(1)
while True:
    event, values = window.read(timeout=0, timeout_key='timeout')
    #实时读取图像，重设画面大小
    ret, frame = cap.read()
    imgSrc = cv2.resize(frame, (600,400))
    res, img = my_inf(frame)
    #画面实时更新
    if res:
        print('推理结果为：',res)
        window['res'].update('推理结果：'+res)
        imgbytes = cv2.imencode('.png', img)[1].tobytes()
    else:
        imgbytes = cv2.imencode('.png', imgSrc)[1].tobytes()
    window['image'].update(data=imgbytes)
    if event in (None, '关闭'):
        break
# 退出窗体
cap.release()
window.close()
```

