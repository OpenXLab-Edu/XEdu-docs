# XEduHub功能详解

## 为什么是XEduHub？

XEduHub是一个专为快速、便捷地利用最先进的深度学习模型完成任务而设计的工具库。其设计灵感源自PyTorchHub，旨在以工作流的方式，迅速高效地完成深度学习任务。XEduHub的独特之处在于它内置了大量优质的深度学习SOTA模型，无需用户自行进行繁琐的模型训练。用户只需将这些现成的模型应用于特定任务，便能轻松进行AI应用实践。

## 解锁XEduHub的使用方法

XEduHub作为一个深度学习工具库，集成了许多深度学习领域优质的SOTA模型，能够帮助用户在不进模型训练的前提下，用少量的代码，快速实现计算机视觉、自然语言处理等多个深度学习领域的任务。

## 计算机视觉

### 关键点识别

关键点识别是深度学习中的一项关键任务，旨在检测图像或视频中的关键位置，通常代表物体或人体的重要部位。

关键点识别在众多领域中有着广泛的应用，包括人脸识别、人体姿态估计、虚拟现实等。虽然面临遮挡、姿态变化等挑战，但研究人员不断改进模型和训练策略，以提高准确性和稳定性，推动了计算机视觉和人机交互领域的进步。

关键点识别的成功应用有望进一步丰富我们对图像和视频数据的理解。

#### 0. 引入库

``` python
from XEdu.hub import Workflow as wf
```

运行代码`wf.support_task()`即可查看当前支持的深度学习任务。

#### 1. 模型声明

在第一次声明模型时代码运行用时较长，是因为要将预训练模型从云端下载到本地中，从而便于用户进行使用。

##### 人体关键点

人体关键点识别是一项计算机视觉任务，旨在检测和定位图像或视频中人体的关键位置，通常是关节、身体部位或特定的解剖结构。

这些关键点的检测可以用于人体姿态估计和分类、动作分析、手势识别等多种应用。

XEduHub提供了两个识别人体关键点的模型，`body17`和`body26` 数字表示了识别出人体关键点的数量，声明代码如下：

```python
body = wf(task='body') # 数字可省略，当省略时，默认为body17
```

##### 人脸关键点

人脸关键点识别是计算机视觉领域中的一项任务，它的目标是检测和定位人脸图像中的关键点，通常是代表面部特征的重要点，

例如眼睛、鼻子、嘴巴、眉毛等。这些关键点的准确定位对于许多应用非常重要，包括人脸识别、表情分析、虚拟化妆、人机交互等。

XEduHub提供了识别人脸关键点的模型：`face106`，这意味着该模型能够识别人脸上的106个关键点，声明代码如下：

```python
face = wf(task='face') # 数字可省略，默认为face106
```

##### 人手关键点

人手关键点识别是一项计算机视觉任务，其目标是检测和定位图像或视频中人手的关键位置，通常包括手指、手掌、手腕等关键部位的位置。

这些关键点的识别对于手势识别、手部姿态估计、手部追踪、手势控制设备等应用具有重要意义。

XEduHub提供了识别人手关键点的模型：`hand21`，该模型能够识别人手上的21个关键点，声明代码如下：

```python
hand = wf(task='hand') # 数字可省略，默认为hand21
```

##### 人体所有关键点

XEduHub提供了识别人体所有关键点，包括人手、人脸和人体躯干部分关键点的模型：`wholebody133`，声明代码如下：

```python
wholebody = wf(task='wholebody') # 数字可省略，默认为wholebody133
```

#### 2. 模型推理

由于已经从云端下载好了预训练的SOTA模型，因此只需要传入相应图片即可进行模型推理任务，识别相应的关键点，以人体关键点识别为例，模型推理代码如下：

```python
img = "data/body.jpg" # 指定待识别关键点的图片的路径
keypoints,img_with_keypoints = body.inference(data=img,img_type='pil') # 进行模型推理
```

`keypoints`保存了所有关键点的坐标，`img`以pil格式保存了关键点识别完成后的图片

`inference()`可传入参数：

- `data`: 指定待识别关键点的图片

- `img_type`: 关键点识别完成后会返回含有关键点的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`
- `bbox`：该参数可配合目标检测使用。在多人关键点检测中，该参数指定了要识别哪个检测框中的关键点

#### 3. 结果输出

XEduHub提供了一种便捷的方式，能够以标准美观的格式查看关键点坐标以及分数（可以理解为置信度），代码如下：

```python
format_result = body.format_output(lang='zh')# 参数language设置了输出结果的语言
```

显示带有关键点和关键点连线的结果图像

```python
body.show(img_with_keypoints)
```

####    4. 结果保存

XEduHub提供了保存带有关键点和关键点连线结果图像的方法，代码如下：

```python
body.save(img_with_keypoints,'img_with_keypoints.jpg')
```

### 目标检测

目标检测是一种计算机视觉任务，其目标是在图像或视频中检测并定位物体的位置，并为每个物体分配类别标签。

实现目标检测通常包括特征提取、物体位置定位、物体类别分类等步骤。这一技术广泛应用于自动驾驶、安全监控、人脸识别、医学影像分析、虚拟现实、图像搜索等各种领域，为实现自动化和智能化应用提供了关键支持。

#### 0. 引入库

``` python
from XEdu.hub import Workflow as wf
```

运行代码`wf.support_task()`即可查看当前支持的深度学习任务。

#### 1. 模型声明

在第一次声明模型时代码运行用时较长，是因为要将预训练模型从云端下载到本地中，从而便于用户进行使用。

##### 人体目标检测

人体目标检测是目标检测的一个领域，它的任务是在图像或视频中检测和定位人体的位置，并为每个检测到的人体分配一个相应的类别标签。

通常，人体目标检测不仅要求准确地定位人体的边界框，还需要对人体进行类别分类，如行人、车辆乘客、骑自行车者等。这一技术在自动驾驶、视频监控、行人识别、人数统计、人体姿态估计等领域中有广泛的应用。

XEduHub提供了进行人体目标检测的模型：`bodydetect`，该模型既能够进行单人的人体目标检测，也能够实现多人检测，声明代码如下：

```python
det = wf(task='bodydetect')
```

##### coco目标检测

COCO（Common Objects in Context）是一个用于目标检测和图像分割任务的广泛使用的数据集和评估基准。它是计算机视觉领域中最重要的数据集之一，在XEduHub中的该模型能够检测出80类coco数据集中的物体：`cocodetect`，声明代码如下

```python
det = wf(task='cocodetect')
```

若要查看coco目标检测中的所有类别可运行以下代码：

```python
wf.coco_class()
```

#### 2. 模型推理

由于已经从云端下载好了预训练的SOTA模型，因此只需要传入相应图片即可进行模型推理任务，识别相应的关键点，以人体目标检测为例，模型推理代码如下：

```python
img = 'data/body.jpg'
result,img_with_box = det.inference(data=img,img_type='cv2')
```

`result`保存了检测框四个顶点的坐标，`img_with_box`以cv2格式保存了包含了检测框的图片

`det.inference()`可传入参数：

- `data`：指定待检测的图片
- `img_type`：目标检测完成后会返回含有检测框的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`
- `thr`: 设置检测框阈值，超过该阈值的检测框被视为有效检测框，进行显示

####   3. 结果输出

XEduHub提供了一种便捷的方式，能够以标准美观的格式查看检测框顶点坐标、检测分数以及目标类别，代码如下：

```python
format_result =det.format_output(lang='zh')# 参数language设置了输出结果的语言
```

 显示带有检测框的图片

```python
det.show(img_with_box)
```

####    4. 结果保存

XEduHub提供了带有检测框图片的方法，代码如下：

```python
det.save(img_with_box,'img_with_box.jpg')
```

### 目标检测+关键点识别综合应用

以下代码可以实时检测摄像头中出现的多个人，并对每一个人体提取关键点

其中，我们先进行目标检测，拿到所有的检测框`bbox`及其顶点坐标

随后对每个检测框中的人体进行关键点提取

```python
from XEdu.hub import Workflow as wf
import cv2
cap = cv2.VideoCapture(0)
body = wf(task='body17')# 实例化pose模型
det = wf(task='bodydetect')#实例化detect模型
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    bboxs = det.inference(data=frame,thr=0.3)
    img = frame
    for i in bboxs:
        keypoints,img =body.inference(data=img,img_type='cv2',bbox=i)
    for [x1,y1,x2,y2] in bboxs: # 画检测框
        cv2.rectangle(img, (int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
    cv2.imshow('video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    
cap.release()
cv2.destroyAllWindows()
```

