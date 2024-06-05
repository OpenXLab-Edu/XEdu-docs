

# XEduHub功能详解

XEduHub作为一个深度学习工具库，集成了许多深度学习领域优质的SOTA模型，能够帮助用户在不进模型训练的前提下，用少量的代码，快速实现计算机视觉、自然语言处理等多个深度学习领域的任务。

XEduHub支持两类任务，分为内置任务和通用任务两种。顾名思义，内置任务指预置的各种优质开源AI模型，包含了图像分类、目标检测、关键点检测、OCR等。通用任务指用XEdu系列工具训练的模型，如MMEdu（计算机视觉）、BaseNN（自定义神经网络）和BaseML（传统机器学习），还支持用其他工具训练的模型。考虑到兼容性，大部分模型以ONNX为主。不管使用哪类任务，XEduHub都使用Workflow工具进行推理，核心代码仅4行，语法非常简洁。


## 通用模型推理工具Workflow

我们在使用XEduHub时都需要执行这段代码`from XEdu.hub import Workflow as wf`。Workflow的基本逻辑是使用训练好的模型对数据进行推理。

那什么是Workflow呢？在使用XEduHub里的单个模型时，Workflow就是模型推理的工作流，从数据，到输入模型，再到输出推理结果。在使用XEduHub里多个模型进行联动时，Workflow可以看做不同模型之间的数据流动，例如首先进行多人的目标检测，将检测到的数据传入关键点识别模型从而对每个人体进行关键点识别。

下面开始介绍Workflow中丰富的深度学习工具。

### 强烈安利项目<a href="https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&backpath=/pjlab/projects/list#public">XEduHub实例代码-入门完整版</a>

<a href="https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&backpath=/pjlab/projects/list#public">https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&backpath=/pjlab/projects/list#public</a>

通过学习“XEduHub实例代码-入门完整版”，可以在项目实践中探索XEduHub的魅力，项目中通俗易懂的讲解和实例代码也能帮助初学者快速入门XEduHub。

## 1. 目标检测

目标检测是一种计算机视觉任务，其目标是在图像或视频中检测并定位物体的位置，并为每个物体分配类别标签。

实现目标检测通常包括特征提取、物体位置定位、物体类别分类等步骤。这一技术广泛应用于自动驾驶、安全监控、医学影像分析、图像搜索等各种领域，为实现自动化和智能化应用提供了关键支持。

XEduHub目标支持目标检测任务有：人体目标检测`det_body`、人脸目标检测`det_face`、人手目标检测`det_hand`和coco目标检测`det_coco`。

### 人体目标检测

人体目标检测的任务是在图像或视频中检测和定位人体的位置，并为每个检测到的人体分配一个相应的类别标签。借助此任务，一般可以实现人体画面的识别与提取，为后续的图像处理和分析提供基础。

XEduHub提供了进行人体目标检测的模型：`det_body`，`det_body_l`，这两个模型能够进行人体目标检测，后者为large增强版模型。

![](../images/xeduhub/det_body.png)

#### 代码样例

```python
from XEdu.hub import Workflow as wf # 导入库
det_body = wf(task='det_body') # 实例化模型
result,img_with_box = det_body.inference(data='data/det_body.jpg',img_type='pil') # 进行模型推理
format_result = det_body.format_output(lang='zh') # 将推理结果进行格式化输出
det_body.show(img_with_box)# 展示推理图片
det_body.save(img_with_box,'img_with_box.jpg') # 保存推理图片
```

#### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
det_body = wf(task='det_body')
```

`wf()`中共有三个参数可以设置：

- `task`(str)：选择任务。人体目标检测模型为`det_body`, `det_body_l`。`det_body_l`相比`det_body`模型规模较大，性能较强，但是推理的速度较慢。
- `checkpoint`(str)：指定模型文件所在的路径，如`det_body = wf(task='det_body',checkpoint='det_body.onnx') `。如果没有指定模型路径，Workflow会默认在本地同级的“checkpoints”文件夹中寻找与任务名对应的模型文件，即`det_body.onnx`。否则将通过网络到浦源平台的专用地址下载。
- `download_path`(str)：指定模型的下载路径。缺省情况下，模型文件会下载到“checkpoints”文件夹中，“checkpoints”文件夹不存在则自动建立。如果希望代码在没有网络的设备上也能运行，请同步复制`checkpoints`文件夹。如希望模型保存放在其他路径，则设置`download_path`参数，如`download_path='my_checkpoint'`。注意，`download_path`参数为文件夹名称。建议该参数留空，使用默认地址。

任务模型文件获取与存放请查看[下文](https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id132)。

#### 2. 模型推理
推理方式1：
```python
result = det_body.inference(data='data/det_body.jpg') # 进行模型推理
```
推理方式2：
```python
result,img_with_box = det_body.inference(data='data/det_body.jpg',img_type='pil') # 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`(str|numpy.ndarray)：指定待目标检测的图片。
- `show`(flag)：可取值`[True,False]` ,如果取值为`True`，在推理完成后会直接输出目标检测完成后的图片，默认为`False`。
- `img_type`(str)：目标检测完成后会返回含有目标检测框的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，即如果不传入值，则不会返回图。
- `thr`(float)：设置检测框阈值，取值范围为`[0,1]`超过该阈值的检测框被视为有效检测框，进行显示，默认值为0.3。

模型推理返回结果：

- `result`(numpy.ndarray)：以二维数组的形式保存了检测框左上角顶点的坐标(x1,y1)和右下角顶点的坐标(x2,y2)（之所以是二维数组，是因为该模型能够检测多个人体，因此当检测到多个人体时，就会有多个[x1,y1,x2,y2]的一维数组，所以需要以二维数组形式保存），我们可以利用这四个数据计算出其他两个顶点的坐标，以及检测框的宽度和高度。

- `img_with_box`(numpy.ndarray)：是个三维数组，以img_type所设置的格式保存了包含了检测框的图片。

#### 3. 结果输出

```python
format_result = det_body.format_output(lang='zh')# 将推理结果进行格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(str)：设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool)：设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_result`以字典形式存储了推理结果，共有两个键：`检测框`、`分数`。检测框以二维数组形式保存了每个检测框的坐标信息[x1,y1,x2,y2]，而分数则是对应下标的检测框的置信度，以一维数组形式保存。

```
# 输出结果
{'检测框': [[323.2777170453753,
          72.95395088195801,
          917.5945354189192,
          1130.7357228142876]],
 '分数': [0.8851305]}
```

```python
det_body.show(img_with_box)# 展示推理图片
```

`show()`能够输出带有检测框的结果图像。

![](../images/xeduhub/det_body_show.png)

#### 4. 结果保存

```python
det_body.save(img_with_box,'img_with_box.jpg')# 保存推理图片
```

`save()`方法能够保存带有检测框的图像

该方法接收两个参数，第一个参数是图像数据，另一个是图像的保存路径。

### 人脸目标检测

人脸目标检测指的是检测和定位一张图片中的人脸。XEduHub使用的是opencv的人脸检测模型，能够快速准确地检测出一张图片中所有的人脸。XEduHub提供了进行人脸目标检测的模型：`det_face`，能够快速准确地检测出图片中的所有人脸。

#### 代码样例

```python
from XEdu.hub import Workflow as wf
det_face = wf(task='det_face')
result,img_with_box = det_face.inference(data='data/det_face.jpg',img_type='pil') # 进行模型推理
format_result = det_face.format_output(lang='zh') # 将推理结果进行格式化输出
det_face.show(img_with_box) # 展示推理图片
det_face.save(img_with_box,'img_with_box.jpg') # 保存推理图片
```

#### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
det_face = wf(task='det_face')
```
`wf()`中共有三个参数可以设置：

- `task`(str)：选择任务。人脸目标检测模型为`det_face`。
- `checkpoint`(str)：指定模型文件所在的路径，如`det_face = wf(task='det_face',checkpoint='det_face.onnx') `。如果没有指定模型路径，Workflow会默认在本地同级的“checkpoints”文件夹中寻找与任务名对应的模型文件，即`det_face.onnx`。否则将通过网络到浦源平台的专用地址下载。
- `download_path`(str)：指定模型的下载路径。缺省情况下，模型文件会下载到“checkpoints”文件夹中，“checkpoints”文件夹不存在则自动建立。如果希望代码在没有网络的设备上也能运行，请同步复制`checkpoints`文件夹。如希望模型保存放在其他路径，则设置`download_path`参数，如`download_path='my_checkpoint'`。注意，`download_path`参数为文件夹名称。建议该参数留空，使用默认地址。

任务模型文件获取与存放请查看[下文](https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id132)。

#### 2. 模型推理
推理方式1：
```python
result = det_face.inference(data='data/det_face.jpg') # 进行模型推理
```

推理方式2：

```python
result,img_with_box = det_face.inference(data='data/det_face.jpg',img_type='pil') # 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`(str|numpy.ndarray)：指定待目标检测的图片。
- `show`(flag)：可取值`[True,False]` ,如果取值为`True`，在推理完成后会直接输出目标检测完成后的图片，默认为`False`。
- `img_type`(str)：目标检测完成后会返回含有目标检测框的图片，该参数指定了返回图片的格式，可选有：`['cv2','pil']`，默认值为`None`，即如果不传入值，则不会返回图。
- `thr`(float)：设置检测框阈值，取值范围为`[0,1]`超过该阈值的检测框被视为有效检测框，进行显示，默认值为0.5。
- `minSize`(tuple(int,int))：检测框的最小尺寸，小于该尺寸的目标会被过滤掉，默认为(50,50)。
- `maxSize`(tuple(int,int))：检测框的最大尺寸,大于该尺寸的目标会被过滤掉，默认为输入图像的大小。
- `scaleFactor`(float)：该参数用于缩放图像，以便在检测过程中使用不同大小的窗口来识别人脸。较小的值会导致检测速度加快，但可能会错过一些小的人脸；较大的值可以提高检测的准确性，但会减慢检测速度。通常，这个值会在1.1到1.5之间进行调整，默认为1.1。
- `minNeighbors`(int)：该参数定义了构成检测目标的最小邻域矩形个数。如果这个值设置得太高，可能会导致检测器过于严格，错过一些实际的人脸；如果设置得太低，检测器可能会变得过于宽松，错误地检测到非人脸区域。通常，这个值会在2到10之间进行调整，默认为5。

模型推理返回结果：

- `result`(numpy.ndarray)：：以二维数组的形式保存了检测框左上角顶点的坐标(x1,y1)和右下角顶点的坐标(x2,y2)（之所以是二维数组，是因为该模型能够检测多个人脸，因此当检测到多个人脸时，就会有多个[x1,y1,x2,y2]的一维数组，所以需要以二维数组形式保存），我们可以利用这四个数据计算出其他两个顶点的坐标，以及检测框的宽度和高度。
- `img_with_box`(numpy.ndarray)：是个三维数组，以img_type所设置的格式保存了包含了检测框的图片。

#### 3. 结果输出

```python
format_result = det_face.format_output(lang='zh')# 将推理结果进行格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(str)：设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool)：设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_result`以字典形式存储了推理结果，只有一个键：`检测框`。检测框以二维数组形式保存了每个检测框的坐标信息[x1,y1,x2,y2]。需要注意的是由于使用的为opencv的人脸检测模型，因此在`format_output`时缺少了分数这一指标。

```
# 输出结果
{'检测框': [[202, 237, 940, 975]]}
```

![](../images/xeduhub/det_face_format.png)

结果可视化：

```python
det_face.show(img_with_box) # 展示推理图片
```

`show()`能够输出带有检测框的结果图像。

![](../images/xeduhub/det_face_show.png)

多人脸检测结果如下：（如果看到很多不该识别为人脸的地方出现了检测框，我们可以在推理是增加`minSize`参数来设置检测框的最小尺寸，将小于该尺寸的目标过滤掉，默认为(50,50)，同时也可以增加`maxSize`参数来设置检测框的最大尺寸，大于该尺寸的目标会被过滤掉，默认为输入图像的大小）

![](../images/xeduhub/det_face_2.png)

#### 4. 结果保存

```python
det_face.save(img_with_box,'img_with_box.jpg')# 保存推理图片
```

`save()`方法能够保存带有检测框的图像

该方法接收两个参数，第一个参数是图像数据，另一个是图像的保存路径。

### 手部目标检测

手部目标检测指的是检测和定位一张图片中的人手。XEduHub提供了进行手部目标检测的模型：`det_hand`，能够快速准确地检测出图片中的所有人手。

#### 代码样例

```python
from XEdu.hub import Workflow as wf
det_hand = wf(task='det_hand')
result,img_with_box = det_hand.inference(data='data/det_hand.jpg',img_type='pil') # 进行模型推理
format_result = det_hand.format_output(lang='zh') # 将推理结果进行格式化输出
det_hand.show(img_with_box) # 展示推理图片
det_hand.save(img_with_box,'img_with_box.jpg') # 保存推理图片
```

#### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
det_hand = wf(task='det_hand')
```

`wf()`中共有三个参数可以设置：

- `task`(str)：选择任务。手部关键点提取模型为`det_hand`。
- `checkpoint`(str)：指定模型文件所在的路径，如`det_hand = wf(task='det_hand',checkpoint='det_hand.onnx') `。如果没有指定模型路径，Workflow会默认在本地同级的“checkpoints”文件夹中寻找与任务名对应的模型文件，即`det_hand.onnx`。否则将通过网络到浦源平台的专用地址下载。
- `download_path`(str)：指定模型的下载路径。缺省情况下，模型文件会下载到“checkpoints”文件夹中，“checkpoints”文件夹不存在则自动建立。如果希望代码在没有网络的设备上也能运行，请同步复制`checkpoints`文件夹。如希望模型保存放在其他路径，则设置`download_path`参数，如`download_path='my_checkpoint'`。注意，`download_path`参数为文件夹名称。建议该参数留空，使用默认地址。

任务模型文件获取与存放请查看[下文](https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id132)。

#### 2. 模型推理

推理方式1：
```python
result = det_hand.inference(data='data/det_hand.jpg') # 进行模型推理
```
推理方式2：
```python
result,img_with_box = det_hand.inference(data='data/det_hand.jpg',img_type='pil') # 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`(str|numpy.ndarray)：指定待检测的图片。
- `show`(flag): 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出目标检测完成后的图片。
- `img_type`(str)：目标检测完成后会返回含有检测框的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。
- `thr`(float): 设置检测框阈值，取值范围为`[0,1]`超过该阈值的检测框被视为有效检测框，进行显示，默认值为0.3。

模型推理返回结果：

- `result`(numpy.ndarray)：以二维数组的形式保存了检测框左上角顶点的坐标(x1,y1)和右下角顶点的坐标(x2,y2)（之所以是二维数组，是因为该模型能够检测多个人手，因此当检测到多个人手时，就会有多个[x1,y1,x2,y2]的一维数组，所以需要以二维数组形式保存），我们可以利用这四个数据计算出其他两个顶点的坐标，以及检测框的宽度和高度。
- `img_with_box`(numpy.ndarray)：是个三维数组，以img_type所设置的格式保存了包含了检测框的图片。

#### 3. 结果输出

```python
format_result = det_hand.format_output(lang='zh') # 将推理结果进行格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(str)：设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool)：设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

```
# 输出结果
{'检测框': [[354.9846431187221,
          171.7757524762835,
          993.5257284981864,
          867.7952793666294]],
 '分数': [0.8558253]}
```

![](../images/xeduhub/det_hand_format2.png)

`format_result`以字典形式存储了推理结果，共有两个键：`检测框`、`分数`。检测框以二维数组形式保存了每个检测框的坐标信息[x1,y1,x2,y2]，而分数则是对应下标的检测框的置信度，以一维数组形式保存。

结果可视化：

```python
det_hand.show(img_with_box) # 展示推理图片
```

`show()`能够输出带有检测框的结果图像。

![](../images/xeduhub/det_hand_show.png)

#### 4. 结果保存

```python
det_hand.save(img_with_box,'img_with_box.jpg') # 保存推理图片
```

`save()`方法能够保存带有检测框的图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

### coco目标检测

COCO（Common Objects in Context）是一个用于目标检测和图像分割任务的广泛使用的数据集和评估基准。它是计算机视觉领域中最重要的数据集之一，在XEduHub中的该模型能够检测出80类coco数据集中的物体：`det_coco`，以及加强版`det_coco_l`。

![](../images/xeduhub/new_coco.png)

#### 代码样例

```python
from XEdu.hub import Workflow as wf
det_coco = wf(task='det_coco')
result,img_with_box = det_coco.inference(data='data/det_coco.jpg',img_type='pil') # 进行模型推理
format_result = det_coco.format_output(lang='zh') # 将推理结果进行格式化输出
det_coco.show(img_with_box) # 展示推理图片
det_coco.save(img_with_box,'img_with_box.jpg')# 保存推理图片
```

#### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
det_coco = wf(task='det_coco')
```

`wf()`中共有三个参数可以设置：

- `task`(str)：选择任务。coco目标检测的模型为`det_coco`, `det_coco_l`。`det_coco_l`相比`det_coco`模型规模较大，性能较强，但是推理的速度较慢。
- `checkpoint`(str)：指定模型文件所在的路径，如`det_coco = wf(task='det_coco',checkpoint='det_coco.onnx') `。如果没有指定模型路径，Workflow会默认在本地同级的“checkpoints”文件夹中寻找与任务名对应的模型文件，即`det_coco.onnx`。否则将通过网络到浦源平台的专用地址下载。
- `download_path`(str)：指定模型的下载路径。缺省情况下，模型文件会下载到“checkpoints”文件夹中，“checkpoints”文件夹不存在则自动建立。如果希望代码在没有网络的设备上也能运行，请同步复制`checkpoints`文件夹。如希望模型保存放在其他路径，则设置`download_path`参数，如`download_path='my_checkpoint'`。注意，`download_path`参数为文件夹名称。建议该参数留空，使用默认地址。

任务模型文件获取与存放请查看[下文](https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id132)。


#### 2. 模型推理

推理方式1：
```python
result = det_coco.inference(data='data/det_coco.jpg')
```
推理方式2：
```python
result,img_with_box = det_coco.inference(data='data/det_coco.jpg',img_type='pil')
```

模型推理`inference()`可传入参数：

- `data`(str|numpy.ndarray)：指定待目标检测的图片。
- `show`(flag)：可取值`[True,False]` ,如果取值为`True`，在推理完成后会直接输出目标检测完成后的图片，默认为`False`。
- `img_type`(str)：目标检测完成后会返回含有目标检测框的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，即如果不传入值，则不会返回图。
- `thr`(float)：设置检测框阈值，取值范围为`[0,1]`超过该阈值的检测框被视为有效检测框，进行显示。
- `target_class`(str|list)：该参数在使用`det_coco`的时候可以指定要检测的对象，如：`target_class`='person'，`target_class`=['person','cake']。

若要查看coco目标检测中的所有类别可运行以下代码：

```python
wf.coco_class()
```

```
# 输出结果
{1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorbike', 5: 'aeroplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench', 15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe', 25: 'backpack', 26: 'umbrella', 27: 'handbag', 28: 'tie', 29: 'suitcase', 30: 'frisbee', 31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite', 35: 'baseball bat', 36: 'baseball glove', 37: 'skateboard', 38: 'surfboard', 39: 'tennis racket', 40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork', 44: 'knife', 45: 'spoon', 46: 'bowl', 47: 'banana', 48: 'apple', 49: 'sandwich', 50: 'orange', 51: 'broccoli', 52: 'carrot', 53: 'hot dog', 54: 'pizza', 55: 'donut', 56: 'cake', 57: 'chair', 58: 'sofa', 59: 'pottedplant', 60: 'bed', 61: 'diningtable', 62: 'toilet', 63: 'tvmonitor', 64: 'laptop', 65: 'mouse', 66: 'remote', 67: 'keyboard', 68: 'cell phone', 69: 'microwave', 70: 'oven', 71: 'toaster', 72: 'sink', 73: 'refrigerator', 74: 'book', 75: 'clock', 76: 'vase', 77: 'scissors', 78: 'teddy bear', 79: 'hair drier', 80: 'toothbrush'}
```

模型推理返回结果：

- `result`(numpy.ndarray)：以二维数组的形式保存了检测框左上角顶点的坐标(x1,y1)和右下角顶点的坐标(x2,y2)（之所以是二维数组，是因为该模型能够检测多个对象，因此当检测到多个对象时，就会有多个[x1,y1,x2,y2]的一维数组，所以需要以二维数组形式保存），我们可以利用这四个数据计算出其他两个顶点的坐标，以及检测框的宽度和高度。

  ```
  # reslut
  array([[  4.14969308, 164.72434085, 157.05228533, 260.79559871],
         [548.76094273, 322.64147513, 585.47729492, 400.10044466],
         [291.30113874, 216.19574056, 378.10302734, 325.63501167],
         [362.52070836, 215.27021027, 442.51865932, 318.6803018 ],
         [547.75268555, 297.44887938, 587.52868652, 399.59864902],
         [214.87919399, 230.39230279, 302.72096906, 321.90567071],
         [548.62234933, 330.30918053, 585.72174072, 399.95723615],
         [198.86025565, 178.02907828, 261.54931205, 215.89466899]])
  ```

![](../images/xeduhub/det_res.png)

- `img_with_box`(numpy.ndarray)：是个三维数组，以img_type所设置的格式保存了包含了检测框的图片。

#### 3. 结果输出

```python
format_result = det_coco.format_output(lang='zh') # 将推理结果进行格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(str)：设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool)：设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

```
# 输出结果
{'检测框': [[4.149693080357143,
          164.72434084756034,
          157.05228533063615,
          260.79559871128623],
         [548.7609427315848,
          322.64147513253346,
          585.477294921875,
          400.100444657462],
         [291.3011387416295,
          216.19574056352886,
          378.10302734375,
          325.63501167297363],
         [362.5207083565848,
          215.27021026611328,
          442.51865931919644,
          318.68030180249895],
         [547.752685546875,
          297.44887937818254,
          587.5286865234375,
          399.5986490249634],
         [214.8791939871652,
          230.39230278560092,
          302.7209690638951,
          321.90567071097234],
         [548.6223493303571,
          330.30918053218295,
          585.7217407226562,
          399.95723615373885],
         [198.8602556501116,
          178.02907827922274,
          261.549312046596,
          215.8946689878191]],
 '分数': [0.74398744,
        0.49514472,
        0.4024352,
        0.3905959,
        0.3684283,
        0.34017423,
        0.33570302,
        0.3242398],
 '类别': ['tvmonitor',
        'vase',
        'chair',
        'chair',
        'bottle',
        'chair',
        'bottle',
        'pottedplant']}
```

`format_result`以字典形式存储了推理结果，共有三个键：`检测框`、`分数`和`类别`。检测框以二维数组形式保存了每个检测框的坐标信息[x1,y1,x2,y2]，而分数则是对应下标的检测框的置信度，以一维数组形式保存，类别则是检测框中对象所属的类别，以一维数组形式保存。

```python
det_coco.show(img_with_box) # 展示推理图片
```

`show()`能够输出带有检测框以及对应类别的结果图像。

![](../images/xeduhub/det_coco_show.png)

#### 4. 结果保存

```python
det_coco.save(img_with_box,'img_with_box.jpg') # 保存推理图片
```

`save()`方法能够保存带有检测框以及对应类别的图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

## 2. 关键点识别

关键点识别是深度学习中的一项关键任务，旨在检测图像或视频中的关键位置，通常代表物体或人体的重要部位。XEduHub支持的关键点识别任务有：人体关键点`pose_body`、人脸关键点`pose_face`、人手关键点`pose_hand`和所有人体关键点识别`pose_wholebody`。

**注意事项**：这里我们强烈建议提取关键点之前应**先进行目标检测**。

例如进行人体关键点检测`pose_body`之前，先使用`det_body`在图片中检测中人体目标，提取人体画面，再对每个人体目标进行更加精准的关键点检测。可参考项目<a href="https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&backpath=/pjlab/projects/list#public">XEduHub实例代码-入门完整版</a>中的 **“3-1 综合项目：目标检测+关键点检测”**。

当然关键点识别也可以单独用，但是效果并不保证。

![](../images/xeduhub/pose.png)

### 人体关键点识别 

人体关键点识别是一项计算机视觉任务，旨在检测和定位图像或视频中人体的关键位置，通常是关节、身体部位或特定的解剖结构。

这些关键点的检测可以用于人体姿态估计和分类、动作分析、手势识别等多种应用。

XEduHub提供了三个识别人体关键点的优质模型:`pose_body17`,`pose_body17_l`和`pose_body26`，能够在使用cpu推理的情况下，快速识别出身体的关键点。

 模型的数字表示识别出人体关键点的数量，l代表了large，表示规模较大的，性能较强的模型，但是缺点在于推理速度较慢。

`pose_body17`与`pose_body17_l`模型能识别出17个人体骨骼关键点，`pose_body26`模型能识别出26个人体骨骼关键点。

![](../images/xeduhub/body.png)

#### 代码样例

```python
from XEdu.hub import Workflow as wf
body = wf(task='pose_body') # 数字可省略，当省略时，默认为pose_body17
keypoints,img_with_keypoints = body.inference(data='data/body.jpg',img_type='pil') # 进行模型推理
format_result = body.format_output(lang='zh') # 将推理结果进行格式化输出
body.show(img_with_keypoints) # 展示推理图片
body.save(img_with_keypoints,'img_with_keypoints.jpg')  # 保存推理图片
```

#### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
body = wf(task='pose_body') # 数字可省略，当省略时，默认为pose_body17
```

`wf()`中共有三个参数可以设置：

- `task`(str)：选择任务。在人体关键点识别模型中，`task`可选取值为：`[pose_body17,pose_body17_l,pose_body26]`，默认为`pose_body17`。
- `checkpoint`(str)：指定模型文件所在的路径，如`pose_body = wf(task='pose_body',checkpoint='pose_body.onnx') `。如果没有指定模型路径，Workflow会默认在本地同级的“checkpoints”文件夹中寻找与任务名对应的模型文件，即`pose_body17.onnx`。否则将通过网络到浦源平台的专用地址下载。
- `download_path`(str)：指定模型的下载路径。缺省情况下，模型文件会下载到“checkpoints”文件夹中，“checkpoints”文件夹不存在则自动建立。如果希望代码在没有网络的设备上也能运行，请同步复制`checkpoints`文件夹。如希望模型保存放在其他路径，则设置`download_path`参数，如`download_path='my_checkpoint'`。注意，`download_path`参数为文件夹名称。建议该参数留空，使用默认地址。

任务模型文件获取与存放请查看[下文](https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id132)。

#### 2. 模型推理
推理方式1：
```python
keypoints = body.inference(data='data/body.jpg') # 进行模型推理
```

推理方式2：

```python
keypoints,img_with_keypoints = body.inference(data='data/body.jpg',img_type='pil') # 进行模型推理
```


模型推理`inference()`可传入参数：

- `data`(str)：指定待识别关键点的图片，可以是以图片路径形式传入，也可直接传入cv2或pil格式的图片。

- `show`(flag)：可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出关键点识别完成后的图片。

- `img_type`(str)：关键点识别完成后会返回含有关键点的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。
- `bbox`(numpy.ndarray)：该参数可配合目标检测结果使用。例如在多人体关键点检测任务中，我们先识别人体位置，再检测人体关键点，该参数指定了要识别哪个目标检测框中的关键点。

模型推理返回结果：

- `keypoints`(numpy.ndarray)：以二维数组的形式保存了所有关键点的坐标，每个关键点(x,y)被表示为`[x,y]`根据前面的图示，要获取到某个特定序号`i`的关键点，只需要访问`keypoints[i]`即可。

- `img_with_keypoints`(numpy.ndarray)：是个三维数组，以对应img_type格式保存了关键点识别完成后图片的像素点信息。

#### 3. 结果输出

```python
format_result = body.format_output(lang='zh') # 参数lang设置了输出结果的语言，默认为中文
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(str)：设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool)：设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_result`以字典形式存储了推理结果，共有两个键：`关键点坐标`和`分数`。关键点坐标以二维数组形式保存了每个关键点的[x,y]坐标，而分数则是对应下标的关键点的分数，以一维数组形式保存。

```
# 输出结果
{'关键点坐标': [[596.3159722222222, 163.53819444444457],
           [624.6579861111111, 140.86458333333326],
           [576.4765625, 149.3671875],
           [658.6684027777778, 166.37239583333326],
           [553.8029513888889, 177.7092013888889],
           [735.1918402777778, 288.24305555555543],
           [511.28993055555566, 322.2534722222222],
           [871.2335069444446, 387.44010416666674],
           [406.4244791666665, 407.2795138888889],
           [789.0416666666667, 347.7612847222222],
           [369.5798611111111, 381.7717013888889],
           [735.1918402777778, 642.5182291666667],
           [590.6475694444443, 656.6892361111111],
           [831.5546875, 832.4097222222224],
           [516.9583333333335, 855.0833333333333],
           [825.8862847222222, 1013.7986111111111],
           [488.61631944444434, 1039.306423611111]],
 '分数': [0.9026135802268982,
        0.9218268990516663,
        0.9566612243652344,
        0.9752001762390137,
        0.9569406509399414,
        0.9008727669715881,
        0.8631673455238342,
        0.9160149693489075,
        0.7716920375823975,
        0.7274723649024963,
        0.7813198566436768,
        0.8071638941764832,
        0.8179947733879089,
        0.8660239577293396,
        0.8697924613952637,
        0.9618276953697205,
        0.9472517371177673]}
```

![](../images/xeduhub/body-format.png)

结果可视化：

```python
body.show(img_with_keypoints)
```

`show()`能够输出带有关键点和关键点连线的结果图像。

![](../images/xeduhub/body_show.png)

**若此时发现关键点识别效果不佳**，关键点乱飞，我们可以果断采用在提取关键点之前**先进行目标检测**的方式。如当前任务`'pose_body'`，就可以在之前先进行`'det_body'`。详情可参考项目<a href="https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&backpath=/pjlab/projects/list#public">XEduHub实例代码-入门完整版</a>中的 **“3-1 综合项目：目标检测+关键点检测”**。

#### 4. 结果保存

```python
body.save(img_with_keypoints,'img_with_keypoints.jpg')
```

`save()`方法能够保存保存带有关键点和关键点连线结果图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

### **人脸关键点**

人脸关键点识别是计算机视觉领域中的一项任务，它的目标是检测和定位人脸图像中代表面部特征的重要点，例如眼睛、鼻子、嘴巴、眉毛等。这些关键点的准确定位对于许多应用非常重要，包括人脸识别、表情分析、虚拟化妆、人机交互等。

XEduHub提供了识别人脸关键点的模型：`pose_face106`，这意味着该模型能够识别人脸上的106个关键点。如下图所示是106个关键点在脸部的分布情况，我们可以利用这些关键点的分布特征进行人脸识别，或者对人的表情进行分析和分类等。

![](../images/xeduhub/new_face106.png)

#### 代码样例

```python
from XEdu.hub import Workflow as wf
face = wf(task='pose_face') # 数字可省略，当省略时，默认为pose_face106
keypoints,img_with_keypoints = face.inference(data='data/face.jpg',img_type='pil') # 进行模型推理
format_result = face.format_output(lang='zh') # 将推理结果进行格式化输出
face.show(img_with_keypoints) # 展示推理图片
face.save(img_with_keypoints,'img_with_keypoints.jpg') # 保存推理图片
```

#### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
face = wf(task='pose_face') # 数字可省略，默认为face106
```

`wf()`中共有三个参数可以设置：

- `task`(str)：选择任务。人脸关键点识别模型为`pose_face106`（数字可省略，默认为pose_face）。
- `checkpoint`(str)：指定模型文件所在的路径，如`pose_face = wf(task='pose_face',checkpoint='pose_face.onnx') `。如果没有指定模型路径，Workflow会默认在本地同级的“checkpoints”文件夹中寻找与任务名对应的模型文件，即`pose_face106.onnx`。否则将通过网络到浦源平台的专用地址下载。
- `download_path`(str)：指定模型的下载路径。缺省情况下，模型文件会下载到“checkpoints”文件夹中，“checkpoints”文件夹不存在则自动建立。如果希望代码在没有网络的设备上也能运行，请同步复制`checkpoints`文件夹。如希望模型保存放在其他路径，则设置`download_path`参数，如`download_path='my_checkpoint'`。注意，`download_path`参数为文件夹名称。建议该参数留空，使用默认地址。

任务模型文件获取与存放请查看[下文](https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id132)。
#### 2. 模型推理
推理方式1：
```python
keypoints = face.inference(data='data/face.jpg') # 进行模型推理
```
推理方式2：
```python
keypoints,img_with_keypoints = face.inference(data='data/face.jpg',img_type='pil') # 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`(string|numpy.ndarray): 指定待识别关键点的图片，可以是以图片路径形式传入，也可直接传入cv2或pil格式的图片。

- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出关键点识别完成后的图片。

- `img_type`: 关键点识别完成后会返回含有关键点的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。

- `bbox`：该参数可配合目标检测使用。在多目标关键点检测中，该参数指定了要识别哪个检测框中的关键点。

模型推理返回结果：

- `keypoints`以二维数组的形式保存了所有关键点的坐标，每个关键点(x,y)被表示为`[x,y]`根据前面的图示，要获取到某个特定序号`i`的关键点，只需要访问`keypoints[i]`即可。

- `img_with_keypoints`是个三维数组，以对应img_type格式保存了关键点识别完成后图片的像素点信息。

#### 3. 结果输出

```python
format_result = face.format_output(lang='zh') # 将推理结果进行格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(str) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_result`以字典形式存储了推理结果，共有两个键：`关键点坐标`和`分数`。关键点坐标以二维数组形式保存了每个关键点的[x,y]坐标（face106模型共检测出106个关键点），而分数则是对应下标的关键点的分数，以一维数组形式保存。

结果可视化

```python
face.show(img_with_keypoints) # 展示推理图片
```

`show()`能够输出带有关键点的结果图像。

![](../images/xeduhub/face_show.png)

**若此时发现关键点识别效果不佳**，关键点乱飞，我们可以果断采用在提取关键点之前**先进行目标检测**的方式。如当前任务`'pose_face'`，就可以在之前先进行`'det_face'`。详情可参考项目<a href="https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&backpath=/pjlab/projects/list#public">XEduHub实例代码-入门完整版</a>中的 **“3-1 综合项目：目标检测+关键点检测”**。

#### 4. 结果保存

```python
face.save(img_with_keypoints,'img_with_keypoints.jpg') # 保存推理图片
```

`save()`方法能够保存带有关键点的图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

### **人手关键点**

人手关键点识别是一项计算机视觉任务，其目标是检测和定位图像或视频中人手的关键位置，通常包括手指、手掌、手腕等关键部位的位置。这些关键点的识别对于手势识别、手部姿态估计、手部追踪、手势控制设备等应用具有重要意义。

XEduHub提供了能够快速识别人手关键点的模型：`pose_hand21`，该模型能够识别人手上的21个关键点，如下图所示。你可以根据自身需要对关键点进行进一步处理。例如：手势的不同会体现在关键点位置的分布上，这样就可以利用这些关键点进行手势的分类和识别。

![](../images/xeduhub/new_hand.png)

#### 代码样例

```python
from XEdu.hub import Workflow as wf
hand = wf(task='pose_hand') # 数字可省略，当省略时，默认为pose_hand21
keypoints,img_with_keypoints = hand.inference(data='data/hand.jpg',img_type='pil') # 进行模型推理
format_result = hand.format_output(lang='zh') # 将推理结果进行格式化输出
hand.show(img_with_keypoints) # 展示推理图片
hand.save(img_with_keypoints,'img_with_keypoints.jpg') # 保存推理图片
```

#### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
hand = wf(task='pose_hand') # 数字可省略，当省略时，默认为pose_hand21
```

`wf()`中共有三个参数可以设置：

- `task`选择任务。人手关键点识别模型为`pose_hand`。
- `checkpoint`指定模型文件所在的路径，如`pose_hand = wf(task='pose_hand',checkpoint='pose_hand.onnx') `。如果没有指定模型路径，Workflow会默认在本地同级的“checkpoints”文件夹中寻找与任务名对应的模型文件，即`pose_hand.onnx`。否则将通过网络到浦源平台的专用地址下载。
- `download_path`指定模型的下载路径。缺省情况下，模型文件会下载到“checkpoints”文件夹中，“checkpoints”文件夹不存在则自动建立。如果希望代码在没有网络的设备上也能运行，请同步复制`checkpoints`文件夹。如希望模型保存放在其他路径，则设置`download_path`参数，如`download_path='my_checkpoint'`。注意，`download_path`参数为文件夹名称。建议该参数留空，使用默认地址。

任务模型文件获取与存放请查看[下文](https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id132)。

#### 2. 模型推理

推理方式1：
```python
keypoints = hand.inference(data='data/hand.jpg') # 进行模型推理
```
推理方式2：
```python
keypoints,img_with_keypoints = hand.inference(data='data/hand.jpg',img_type='pil') # 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`: 指定待识别关键点的图片，可以是以图片路径形式传入，也可直接传入cv2或pil格式的图片。

- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出关键点识别完成后的图片。

- `img_type`: 关键点识别完成后会返回含有关键点的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。
- `bbox`：该参数可配合目标检测使用。在多人手关键点检测中，该参数指定了要识别哪个检测框中的关键点。

模型推理返回结果：

- `keypoints`以二维数组的形式保存了所有关键点的坐标，每个关键点(x,y)被表示为`[x,y]`根据前面的图示，要获取到某个特定序号`i`的关键点，只需要访问`keypoints[i]`即可。

- `img_with_keypoints`是个三维数组，以img_type设置的保存了关键点识别完成后的图片。

#### 3. 结果输出

```python
format_result = hand.format_output(lang='zh')# 将推理结果进行格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(str) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。
`format_result`以字典形式存储了推理结果，共有两个键：`关键点坐标`和`分数`。关键点坐标以二维数组形式保存了每个关键点的[x,y]坐标，而分数则是对应下标的关键点的分数，以一维数组形式保存。

```python
hand.show(img_with_keypoints) # 展示推理图片
```

`show()`能够输出带有关键点的结果图像。

![](../images/xeduhub/hand_show.png)

**若此时发现关键点识别效果不佳**，关键点乱飞，我们可以果断采用在提取关键点之前**先进行目标检测**的方式。如当前任务`'pose_hand'`，就可以在之前先进行`'det_hand'`。详情可参考项目<a href="https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&backpath=/pjlab/projects/list#public">XEduHub实例代码-入门完整版</a>中的 **“3-1 综合项目：目标检测+关键点检测”**。

#### 4. 结果保存

```python
hand.save(img_with_keypoints,'img_with_keypoints.jpg') # 保存推理图片
```

`save()`方法能够保存带有关键点的图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

### **人体所有关键点**

XEduHub提供了识别人体所有关键点，包括人手、人脸和人体躯干部分关键点的模型：`pose_wholebody133`。具体关键点的序号及其分布如下图所示：

![](../images/xeduhub/wholebody.png)

#### 代码样例

```python
from XEdu.hub import Workflow as wf
wholebody = wf(task='pose_wholebody') # 数字可省略，当省略时，默认为pose_wholebody133
keypoints,img_with_keypoints = wholebody.inference(data='data/wholebody.jpg',img_type='pil') # 进行模型推理
format_result = wholebody.format_output(lang='zh') # 将推理结果进行格式化输出
wholebody.show(img_with_keypoints) # 展示推理图片
wholebody.save(img_with_keypoints,'img_with_keypoints.jpg') # 保存推理图片
```

#### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
wholebody = wf(task='pose_wholebody') # 数字可省略，当省略时，默认为pose_wholebody133
```

`wf()`中共有三个参数可以设置：

- `task`选择任务。全身关键点提取模型为`pose_wholebody`。
- `checkpoint`指定模型文件所在的路径，如`wholebody = wf(task='pose_wholebody',checkpoint='pose_wholebody.onnx') `。如果没有指定模型路径，Workflow会默认在本地同级的“checkpoints”文件夹中寻找与任务名对应的模型文件，即`pose_wholebody.onnx`。否则将通过网络到浦源平台的专用地址下载。
- `download_path`指定模型的下载路径。缺省情况下，模型文件会下载到“checkpoints”文件夹中，“checkpoints”文件夹不存在则自动建立。如果希望代码在没有网络的设备上也能运行，请同步复制`checkpoints`文件夹。如希望模型保存放在其他路径，则设置`download_path`参数，如`download_path='my_checkpoint'`。注意，`download_path`参数为文件夹名称。建议该参数留空，使用默认地址。

任务模型文件获取与存放请查看[下文](https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id132)。

#### 2. 模型推理

```python
# keypoints = wholebody.inference(data='data/wholebody.jpg') # 模型推理方式1
keypoints,img_with_keypoints = wholebody.inference(data='data/wholebody.jpg',img_type='pil') # 模型推理方式2
```

模型推理`inference()`可传入参数：

- `data`: 指定待识别关键点的图片，可以是以图片路径形式传入，也可直接传入cv2或pil格式的图片。

- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出关键点识别完成后的图片。

- `img_type`: 关键点识别完成后会返回含有关键点的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。
- `bbox`：该参数可配合目标检测使用。在多人手关键点检测中，该参数指定了要识别哪个检测框中的关键点。

模型推理返回结果：

- `keypoints`以二维数组的形式保存了所有关键点的坐标，每个关键点(x,y)被表示为`[x,y]`根据前面的图示，要获取到某个特定序号`i`的关键点，只需要访问`keypoints[i]`即可。

- `img_with_keypoints`是个三维数组，以img_type设置的格式保存了关键点识别完成后的图片。

#### 3. 结果输出

```python
format_result = wholebody.format_output(lang='zh') # 将推理结果进行格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(str) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_result`以字典形式存储了推理结果，共有两个键：`关键点坐标`和`分数`。关键点坐标以二维数组形式保存了每个关键点的[x,y]坐标，而分数则是对应下标的关键点的分数，以一维数组形式保存。

```python
wholebody.show(img_with_keypoints) # 展示推理图片
```

`show()`能够输出带有关键点的结果图像。

![](../images/xeduhub/wholebody_show.png)

**若此时发现关键点识别效果不佳**，关键点乱飞，请在提取关键点之前**先进行目标检测**。如当前任务`'pose_wholebody'`，就可以在之前先进行`'det_body'`。详情可参考项目<a href="https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&backpath=/pjlab/projects/list#public">XEduHub实例代码-入门完整版</a>中的 **“3-1 综合项目：目标检测+关键点检测”**。

#### 4. 结果保存

```python
wholebody.save(img_with_keypoints,'img_with_keypoints.jpg') # 保存推理图片
```

`save()`方法能够保存带有关键点的图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

## 3. 光学字符识别（OCR）

光学字符识别（Optical Character Recognition, OCR）是一项用于将图像或扫描的文档转换为可编辑的文本格式的技术。OCR技术能够自动识别和提取图像或扫描文档中的文本，并将其转化为计算机可处理的文本格式。OCR技术在车牌识别、证件识别、文档扫描、拍照搜题等多个场景有着广泛应用。

XEduHub使用的OCR模型是来自百度的开源免费的OCR模型：rapidocr，这个模型运行速度快，性能优越，小巧灵活，并且能支持超过6000种字符的识别，如简体中文、繁体中文、英文、数字和其他艺术字等等。

注意：你可以在当前项目中找到名为**font**的文件夹，里面的FZVTK.TTF文件是一种字体文件，为了显示识别出的文字而使用。

### 代码样例

```python
from XEdu.hub import Workflow as wf
ocr = wf(task='ocr')
result,ocr_img = ocr.inference(data='data/ocr_img.png',img_type='cv2') # 进行模型推理
ocr_format_result = ocr.format_output(lang="zh") # 推理结果格式化输出
ocr.show(ocr_img) # 展示推理结果图片
ocr.save(ocr_img,'ocr_result.jpg') # 保存推理结果图片
```

### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
ocr = wf(task='ocr')
```

`wf()`中只有一个参数可以设置：

- `task`选择任务类型，光学字符识别（OCR）的模型为`ocr`。

**注意**：ocr任务并没有使用ONNX模型，也不会以onnx方式下载模型文件，而是自动下载和安装一个Python库，因此不同于之前任务的下载方式，无需指定下载路径。如果在断网情况第一次使用，可以先通过`pip install rapidocr_onnxruntime==1.3.7`预先下载库。

#### 2. 模型推理

```python
# result = ocr.inference(data='data/ocr_img.png') # 模型推理方式1
result,ocr_img = ocr.inference(data='data/ocr_img.png',img_type='cv2') # 模型推理方式2
```

模型推理`inference()`可传入参数：

- `data`: 指定待进行OCR识别的图片，可以是以图片路径形式传入，也可直接传入cv2或pil格式的图片。
- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出OCR识别完成后的图片。
- `img_type`：OCR识别完成后会返回含有识别结果的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`。

`result`以一维数组的形式保存了识别出的文本及其检测框的四个顶点(x,y)坐标。

```
# result
[('8-01', [[37.0, 82.0], [226.0, 94.0], [221.0, 177.0], [32.0, 165.0]]),
 ('消火栓', [[177.0, 282.0], [1109.0, 288.0], [1106.0, 610.0], [174.0, 603.0]]),
 ('FIREHYDRANT',
  [[278.0, 580.0], [1010.0, 580.0], [1010.0, 642.0], [278.0, 642.0]]),
 ('种花元学',
  [[1111.0, 604.0], [1160.0, 604.0], [1160.0, 618.0], [1111.0, 618.0]]),
 ('消连道一酒防安全连查点',
  [[1080.0, 635.0], [1155.0, 635.0], [1155.0, 652.0], [1080.0, 652.0]]),
 ('地址：文料大楼6楼8-',
  [[918.0, 647.0], [1000.0, 647.0], [1000.0, 664.0], [918.0, 664.0]]),
 ('码：3T-00001018',
  [[1081.0, 652.0], [1147.0, 652.0], [1147.0, 666.0], [1081.0, 666.0]]),
 ('区中生校区',
  [[1081.0, 670.0], [1126.0, 670.0], [1126.0, 683.0], [1081.0, 683.0]]),
 ('查4-1',
  [[1097.0, 663.0], [1130.0, 663.0], [1130.0, 674.0], [1097.0, 674.0]]),
 ('工大楼（中）上公#',
  [[1076.0, 676.0], [1253.0, 678.0], [1253.0, 696.0], [1076.0, 693.0]]),
 ('消火栓使用说明', [[469.0, 761.0], [901.0, 764.0], [901.0, 821.0], [469.0, 818.0]]),
 ('How to Use a Fire Hydrant',
  [[463.0, 822.0], [902.0, 826.0], [902.0, 857.0], [463.0, 852.0]]),
 ('1.按下消火栓箱门右侧的按钮，并打开箱门。',
  [[166.0, 1039.0], [525.0, 1037.0], [525.0, 1060.0], [166.0, 1062.0]]),
 ('Press thebuttonontheright sideof thefirecabinet,thenopen thedoor.',
  [[178.0, 1061.0], [763.0, 1059.0], [763.0, 1082.0], [178.0, 1084.0]]),
 ('2.拉出水带，并将水带两端接口分别对准消火栓固定接口和水枪接口，顺时针方向旋转、拧紧。',
  [[165.0, 1102.0], [924.0, 1102.0], [924.0, 1128.0], [165.0, 1128.0]]),
 ('Takeoutthehose,andconnect the interfacesof thehosewiththehydrantand thenozzlebyrotatingclockwise.',
  [[184.0, 1123.0], [1094.0, 1125.0], [1094.0, 1151.0], [184.0, 1149.0]]),
 ('3.按下消火栓箱内的启泵按钮，指示灯闪亮说明消防泵已启动。',
  [[165.0, 1169.0], [669.0, 1168.0], [669.0, 1191.0], [165.0, 1192.0]]),
 ('4.由一人紧握水枪，另一人逆时针旋开消火栓阀门至最大开度。',
  [[167.0, 1234.0], [668.0, 1234.0], [668.0, 1257.0], [167.0, 1257.0]]),
 ('Onepersonholdsthenozzle tightly,whiletheother turnsonthevalveanticlockwise tomaximumopening',
  [[187.0, 1253.0], [1039.0, 1258.0], [1039.0, 1284.0], [187.0, 1279.0]]),
 ('火警电话',
  [[295.0, 1327.0], [458.0, 1332.0], [457.0, 1375.0], [294.0, 1370.0]]),
 ('119', [[466.0, 1330.0], [569.0, 1330.0], [569.0, 1396.0], [466.0, 1396.0]]),
 ('内部应急电话',
  [[629.0, 1331.0], [869.0, 1334.0], [869.0, 1375.0], [629.0, 1373.0]]),
 ('Fire Telephone',
  [[292.0, 1369.0], [460.0, 1373.0], [460.0, 1400.0], [292.0, 1396.0]]),
 ('Emergency24/7Hotline(021)62238110',
  [[624.0, 1372.0], [1072.0, 1374.0], [1072.0, 1401.0], [624.0, 1399.0]]),
 ('消防设施 请保持完好',
  [[409.0, 1456.0], [851.0, 1459.0], [851.0, 1503.0], [409.0, 1500.0]])]
```

如呈现的输出结果所示，数组中每个元素的形式为元组：（识别文本，检测框顶点坐标）。四个顶点坐标顺序分别为[左上，右上，左下，右下]。

`ocr_img`的格式为cv2，保存了ocr识别后的结果图片。

#### 3. 结果输出

```python
ocr_format_result = ocr.format_output(lang="zh")
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(str) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

```
# 输出结果
{'检测框': [[[37.0, 82.0], [226.0, 94.0], [221.0, 177.0], [32.0, 165.0]],
         [[177.0, 282.0], [1109.0, 288.0], [1106.0, 610.0], [174.0, 603.0]],
         [[278.0, 580.0], [1010.0, 580.0], [1010.0, 642.0], [278.0, 642.0]],
         [[1111.0, 604.0], [1160.0, 604.0], [1160.0, 618.0], [1111.0, 618.0]],
         [[1080.0, 635.0], [1155.0, 635.0], [1155.0, 652.0], [1080.0, 652.0]],
         [[918.0, 647.0], [1000.0, 647.0], [1000.0, 664.0], [918.0, 664.0]],
         [[1081.0, 652.0], [1147.0, 652.0], [1147.0, 666.0], [1081.0, 666.0]],
         [[1081.0, 670.0], [1126.0, 670.0], [1126.0, 683.0], [1081.0, 683.0]],
         [[1097.0, 663.0], [1130.0, 663.0], [1130.0, 674.0], [1097.0, 674.0]],
         [[1076.0, 676.0], [1253.0, 678.0], [1253.0, 696.0], [1076.0, 693.0]],
         [[469.0, 761.0], [901.0, 764.0], [901.0, 821.0], [469.0, 818.0]],
         [[463.0, 822.0], [902.0, 826.0], [902.0, 857.0], [463.0, 852.0]],
         [[166.0, 1039.0], [525.0, 1037.0], [525.0, 1060.0], [166.0, 1062.0]],
         [[178.0, 1061.0], [763.0, 1059.0], [763.0, 1082.0], [178.0, 1084.0]],
         [[165.0, 1102.0], [924.0, 1102.0], [924.0, 1128.0], [165.0, 1128.0]],
         [[184.0, 1123.0], [1094.0, 1125.0], [1094.0, 1151.0], [184.0, 1149.0]],
         [[165.0, 1169.0], [669.0, 1168.0], [669.0, 1191.0], [165.0, 1192.0]],
         [[167.0, 1234.0], [668.0, 1234.0], [668.0, 1257.0], [167.0, 1257.0]],
         [[187.0, 1253.0], [1039.0, 1258.0], [1039.0, 1284.0], [187.0, 1279.0]],
         [[295.0, 1327.0], [458.0, 1332.0], [457.0, 1375.0], [294.0, 1370.0]],
         [[466.0, 1330.0], [569.0, 1330.0], [569.0, 1396.0], [466.0, 1396.0]],
         [[629.0, 1331.0], [869.0, 1334.0], [869.0, 1375.0], [629.0, 1373.0]],
         [[292.0, 1369.0], [460.0, 1373.0], [460.0, 1400.0], [292.0, 1396.0]],
         [[624.0, 1372.0], [1072.0, 1374.0], [1072.0, 1401.0], [624.0, 1399.0]],
         [[409.0, 1456.0], [851.0, 1459.0], [851.0, 1503.0], [409.0, 1500.0]]],
 '分数': [0.7910715460777282,
        0.7438937872648239,
        0.8779238214095434,
        0.37094803899526596,
        0.6871711413065592,
        0.8419906993707021,
        0.7611332259007862,
        0.6775176425774893,
        0.43559680581092836,
        0.5932324767112732,
        0.8621773198246956,
        0.896224904518861,
        0.9342550689523871,
        0.9395184453689691,
        0.9641873836517334,
        0.9524000250563329,
        0.9582841634750366,
        0.9612718860308329,
        0.9428217871033627,
        0.798493766784668,
        0.7463672459125519,
        0.8557159645216805,
        0.8900542497634888,
        0.9359333357390236,
        0.8789843645962802],
 '文本': ['8-01',
        '消火栓',
        'FIREHYDRANT',
        '种花元学',
        '消连道一酒防安全连查点',
        '地址：文料大楼6楼8-',
        '码：3T-00001018',
        '区中生校区',
        '查4-1',
        '工大楼（中）上公#',
        '消火栓使用说明',
        'How to Use a Fire Hydrant',
        '1.按下消火栓箱门右侧的按钮，并打开箱门。',
        'Press thebuttonontheright sideof thefirecabinet,thenopen thedoor.',
        '2.拉出水带，并将水带两端接口分别对准消火栓固定接口和水枪接口，顺时针方向旋转、拧紧。',
        'Takeoutthehose,andconnect the interfacesof thehosewiththehydrantand '
        'thenozzlebyrotatingclockwise.',
        '3.按下消火栓箱内的启泵按钮，指示灯闪亮说明消防泵已启动。',
        '4.由一人紧握水枪，另一人逆时针旋开消火栓阀门至最大开度。',
        'Onepersonholdsthenozzle tightly,whiletheother '
        'turnsonthevalveanticlockwise tomaximumopening',
        '火警电话',
        '119',
        '内部应急电话',
        'Fire Telephone',
        'Emergency24/7Hotline(021)62238110',
        '消防设施 请保持完好']}
```

`format_output`的结果以字典形式存储了推理结果，共有三个键：`检测框`、`分数`和`文本`。检测框以三维数组形式保存了每个检测框的四个顶点的[x,y]坐标，而分数则是对应下标的检测框分数，以一维数组形式保存。文本则是每个检测框中识别出的文本，以一维数组形式保存。

结果可视化：

```python
ocr.show(ocr_img) # 展示推理结果图片
```

显示结果图片：由两部分组成，左侧为原图片，右侧为经过ocr识别出的文本，并且该文本的位置与原图片中文本的位置保持对应。

![](../images/xeduhub/ocr_show.png)

#### 4. 结果保存

```python
ocr.save(ocr_img,'ocr_result.jpg') # 保存推理结果图片
```

`save()`方法能够保存ocr识别后的结果图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

## 4. 图像分类

图像分类是一个分类任务，它能够将不同的图像划分到指定的类别中，实现最小的分类误差和最高精度。XEduHub提供了进行图像分类的模型：`cls_imagenet`，该模型的分类类别取自ImageNet的一千个分类，这意味着该模型能够将输入的图像划分到这一千个分类中的类别上。

### 代码样例

```python
from XEdu.hub import Workflow as wf
cls = wf(task='cls_imagenet') # 模型声明
img_path = 'demo/cat.png' # 指定进行推理的图片路径
result,cls_img = cls.inference(data=img_path,img_type='cv2') # 进行推理
cls_format_result = cls.format_output(lang='zh')  # 结果格式化输出
cls.show(cls_img) # 展示原图
cls.save(cls_img,'cls_result.jpg')# 保存图片
```

### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
cls = wf(task="cls_imagenet") # 模型声明
```

`wf()`中共有三个参数可以设置：

- `task`选择任务。图像分类的模型为`cls_imagenet`。
- `checkpoint`指定模型文件所在的路径，如`cls = wf(task='cls_imagenet',checkpoint='cls_imagenet.onnx') `。如果没有指定模型路径，Workflow会默认在本地同级的“checkpoints”文件夹中寻找与任务名对应的模型文件，即`cls_imagenet.onnx`。否则将通过网络到浦源平台的专用地址下载。
- `download_path`指定模型的下载路径。缺省情况下，模型文件会下载到“checkpoints”文件夹中，“checkpoints”文件夹不存在则自动建立。如果希望代码在没有网络的设备上也能运行，请同步复制`checkpoints`文件夹。如希望模型保存放在其他路径，则设置`download_path`参数，如`download_path='my_checkpoint'`。注意，`download_path`参数为文件夹名称。建议该参数留空，使用默认地址。

任务模型文件获取与存放请查看[下文](https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id132)。

#### 2. 模型推理

```python
# result = cls.inference(data='data/cat101.jpg') # 模型推理方式1
result,cls_img = cls.inference(data='data/cat101.jpg', img_type='pil') # 进行模型推理方式2
```

模型推理`inference()`可传入参数：

- `data`: 指定待分类的图片，可以是以图片路径形式传入，也可直接传入cv2或pil格式的图片。
- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出原图。
- `img_type`：返回原图，该参数指定了返回图片的格式，可选有:`['cv2','pil']`。

推理结果`result`是一个二维数组，表示这个图片在ImageNet的一千个分类中，属于每个分类的概率。

![](../images/xeduhub/cls_result.png)

`cls_img`的格式为cv2，呈现的就是一张原图，并非带模型推理结果的图。

#### 3. 结果输出

```python
format_result = cls.format_output(lang='zh') #推理结果格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(str) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_result`是一个字典，以格式化的方式展示了这张图片最有可能的分类结果。预测值表示图片分类标签在所有一千个分类中的索引，分数是属于这个分类的概率，预测类别是分类标签的内容。

```
# 输出结果
{'预测值': 281, '分数': 0.5337719, '预测类别': 'tabby, tabby cat'}
```

![](../images/xeduhub/cls_format.png)

配合展示原图：

```
cls.show(cls_img) # 展示原图
```

![](../images/xeduhub/cls_show.png)

## 5. 内容生成

内容生成模型是一种人工智能模型，它能够根据输入的提示或指令生成新的内容，如文本、图像、音频或视频。

XEduHub提供了两个图像内容生成任务：图像风格迁移`gen_style`和图像着色`gen_color`。
风格迁移

### 1） 图像风格迁移模型的使用

图像风格迁移就是根据一幅风格图像(style image)，将任意一张其他图像转化成这个风格，并尽量保留原图的内容(content image)。

XEduHub中的风格迁移使用有两类：

1. 预设风格迁移：预设好五种风格，用户只传入一张内容图像，迁移至该风格。
2. 自定义风格迁移：用户传入一张内容图像和一张风格图像，迁移至风格图像的风格。

### 实例讲解1：马赛克（mosaic）风格迁移模型的使用

每个风格使用的代码风格是类似的，接下来通过学习一个完整示范，可以达到举一反三的效果

下面是实例马赛克（mosaic）风格迁移模型的完整代码：

```python
from XEdu.hub import Workflow as wf
style = wf(task='gen_style',style='mosaic')
result, img = style.inference(data='data/cat101.jpg',img_type='cv2') # 进行模型推理
style.show(img) # 展示推理图片
style.save(img,'style_cat.jpg') # 保存推理图片
```

### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
style = wf(task='gen_style',style='mosaic')
```

`wf()`中共有四个参数可以设置：

- `task`选择任务。风格迁移的模型为`gen_style `。
- `style`选择风格迁移所使用的风格。可选的风格有`udnie`、`mosaic`、`rain-princess`、`candy`和`pointilism`，也可以用一张图片作为风格源。
- `checkpoint`指定模型文件所在的路径，如`style = wf(task='gen_style',style='mosaic',checkpoint='gen_style_mosaic.onnx') `。如果没有指定模型路径，Workflow会默认在本地同级的“checkpoints”文件夹中寻找与任务名对应的模型文件，即`gen_style_mosaic.onnx`（任务名加下划线加风格名）。否则将通过网络到浦源平台的专用地址下载。
- `download_path`指定模型的下载路径。缺省情况下，模型文件会下载到“checkpoints”文件夹中，“checkpoints”文件夹不存在则自动建立。如果希望代码在没有网络的设备上也能运行，请同步复制`checkpoints`文件夹。如希望模型保存放在其他路径，则设置`download_path`参数，如`download_path='my_checkpoint'`。注意，`download_path`参数为文件夹名称。建议该参数留空，使用默认地址。

任务模型文件获取与存放请查看[下文](https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id132)。

运行代码`wf.support_style()`可查看当前预设的风格。当前预设风格共有五种，如下图所示。

![](../images/xeduhub/style_all.png)

- udnie: 该幅作品是法国艺术家弗朗西斯·毕卡比亚 （Francis Picabia） 于 1913 年创作的一幅布面油画。这幅抽象画抽象的形式和金属色的反射让人想起机器的世界。

- mosaic: 马赛克是由有色石头、玻璃或陶瓷制成的规则或不规则的小块图案或图像，由石膏/砂浆固定到位并覆盖表面。

- rain-princess：该幅作品的作者是李奥尼德·阿夫列莫夫，他继梵高之后，当代最著名的现代印象派艺术家。风景、城市和人物在他的画笔下（更确切的可以说是刮刀），具有一种独特的风格，用色大胆、明亮，传达他的乐观。
  
- candy: 该风格通过糖果般绚丽的色块以及象征棒棒糖的圆圈图案，传递出甜蜜童真。
  
- pointilism： 点彩画是一种绘画技术，其中将小而独特的色点应用于图案以形成图像。


`style`可选参数为：`['mosaic','candy','rain-princess','udnie','pointilism']`，也可以输入一张其他图片的路径来自定义风格，如`style='fangao.jpg'`。为了方便用户使用预设风格，还可以通过输入预设风格对应的标签值来进行设定，如`style=0`。

<table class="docutils align-default">
    <thead>
        <tr class="row-odd">
            <th class="head">预设风格</th>
            <th class="head">对应标签值</th>
        </tr>
    </thead>
    <tbody>
        <tr class="row-even">
            <td>mosaic</td>
            <td>0</td>
        </tr>
        <tr class="row-even">
            <td>candy</td>
            <td>1</td>
        </tr>
        <tr class="row-even">
            <td>rain-princess</td>
            <td>2</td>
        </tr>
        <tr class="row-even">
            <td>udnie</td>
            <td>3</td>
        </tr>
        <tr class="row-even">
            <td>pointilism</td></td>
            <td>4</td>
        </tr>
    </tbody>
</table>

#### 2. 模型推理

```python
result, img = style.inference(data='data/cat101.jpg',img_type='cv2') # 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`: 待进行风格迁移的图片，可以是以图片路径形式传入，也可直接传入cv2或pil格式的图片。
- `show`: 可取值：`[true,false]` 默认为`false`。如果取值为`true`，在推理完成后会直接输出风格迁移完成后的图片。
- `img_type`: 推理完成后会直接输出风格迁移完成后的图片。该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。

模型推理返回结果：

- `result`和`img`都是三维数组，以img_type设置的格式保存了风格迁移完成后的图片。

#### 3. 结果输出

```python
style.show(img) # 展示推理后的图片
```

`show()`能够输出风格迁移后的结果图像。

![](../images/xeduhub/style_show.png)

#### 4. 结果保存

```python
style.save(img,'style_cat.jpg') # 保存推理图片
```

`save()`方法能够保存风格迁移后的图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

### 实例讲解2：自定义风格迁移模型的使用

当我们看到喜欢的风格的图像，并想要迁移到其他图像上时，我们就可以使用XEduHub中的自定义风格迁移模型。

例如我喜欢“my_style”这张图片，我想要将其风格迁移到我的风景照上，生成新的图像

**将图片的路径来自定义风格，style='demo/my_style.jpg'**

下面是实例自定义风格迁移模型的完整代码：

```python
from XEdu.hub import Workflow as wf # 导入库
style = wf(task='gen_style',style='demo/my_style.jpg') # 实例化模型
img_path = 'demo/ShangHai.jpg'  # 指定进行推理的图片路径
result, new_img = style.inference(data=img_path,img_type='cv2') # 进行模型推理
style.show(new_img) # 可视化结果
style.save(new_img, 'demo/style_my_style_ShangHai.jpg') # 保存可视化结果
```

### 2. 图像着色模型的使用

图像着色模型是将灰度图像转换为彩色图像的模型，它根据图像的内容、场景和上下文等信息来推断合理的颜色分布，实现从灰度到彩色的映射。

当我们有一张黑白图片想要为它上色时，可以使用XEduHub提供的gen_color图像着色任务。通过调用基于卷积神经网络 (CNN)训练的模型进行推理，自动地为黑白图像添加颜色，实现了快速生成逼真的着色效果。

### 代码样例

```python
from XEdu.hub import Workflow as wf # 导入库
color = wf(task='gen_color') # 实例化模型
result, img = color.inference(data='demo/gray_img1.jpg',img_type='cv2') # 进行模型推
color.show(img) # 可视化结果
color.save(img,'demo/color_img.jpg') # 保存可视化结果
```

### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf # 导入库
color = wf(task='gen_color') # 实例化模型
```

`wf()`中共有三个参数可以设置：

- `task`选择任务。图像分类的模型为`gen_color`。
- `checkpoint`指定模型文件所在的路径，如`color = wf(task='gen_color',checkpoint='gen_color.onnx') `。如果没有指定模型路径，Workflow会默认在本地同级的“checkpoints”文件夹中寻找与任务名对应的模型文件，即`gen_color.onnx`。否则将通过网络到浦源平台的专用地址下载。
- `download_path`指定模型的下载路径。缺省情况下，模型文件会下载到“checkpoints”文件夹中，“checkpoints”文件夹不存在则自动建立。如果希望代码在没有网络的设备上也能运行，请同步复制`checkpoints`文件夹。如希望模型保存放在其他路径，则设置`download_path`参数，如`download_path='my_checkpoint'`。注意，`download_path`参数为文件夹名称。建议该参数留空，使用默认地址。

任务模型文件获取与存放请查看[下文](https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id132)。


#### 2. 模型推理

```python
result, img = color.inference(data='demo/gray_img1.jpg',img_type='cv2') # 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`: 待进行风格迁移的图片，可以是以图片路径形式传入，也可直接传入cv2或pil格式的图片。
- `show`: 可取值：`[true,false]` 默认为`false`。如果取值为`true`，在推理完成后会直接输出风格迁移完成后的图片。
- `img_type`: 推理完成后会直接输出图像着色完成后的图片。该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。

模型推理返回结果：

- `result`和`img`都是三维数组，以cv2格式保存了风格迁移完成后的图片。

#### 3. 结果输出

```python
color.show(img) # 展示推理后的图片
```

`show()`能够输出着色后的结果图像。

![](../images/xeduhub/color_show.png)

#### 4. 结果保存

```python
color.save(img,'color_img.jpg') # 保存推理图片
```

`save()`方法能够保存着色后的图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

## 6. 全景驾驶感知系统

全景驾驶感知系统是一种高效的多任务学习网络，“多任务”表示该模型可同时执行交通对象检测、可行驶道路区域分割和车道检测任务，能很好地帮助自动驾驶汽车通过摄像头全面了解周围环境。我们可以在实时自动驾驶的项目中组合运用不同的检测任务，来控制车辆的动作，以达到更好的效果。XEduHub提供了进行全景驾驶感知的任务：`drive_perception`。

### 代码样例

```python
from XEdu.hub import Workflow as wf
drive = wf(task='drive_perception') # 实例化模型
result,img = drive.inference(data='demo/drive.png',img_type='cv2') # 模型推理
drive.format_output(lang='zh') # 将推理结果进行格式化输出
drive.show(img) # 展示推理图片
drive.save(img,'img_perception.jpg') # 保存推理图片
```

### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
drive = wf(task='drive_perception') # 实例化模型
```

`wf()`中共有三个参数可以设置：

- `task`选择任务。全景驾驶感知系统的模型为`drive_perception`。
- `checkpoint`指定模型文件所在的路径，如`drive = wf(task='drive_perception',checkpoint='drive_perception.onnx') `。如果没有指定模型路径，Workflow会默认在本地同级的“checkpoints”文件夹中寻找与任务名对应的模型文件，即`drive_perception.onnx`。否则将通过网络到浦源平台的专用地址下载。
- `download_path`指定模型的下载路径。缺省情况下，模型文件会下载到“checkpoints”文件夹中，“checkpoints”文件夹不存在则自动建立。如果希望代码在没有网络的设备上也能运行，请同步复制`checkpoints`文件夹。如希望模型保存放在其他路径，则设置`download_path`参数，如`download_path='my_checkpoint'`。注意，`download_path`参数为文件夹名称。建议该参数留空，使用默认地址。

任务模型文件获取与存放请查看[下文](https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id132)。

#### 2. 模型推理

```python
result,img = drive.inference(data='demo/drive.png',img_type='cv2') # 模型推理
```

![](../images/xeduhub/drive_result.png)

模型推理`inference()`可传入参数：

- `data`：指定待检测的图片。
- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出目标检测完成后的图片。
- `img_type`：目标检测完成后会返回含有检测框的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。
- `thr`: 设置检测框阈值，取值范围为`[0,1]`超过该阈值的检测框被视为有效检测框，进行显示。

模型推理返回结果：

```
[array([[671, 398,  98,  78],
       [627, 393,  26,  25]]), array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]], dtype=uint8), array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 1, 1, 1],
       [0, 0, 0, ..., 1, 1, 1],
       [0, 0, 0, ..., 1, 1, 1]], dtype=uint8)]
```

![](../images/xeduhub/drive_printresult.png)

- `result`：以三维数组的形式保存了车辆检测（红色框），车道线分割（蓝色色块），可行驶区域（绿色色块）。
- `车辆检测result[0]`：以二维数组保存了车辆目标检测框左上角顶点的坐标(x1,y1)和右下角顶点的坐标(x2,y2)（之所以是二维数组，是因为该模型能够检测多辆车，因此当检测到多辆车时，就会有多个[x1,y1,x2,y2]的一维数组，所以需要以二维数组形式保存），我们可以利用这四个数据计算出其他两个顶点的坐标，以及检测框的宽度和高度。
- `车道线分割result[1]`：以由0，1组成二维数组（w*h），保存图像中每个像素的mask，mask为1表示该像素为车道线分割目标，mask为0表示该像素是背景。
- `可行驶区域result[2]`：以由0，1组成二维数组（w*h），保存图像中每个像素的mask，mask为1表示该像素为可驾驶区域分割目标，mask为0表示该像素是背景。
- `img_with_box`(numpy.ndarray)：是个三维数组，以img_type所设置的格式保存了包含了检测框与分割目标的图片。

#### 3. 结果输出

```python
format_result = drive.format_output(lang='zh') # 将推理结果进行格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(str) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_result`以字典形式存储了推理结果，有四个键：`检测框`、`分数`、`车道线掩码`、`可行驶区域掩码`。检测框以二维数组形式保存了每个检测框的坐标信息[x1,y1,x2,y2]。

```
# 输出结果
{'检测框': [[671, 398, 98, 78], [627, 393, 26, 25]],
 '分数': [0.9999916553497314, 0.999988317489624],
 '车道线掩码': array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0]], dtype=uint8),
 '可行驶区域掩码': array([[0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       [0, 0, 0, ..., 0, 0, 0],
       ...,
       [0, 0, 0, ..., 1, 1, 1],
       [0, 0, 0, ..., 1, 1, 1],
       [0, 0, 0, ..., 1, 1, 1]], dtype=uint8)}
```

![](../images/xeduhub/drive_format.png)

```python
drive.show(img) # 展示推理图片
```

`show()`能够输出带有检测框与分割目标的图片。

![](../images/xeduhub/drive_show.png)

#### 4. 结果保存

```python
drive.save(img,'img_perception.jpg') # 保存推理图片
```

`save()`方法能够保存带有检测框与分割目标的图片。

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

## 7. 多模态图文特征提取

多模态图文特征提取技术是一种将计算机无法直接理解图像或文本转换成计算机擅长理解的数字数字向量。通过“特征提取”方式得到的数字向量，能完成零样本分类、文本翻译，图像聚类等任务。XEduHub提供了图像特征提取和文本特征提取任务：`'embedding_image'`，`'embedding_text'`。

### 图像特征提取

当我们使用图像特征提取，本质上是将图像“编码”或“嵌入”到向量形式的一系列数字中，让图像->向量。
这些向量可以捕捉图像中的局部特征，如颜色、纹理和形状等。图像特征提取有助于计算机识别图像中的对象、场景和动作等。

#### 代码样例

```python
from XEdu.hub import Workflow as wf # 导入库
img_emb = wf(task='embedding_image') # 实例化模型
image_embeddings = img_emb.inference(data='demo/cat.png') # 模型推理
print(image_embeddings) # 输出向量
```

### 代码解释

#### 1. 模型声明
```python
from XEdu.hub import Workflow as wf # 导入库
img_emb = wf(task='embedding_image') # 实例化模型
```

`wf()`中有三个参数可以设置：

- `task`选择任务。任务名称为`embedding_image `。
- `checkpoint`指定模型文件所在的路径，如`img_emb = wf(task='embedding_image',checkpoint='embedding_image.onnx') `。如果没有指定模型路径，Workflow会默认在本地同级的“checkpoints”文件夹中寻找与任务名对应的模型文件，即`embedding_image.onnx`。否则将通过网络到浦源平台的专用地址下载。
- `download_path`指定模型的下载路径。缺省情况下，模型文件会下载到“checkpoints”文件夹中，“checkpoints”文件夹不存在则自动建立。如果希望代码在没有网络的设备上也能运行，请同步复制`checkpoints`文件夹。如希望模型保存放在其他路径，则设置`download_path`参数，如`download_path='my_checkpoint'`。注意，`download_path`参数为文件夹名称。建议该参数留空，使用默认地址。

任务模型文件获取与存放请查看[下文](https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id132)。

#### 2. 模型推理

```python
image_embeddings = img_emb.inference(data='demo/cat.png') # 模型推理
```

模型推理`inference()`可传入参数：

- `data`：指定待特征提取的图片。可以直接传入图像路径`data='cat.jpg'` 或者多张图像路径列表`data= ['cat.jpg','dog.jpg'] `。

模型推理返回结果：

```
[[-3.85740042e-01 -1.81814015e-01  2.94580430e-01 -3.75688493e-01
  -9.97719467e-02 -3.99263024e-01 -4.88978140e-02  4.52158630e-01
   3.44679683e-01  8.63113552e-02  3.00731242e-01 -4.26063895e-01
   3.67905170e-01 -1.52026504e-01  2.76561409e-01 -1.71330012e-02
   1.70775101e-01  5.59596419e-02  1.35128006e-01  1.62233084e-01
  -3.01426768e-01  2.13003829e-01  4.14773196e-01  6.58608004e-02
  -3.26728284e-01 -8.21363479e-02  3.83218110e-01  1.98388845e-03
   7.52402395e-02 -9.47817862e-02 -3.44407916e-01  3.05405080e-01
   2.16008723e-02  2.35447213e-02  1.86178669e-01  9.39529836e-02
   8.10235366e-02 -1.63445249e-01  2.17456058e-01  1.91864216e+00
  -8.11877787e-01 -5.21380484e-01 -1.63235351e-01 -5.83956987e-02
   1.12670302e-01 -1.23800576e+00  3.60498071e-01  1.70865953e-01
   2.36062884e-01 -1.30168095e-01  2.28240922e-01  4.12806034e-01
  -1.44194707e-01  5.99492826e-02  5.90620399e-01  3.17890644e-02
   5.66926479e-01  3.08387011e-01 -1.93981409e-01 -3.26512724e-01
   1.30550885e+00 -1.63456917e-01 -1.71479523e-01  1.87635124e-01
  -1.62023515e-01 -5.04231989e-01  1.93340555e-01  7.79000640e-01
  -2.60949045e-01  4.51564491e-01  1.93995193e-01 -5.56772709e-01
   2.24175125e-01 -2.24783584e-01 -4.67097044e-01 -1.89082280e-01
  -5.35702705e-03 -5.06964564e-01 -6.83084354e-02 -1.69336617e-01
  -3.91773462e-01 -3.84069264e-01 -5.86372465e-02 -4.74828005e-01
   1.57550305e-01 -4.58408058e-01  1.21461821e+00 -5.39815187e-01
  -1.67127803e-01  1.52734384e-01  1.82488933e-02 -3.33601654e-01
  -5.91347456e+00  1.90935433e-01 -8.74737203e-01  5.37124157e-01
   3.54932584e-02 -4.14988220e-01 -2.38636166e-01  1.60797548e+00
   2.25136548e-01 -4.80371296e-01 -1.21367343e-01  2.91841656e-01
  -1.68474913e-01  4.46746111e-01  2.07979515e-01  2.23423451e-01
   1.83344692e-01 -1.40982628e-01 -6.63975060e-01  6.43657669e-02
   4.81830776e-01  1.65944129e-01 -2.49111857e-02 -1.20062128e-01
   1.40799791e-01  2.88082600e-01  2.76114374e-01 -1.58252344e-01
   2.33967036e-01  3.16998273e-01  3.99250299e-01 -9.65632871e-03
   1.15269527e-01 -3.71721506e-01  1.94188878e-01  3.90523285e-01
   4.12000626e-01 -1.80458277e-03 -3.60838771e-01 -1.12181015e-01
  -5.05329251e-01  7.17117429e-01 -5.48307180e-01  2.06300825e-01
  -4.58916605e-01 -1.18503585e-01 -1.23531006e-01 -2.95604229e-01
   1.80596381e-01  1.27064258e-01 -1.58087850e-01  2.26205975e-01
  -6.21176720e-01 -7.06477538e-02  5.28910637e-01  5.26533902e-01
   9.11014974e-02  3.42928290e-01 -3.36795449e-01  2.60566026e-01
  -5.16621172e-01  2.01655403e-01 -4.43294048e-01 -1.68800324e-01
  -2.02344880e-02 -7.48819470e-01  6.20617419e-02 -1.59256980e-01
   3.74310374e-01  1.32567585e-01 -7.37757385e-02  8.41598064e-02
   2.06333309e-01 -2.64279962e-01  9.55645740e-02  1.89634040e-01
  -2.16711044e-01 -5.38360178e-02 -2.44866431e-01  6.27175808e-01
   4.68757153e-02  3.16201866e-01 -2.51018405e-01  2.42545903e-02
   9.87151146e-01 -3.33542049e-01  3.88702422e-01  2.55128801e-01
   3.66802216e-01 -7.47172385e-02 -5.85177168e-02 -3.47979367e-05
  -1.37775317e-01  6.06879890e-02  1.93679333e-01 -5.56745172e-01
  -3.32078993e-01 -5.45896962e-02  2.34644830e-01 -1.59180284e-01
   3.32888722e-01  1.31697670e-01  6.03572786e-01  1.21926449e-01
   8.54009390e-02 -6.74351096e-01 -1.53455126e+00 -1.49148270e-01
  -1.66233048e-01 -6.23880923e-02  1.20600522e-01  8.16958994e-02
   1.58847123e-02 -7.76146650e-02 -3.43320146e-02 -1.23784691e-01
  -1.11312561e-01  2.94828147e-01 -4.02154982e-01  9.57918584e-01
  -5.63874245e-01  3.49417627e-01  7.77897388e-02 -5.08594215e-01
   3.05591300e-02 -2.32043847e-01  9.69643235e-01 -5.39023459e-01
   4.33879554e-01  8.27765942e-01 -1.34395599e-01 -2.25581735e-01
  -2.71910757e-01  2.28383809e-01 -6.05013222e-04  2.75357395e-01
  -2.09673107e-01 -1.57775283e-01  1.27838835e-01  1.00237012e-01
  -4.72770751e-01  3.51460636e-01  3.12582344e-01  2.68262267e-01
   2.98875272e-01 -2.82303870e-01 -1.84585154e-01 -3.18509102e-01
   1.99579477e-01  1.22903138e-01  3.13289464e-04 -1.19046569e-02
   2.33945757e-01  4.52335238e-01  5.41643262e-01 -1.11395741e+00
  -1.81421086e-01 -1.79955930e-01 -7.46780261e-02  1.34541512e-01
   1.62621692e-01 -2.31587499e-01  1.91942528e-01  7.60079473e-02
  -8.75764787e-02 -1.25062481e-01  3.01574469e-01 -3.55449557e-01
  -2.55525708e-01  5.87767124e-01  1.70087546e-01 -1.90540528e+00
  -2.10818663e-01 -4.08018738e-01 -1.20051704e-01 -2.18928725e-01
  -3.14087391e-01 -4.32843268e-02 -2.06375062e-01 -1.36374623e-01
   6.00037217e-01  3.48301530e-02 -3.98940384e-01  1.20394975e-01
   1.24070890e-01  1.04261652e-01 -2.98556775e-01 -1.18747264e-01
   2.42990702e-01  1.51160806e-02 -4.76059765e-01  2.83466309e-01
  -2.65547872e-01  2.53341258e-01 -1.21838975e+00 -4.79129553e-02
  -3.88028845e-02  1.18025638e-01 -1.82919413e-01 -3.47036183e-01
  -3.42601538e-02  1.42673150e-01  2.14231908e-01 -2.30143934e-01
   1.30996168e-01 -1.37965798e-01  3.79844606e-01  4.15164113e-01
   1.05876885e-01 -1.57549411e-01 -2.77942568e-01 -4.60220486e-01
   1.38039052e-01 -1.69159055e-01 -3.82189423e-01  3.77442986e-02
   2.67798603e-01 -5.85539937e-02 -4.75669950e-01  3.39839756e-02
   7.16263771e-01 -2.56723523e-01  2.77137637e-01  4.75717485e-01
   2.24060252e-01  1.42611891e-01 -1.47834927e-01 -2.51081705e-01
   5.57422936e-01 -1.40718400e-01 -2.34217167e-01  4.03005749e-01
   1.54427320e-01  3.60494554e-01  9.46574286e-02 -2.43841931e-02
   1.44166082e-01 -4.30197239e-01  7.96608254e-02 -4.21275467e-01
   2.90648878e-01  8.94440413e-02 -5.84194437e-02  3.80072743e-01
   1.68659866e-01 -1.26878828e-01  3.88667434e-02 -5.88277280e-02
  -2.54280746e-01 -5.78271866e-01  1.36692956e-01  5.84963083e-01
   1.42402038e-01 -3.91594678e-01 -2.75257006e-02 -3.98596495e-01
   2.12786466e-01 -1.99996769e-01 -6.46989644e-01 -5.18083394e-01
   7.53146887e-01 -2.66945362e-01 -2.31973708e-01  5.26251793e-01
  -2.86485106e-01 -7.33324647e-01  3.97864729e-02  1.16903648e-01
   6.12574518e-02  1.77814111e-01  2.99390286e-01  1.37499496e-01
  -4.10501480e-01  1.91558301e-02 -1.72412276e-01 -2.86826253e-01
  -1.28502920e-01 -3.30712676e-01  3.59060019e-01  3.30571085e-01
  -3.34168911e-01 -4.03294861e-01 -6.88896000e-01 -8.84154737e-02
   8.99289250e-01  4.45285738e-01 -9.32471633e-01  3.20112407e-02
   4.82425421e-01 -7.42304802e-01 -5.27542889e-01  7.85614848e-02
   4.69861269e-01  1.79590791e-01  1.59789890e-01 -2.87791759e-01
   4.69775200e-02 -1.83868849e+00  1.53750971e-01  4.77876440e-02
   3.37841123e-01  1.50803804e-01  1.39148414e-01  3.65805954e-01
   1.10074878e-02 -3.03741008e-01 -1.82401910e-01 -4.56681699e-02
   7.47369975e-02  5.53856045e-03  1.22067243e-01  2.81827092e-01
  -8.37573111e-02  8.54545832e-02 -5.03253818e-01 -4.87406909e-01
   9.99620199e-01 -4.46691096e-01  4.88039315e-01  4.28979099e-03
  -7.01835155e-01  1.41174972e-01 -4.24902737e-02 -1.45253658e-01
   2.06106380e-02 -8.70095640e-02  2.44391918e-01 -6.75197363e-01
  -7.78099895e-02  6.13597393e-01 -2.50040293e-02 -2.12778434e-01
   1.43364608e-01  5.92731893e-01 -5.92901148e-02  2.67523468e-01
  -9.79188681e-01  6.87061399e-02 -2.04341471e-01  3.16503465e-01
  -5.24448574e-01  4.33634557e-02  1.32866547e-01 -5.65323114e-01
   1.08812883e-01  2.03480750e-01  2.07546815e-01 -1.16574995e-01
  -1.09661967e-02  2.19430268e-01 -7.74818435e-02  2.30491683e-01
   5.95359147e-01 -4.35534179e-01 -3.18401903e-01 -2.49920055e-01
   9.30231512e-02 -2.82962203e-01  1.35547385e-01 -6.19788170e-01
  -5.45631349e-02  8.27184916e-01 -2.99773753e-01  5.22855222e-01
   6.19698524e-01  8.09411883e-01  1.37993291e-01  3.32339823e-01
  -3.24640989e-01 -1.05401665e-01  2.94174254e-01  5.19716069e-02
   6.32487774e-01  3.20299491e-02 -4.62978512e-01 -1.87023833e-01
   4.52803016e-01 -3.47914696e-02  3.27060193e-01 -4.21827257e-01
  -4.97323275e-01 -4.16128635e-01 -3.15771133e-01  5.17482832e-02
  -3.62640381e-01 -1.50059134e-01 -4.28877860e-01 -7.57826626e-01
   2.19028667e-01 -4.49721038e-01  2.82200187e-01  1.38432071e-01
  -6.74649239e-01  5.60074598e-02 -1.86310157e-01  3.48043889e-02
  -4.19803619e-01 -1.81031242e-01  2.40428239e-01 -6.49814829e-02
   4.65104222e-01 -1.13697350e-02  1.97285146e-01  1.71784297e-01
  -1.05541974e-01  3.76087278e-01 -4.27732319e-01 -1.43245876e-01
  -2.69167125e-01  4.53836232e-01 -1.38121948e-01  3.12318951e-02
   2.15785950e-02 -2.58834958e-01 -2.91318238e-01 -1.84020817e-01
  -2.87409008e-01  4.47631925e-01  1.90001428e-02 -3.18864509e-02]]
```

![](../images/xeduhub/emb_show1.png)

- `result`：以二维数组的形式保存了每张图片特征提取后的512维向量。

### 文本特征提取
当我们使用文本特征提取，本质上是将文本的上下文和场景“编码”或“嵌入”到向量形式的一系列数字中，让文本->向量。
这些向量将词语映射到数值空间中，使得词语成为有意义的数值向量。

因为该模型的训练集来源是互联网网页提取的4亿对图像文本对的编码，所以这里的文本可以为网络上出现的任意名词，或是一段文字。你可以加入很多描述性文本，让之后的“零样本分类”变得十分有趣！

#### 代码样例

```python
from XEdu.hub import Workflow as wf # 导入库
txt_emb = wf(task='embedding_text') # 实例化模型
txt_embeddings = txt_emb.inference(data=['a black cat','a yellow cat']) # 模型推理
print(txt_embeddings) # 输出向量
```

### 代码解释

#### 1. 模型声明
```python
from XEdu.hub import Workflow as wf # 导入库
txt_emb = wf(task='embedding_text') # 实例化模型
```

`wf()`中只有一个参数可以设置：

- `task`选择任务。任务名称为`embedding_text`。

- `checkpoint`指定模型文件所在的路径，如`txt_emb = wf(task='embedding_text',checkpoint='embedding_text.onnx') `。如果没有指定模型路径，Workflow会默认在本地同级的“checkpoints”文件夹中寻找与任务名对应的模型文件，即`embedding_text.onnx`。否则将通过网络到浦源平台的专用地址下载。
- `download_path`指定模型的下载路径。缺省情况下，模型文件会下载到“checkpoints”文件夹中，“checkpoints”文件夹不存在则自动建立。如果希望代码在没有网络的设备上也能运行，请同步复制`checkpoints`文件夹。如希望模型保存放在其他路径，则设置`download_path`参数，如`download_path='my_checkpoint'`。注意，`download_path`参数为文件夹名称。建议该参数留空，使用默认地址。

#### 2. 模型推理

```python
txt_embeddings = txt_emb.inference(data=['a black cat','a yellow cat']) # 模型推理
```

模型推理`inference()`可传入参数：

- `data`：指定待特征提取的文本。可以直接传入文本`data= 'cat' `或者多条文本列表`data= ['a black cat','a yellow cat']`。

模型推理返回结果：

```
[[ 0.41094494 -0.29223138 -0.06906292 ...  0.00064418 -0.23246601
   0.12385264]
 [ 0.24434651 -0.19498482 -0.30111405 ... -0.06937407 -0.2500025
   0.08038913]]
```

![](../images/xeduhub/emb_show2.png)

- `result`：以二维数组的形式保存了每条文本特征提取后的512维向量。

### 提完了特征能干啥？

零样本分类！

什么是零样本分类呢？举个例子，现在我们想要分类图片中的猫是黑色的还是黄色的，按照图像分类的方式，我们需要收集数据集，并且标注数据集，再进行模型训练，最后才能使用训练出来的模型对图像进行分类。而现在，我们使用的“图像特征提取”和“文本特征提取”只需通过特征向量就可以进行分类，避免了大量的标注工作。

上文中我们已经通过图像特征提取和文本特征提取把`cat.jpg`,`'a black cat'`,`'a yellow cat'`分别变成了3堆数字（3个512维向量），但是很显然，我们看不懂这些数字，但是计算机可以！
通过让计算机将数字进行运算，即将图像和文本的特征向量作比较，就能看出很多信息，这也叫计算向量之间相似度。

为了方便大家计算向量之间的相似度，我们也提供了一系列数据处理函数，函数具体内容请见<a href="https://xedu.readthedocs.io/zh/master/about/functions.html#">XEdu的常见函数</a>。

下面就示范使用<a href="https://xedu.readthedocs.io/zh/master/about/functions.html#cosine-similarity">cosine_similarity</a>比较两个embedding序列的相似度。可以直接使用[get_similarity](https://xedu.readthedocs.io/zh/master/about/functions.html#get-similarity)函数，选择method='cosine'来实现。

```python
from XEdu.utils import get_similarity # 导入库
get_similarity(image_embeddings, txt_embeddings,method='cosine') # 计算相似度
```

该函数可以比较两个embedding序列的相似度，这里的相似度是以余弦相似度为计算指标的，其公式为：$$Cosine(x,y) = \frac{x \cdot y}{|x||y|}$$。

假设输入的待比较embedding序列尺度分别为(N, D)和(M, D)，则输出的结果尺度为(N, M)。

现在我们可以看到cat.jpg与'a black cat'向量的相似度为0.007789988070726395，而与'a yellow cat'向量的相似度为0.9922100305557251。显而易见，这张可爱的黄色猫咪图像与'a yellow cat'文本描述更为贴近。


## 8. MMEdu模型推理

XEduHub现在可以支持使用MMEdu导出的onnx模型进行推理啦！如果你想了解如何使用MMEdu训练模型，可以看这里：[解锁图像分类模块：MMEduCls](https://xedu.readthedocs.io/zh/master/mmedu/mmeducls.html)、[揭秘目标检测模块：MMEduDet](https://xedu.readthedocs.io/zh/master/mmedu/mmedudet.html#mmedudet)。

如果你想了解如何将使用[MMEdu](https://xedu.readthedocs.io/zh/master/mmedu.html)训练好的模型转换成ONNX格式，可以前往[最后一步：模型转换](https://xedu.readthedocs.io/zh/master/mmedu/mmedumodel_convert.html)。OK，准备好了ONNX模型，那么就开始使用XEduHub吧！

### MMEdu训练的图像分类模型

#### 代码样例

```python
from XEdu.hub import Workflow as wf
mmcls = wf(task='mmedu',checkpoint='cats_dogs.onnx')# 指定使用的onnx模型
result, result_img =  mmcls.inference(data='data/cat.png',img_type='pil')# 进行模型推理
format_result = mmcls.format_output(lang='zh') # 推理结果格式化输出
mmcls.show(result_img) # 展示推理结果图片
mmcls.save(result_img,'new_cat.jpg')# 保存推理结果图片
```

#### 代码解释

##### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
mmcls = wf(task='mmedu',checkpoint='cats_dogs.onnx') # 指定使用的onnx模型
```

`wf()`中共有两个参数可以设置：

- `task`：只需要设置task为`mmedu` ，而不需要指定是哪种任务。
- `checkpoint`：指定你的模型的路径，该参数不能为空，如`checkpoint='cats_dogs.onnx'`。

这里我们以猫狗分类模型为例，项目指路：[猫狗分类](https://www.openinnolab.org.cn/pjlab/project?id=63c756ad2cf359369451a617&sc=647b3880aac6f67c822a04f5#public)。

##### 2. 模型推理

```python
result, result_img =  mmcls.inference(data='data/cat101.jpg',img_type='pil') # 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`：指定待检测的图片。
- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出目标检测完成后的图片。
- `img_type`：分类完成后会返回含有分类标签的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。
- `thr`(float): 设置推理阈值，取值范围为`[0,1]`，预测结果的置信度高于这个阈值时，这些结果才会被认为是有效的。

`result`是一个字典，包含三个键：`标签`、`置信度`和`预测结果`。

`result_img`以img_type设置的格式保存了模型推理完成后的图片（原图+推理结果）。

##### 3. 结果输出

```python
format_result = mmcls.format_output(lang='zh') # 推理结果格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(str) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

```
# 输出结果
{'标签': 0, '置信度': 0.79525626, '预测结果': 'cat'}
```

![](../images/xeduhub/mmcls_format2.png)

`format_result`以字典形式保存了模型的推理结果，包括所属`标签`、`置信度`、以及`预测结果`。

```python
mmcls.show(result_img) # 展示推理结果图片
```

`show()`能够推理后的结果图像。与原图相比，结果图片在左上角多了`pred_label`, `pred_socre`和`pred_class`三个数据，对应着标签、置信度和预测结果。

![](../images/xeduhub/mmcls_show.png)

##### 4. 结果保存

```python
mmcls.save(img,'new_cat.jpg') # 保存推理结果图片
```

`save()`方法能够保存推理后的结果图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

### MMEdu的目标检测模型

#### 代码样例

```python
from XEdu.hub import Workflow as wf
mmdet = wf(task='mmedu',checkpoint='plate.onnx') # 指定使用的onnx模型
result, result_img =  mmdet.inference(data='data/plate0.png',img_type='pil') # 进行模型推理
format_result = mmdet.format_output(lang='zh') # 推理结果格式化输出
mmdet.show(result_img) # 展示推理结果图片
mmdet.save(result_img,'new_plate.jpg') # 保存推理结果图片
```

#### 代码解释

##### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
mmdet = wf(task='mmedu',checkpoint='plate.onnx') # 指定使用的onnx模型
```

`wf()`中共有两个参数可以设置：

- `task`：只需要设置task为`mmedu` ，而不需要指定是哪种任务。
- `checkpoint`：指定你的模型的路径，该参数不能为空，如`checkpoint='plate.onnx'`。

这里以车牌识别为例进行说明。项目指路：[使用MMEdu实现车牌检测](https://www.openinnolab.org.cn/pjlab/project?id=641426fdcb63f030544017a2&backpath=/pjlab/projects/list#public)

##### 2. 模型推理

```python
result, result_img =  mmdet.inference(data='data/plate0.png',img_type='pil') # 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`：指定待检测的图片。
- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出目标检测完成后的图片。
- `img_type`：目标检测完成后会返回含有检测框的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。
- `thr`(float): 设置检测框阈值，取值范围为`[0,1]`，预测结果的置信度高于这个阈值时，这些结果才会被认为是有效的。

`result`的结果是一个数组，里面保存了结果字典。该字典有四个键：`标签`、`置信度`、`坐标`以及`预测结果`。其中坐标表示了检测框的两个顶点：左上(x1,y1)和右下(x2,y2)。

`result_img`以img_type设置的格式保存了模型推理完成后的图片（原图+推理结果）。

##### 3. 结果输出

```python
format_result = mmdet.format_output(lang='zh') # 推理结果格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(str) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

```
# 输出结果
[{'标签': 0,
  '置信度': 0.63597506,
  '坐标': {'x1': 358, 'y1': 476, 'x2': 680, 'y2': 580},
  '预测结果': 'plate'}]
```

![](../images/xeduhub/mmdet_format2.png)

`format_output`的结果是一个数组，里面保存了结果字典。该字典有四个键：`标签`、`置信度`、`坐标`以及`预测结果`。其中坐标表示了检测框的两个顶点：左上(x1,y1)和右下(x2,y2)。

```python
mmdet.show(result_img) # 展示推理结果图片
```

`show()`能够推理后的结果图像。与原图相比，结果图片还包含车牌周围的检测框以及结果信息。

![](../images/xeduhub/mmdet_show.png)

##### 4. 结果保存

```python
mmdet.save(img,'new_plate.jpg') # 保存推理结果图片
```

`save()`方法能够保存推理后的结果图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

## 9. BaseNN模型推理

XEduHub现在可以支持使用BaseNN导出的onnx模型进行推理啦！如果你想了解如何将使用[BaseNN](https://xedu.readthedocs.io/zh/master/basenn.html)训练好的模型转换成ONNX格式，可以看这里：[BaseNN模型文件格式转换](https://xedu.readthedocs.io/zh/master/basenn/introduction.html#id24)。OK，准备好了ONNX模型，那么就开始使用XEduHub吧！

#### 代码样例

```python
# 使用BaseNN训练的手写数字识别模型进行推理
from XEdu.hub import Workflow as wf
basenn = wf(task='basenn',checkpoint='basenn.onnx') # 指定使用的onnx模型
result = base.inference(data='data/6.jpg') # 进行模型推理
format_result = basenn.format_output()
```

#### 代码解释

##### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
basenn = wf(task='basenn',checkpoint='basenn.onnx') # 指定使用的onnx模型
```

`wf()`中共有两个参数可以设置：

- `task`：只需要设置task为`basenn` ，而不需要指定是哪种任务。
- `checkpoint`：指定你的模型的路径，该参数不能为空，如`checkpoint='basenn.onnx'`。

##### 2. 模型推理

```python
result = base.inference(data='data/6.jpg') # 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`：指定待检测的图片。

`result`的结果是一个二维数组，第一个元素表示这张图属于0-9的数字类别的概率，可以看到当为0时，概率接近，因此该手写数字是0。

```
# 输出结果
[array([[1.0000000e+00, 1.2138132e-24, 1.7666091e-10, 1.8000461e-19,
         1.5096989e-24, 1.4399931e-17, 1.0446696e-16, 3.8026964e-19,
         2.8802003e-18, 1.3136030e-15]], dtype=float32)]
```

![](../images/xeduhub/basenn_res.png)

**注意！**基于BaseNN模型推理结果不包含图片！不需要指定`img_type`参数并返回图片，因为大部分使用BaseNN解决的任务只需要输出分类标签、文本或者数组数据等。

##### 3. 结果输出

```python
format_result = basenn.format_output()
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(str) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

```
# 输出结果
{0: {'预测值': 0, '分数': 1.0}}
```

![](../images/xeduhub/basenn_format2.png)

`format_output`的结果是一个结果字典，这个字典的第一个元素有两个键，`预测值`、`分数`，代表着该手写数字的分类标签以及属于该分类标签的概率。

## 10. BaseML模型推理

XEduHub现在可以支持使用BaseML导出的pkl模型文件进行推理啦！如果你想了解如何将使用[BaseML](https://xedu.readthedocs.io/zh/master/baseml.html)训练模型并保存成.pkl模型文件，可以看这里：[BaseML模型保存](https://xedu.readthedocs.io/zh/master/baseml/introduction.html#id16)。OK，准备好了pkl模型，那么就开始使用XEduHub吧！

#### 代码样例

```python
# 使用BaseML训练的鸢尾花聚类模型推理
from XEdu.hub import Workflow as wf
baseml = wf(task='baseml',checkpoint='baseml.pkl') # 指定使用的pkl模型
data = [[5.1,1.5],[7,4.7]] # 该项目中训练数据只有两维，因此推理时给出两维数据
result= baseml.inference(data=data) # 进行模型推理
format_output = baseml.format_output(lang='zh') # 推理结果格式化输出
```

#### 代码解释

##### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
baseml = wf(task='baseml',checkpoint='baseml.pkl') # 指定使用的pkl模型
```

`wf()`中共有两个参数可以设置：

- `task`：只需要设置task为`baseml` ，而不需要指定是哪种任务。
- `checkpoint`：指定你的模型的路径，该参数不能为空，如`checkpoint='baseml.pkl'`。

##### 2. 模型推理

```python
data = [[5.1,1.5],[7,4.7]] # 该项目中训练数据只有两维，因此推理时给出两维数据
result= baseml.inference(data=data) # 进行模型推理
```

`mmdet.inference`可传入参数：

- `data`：指定待推理数据（数据类型和格式跟模型训练有关）。

**注意！**基于BaseML模型推理结果不包含图片，不需要指定`img_type`参数并返回图片，因为大部分使用BaseML解决的任务只需要输出分类标签、文本或者数组数据等。

`result`的结果是一个一维数组，里面保存了模型推理结果。

```
# result
array([1，0])
```

##### 3. 结果输出

```python
format_output = baseml.format_output(lang='zh')# 推理结果格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(str) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

```
# 输出结果
{'预测值': array([1，0])}
```

`format_result`以字典形式保存了模型的推理结果，由于使用的是聚类模型，输出结果为这两个特征数据所对应的聚类标签。

如果此时你有冲动去使用BaseML完成模型训练到推理，再到转换与应用，快去下文学习[BaseML的相关使用](https://xedu.readthedocs.io/zh/master/baseml.html)吧！

## 11. 其他onnx模型推理

XEduHub现在可以支持使用用户自定义的ONNX模型文件进行推理啦！这意味着你可以不仅仅使用MMEdu或者BaseNN训练模型并转换而成的ONNX模型文件进行推理，还可以使用其他各个地方的ONNX模型文件，但是有个**重要的前提：你需要会使用这个模型，了解模型输入的训练数据以及模型的输出结果**。OK，如果你已经做好了充足的准备，那么就开始使用XEduHub吧！

#### 代码样例

```python
from XEdu.hub import Workflow as wf
import cv2
import numpy as np

custom = wf(task="custom",checkpoint="custom.onnx")

def pre(path): # 输入数据（此处为文件路径）前处理的输入参数就是模型推理时的输入参数，这里是图片的路径
    """
    这个前处理方法实现了将待推理的图片读入并进行数字化，调整数据类型、增加维度、调整各维度的顺序。
    """
    img = cv2.imread(path) # 读取图像
    img = img.astype(np.float32) # 调整数据类型
    img = np.expand_dims(img,0) # 增加batch维
    img = np.transpose(img, (0,3,1,2)) # [batch,channel,width,height]
    return img # 输出前处理过的数据（此处为四维numpy数组）

def post(res,data): # 输入推理结果和前处理后的数据
    """
    这个后处理方法实现了获取并返回推理结果中置信度最大的类别标签。
    """
    res = np.argmax(res[0]) # 返回类别索引
    return res # 输出结果

result = custom.inference(data='det.jpg',preprocess=pre,postprocess=post)
print(result)
```

#### 代码解释

##### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
custom = wf(task="custom",checkpoint="custom.onnx")
```

`wf()`中共有两个参数可以设置：

- `task`：只需要设置task为`custom` ，而不需要指定是哪种任务。
- `checkpoint`：指定你的模型的路径，如`checkpoint='custom.onnx'`。

##### 2. 模型推理

```python
import cv2
import numpy as np

custom = wf(task="custom",checkpoint="custom.onnx")

def pre(path): # 输入数据（此处为文件路径）前处理的输入参数就是模型推理时的输入参数，这里是图片的路径
    """
    这个前处理方法实现了将待推理的图片读入并进行数字化，调整数据类型、增加维度、调整各维度的顺序。
    """
    img = cv2.imread(path) # 读取图像
    img = img.astype(np.float32) # 调整数据类型
    img = np.expand_dims(img,0) # 增加batch维
    img = np.transpose(img, (0,3,1,2)) # [batch,channel,width,height]
    return img # 输出前处理过的数据（此处为四维numpy数组）

def post(res,data): # 输入推理结果和前处理后的数据
    """
    这个后处理方法实现了获取并返回推理结果中置信度最大的类别标签。
    """
    res = np.argmax(res[0]) # 返回类别索引
    return res # 输出结果
```

在这里，使用自定义的ONNX模型进行推理的时候，你需要自己的需求实现模型输入数据的前处理以及输出数据的后处理方法，确保在进行模型推理的正常运行。

举一个例子，如果你手中有一个onnx模型文件，这个模型是一个**目标检测模型**，在训练时的训练数据是将图片读入后进行数字化处理得到的**numpy数组**，那么你在使用XEduHub时，基于该模型进行推理之前，**需要设计对应的前处理方法**将图片进行数字化。

同样地，如果你的模型的输出结果是一个一维数组，里面包含了所有类别标签对应的置信度，那么如果你想要输出检测出来的最有可能的类别标签，你就**需要设计后处理方法**，使得输出的推理结果满足你的需要。

以上是前处理和后处理方法的代码示例，以前文提到的目标检测为例。

在定义好了前处理和后处理函数之后，就可以进行模型推理了！记得要传入前后处理函数的名称到模型参数中。

```python
result = custom.inference(data='det.jpg',preprocess=pre,postprocess=post)
print(result
# 812
```

![](../images/xeduhub/custom_result.png)

模型推理`inference()`可传入参数：

- `data`：指定待检测的图片
- `preprocess`: 指定前处理函数
- `postprocess`：指定后处理函数

## XEduHub任务模型文件获取与存放

XEduHub提供了大量优秀的任务模型，我们不仅可以通过`wf()`代码的运行实现模型的自动下载，还可以自主通过浦源平台（上海AI实验室的另一个平台）下载。

只要进入<a href="https://openxlab.org.cn/models/detail/xedu/hub-model">模型仓库</a>，在Model File里就可以看到各种任务模型。网址：<a href="https://openxlab.org.cn/models/detail/xedu/hub-model">https://openxlab.org.cn/models/detail/xedu/hub-model</a>

![](../images/xeduhub/downloadmodel.png)

没有网络，如何让代码`wf()`运行时找到找到模型文件呢？

在没有指定模型路径`checkpoints`参数的情况下，`wf()`运行时会先检查是否已下载了对应任务的模型，检查的顺序如下：

1. 本地的同级目录的checkpoints文件夹中，与任务同名的模型文件。
2. 本地缓存中。

如果都没有，就会到网络上下载。

因此，无论是网络下载还是自己训练的模型使用，有两种解决思路：

- 在本地同级目录中新建checkpoints文件夹，将模型存放在该文件夹中。
  
- 使用参数`checkpoint，指定模型路径，如`model=wf(task='pose_body17',checkpoint='my_path/pose_body17.onnx')`

最后提醒一下，自己到网络下载或自己训练的模型需要是ONNX格式。ONNX是由微软、亚马逊 、Facebook 和 IBM 等公司共同开发的开放神经网络交换格式，即`Open Neural Network Exchange`，兼容性比较好。

