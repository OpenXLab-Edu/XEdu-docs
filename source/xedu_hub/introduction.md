# XEduHub功能详解

XEduHub作为一个深度学习工具库，集成了许多深度学习领域优质的SOTA模型，能够帮助用户在不进模型训练的前提下，用少量的代码，快速实现计算机视觉、自然语言处理等多个深度学习领域的任务。

XEduHub的核心分为两个部分：内置模型和Workflow。


## 什么是Workflow？

我们在使用XEduHub时都需要执行这段代码`from XEdu.hub import Workflow as wf`。Workflow的基本逻辑是使用训练好的模型对数据进行推理。

那什么是Workflow呢？在使用XEduHub里的单个模型时，Workflow就是模型推理的推理流，从数据，到输入模型，再到输出推理结果。在使用XEduHub里多个模型进行联动时，Workflow可以看做不同模型之间的数据流动，例如首先进行多人的目标检测，将检测到的数据传入关键点识别模型从而对每个人体进行关键点识别。

XEduHub就像是一个充满了AI玩具的箱子，里面有很多已经做好的AI模型，我们可以直接用它们来完成不同的任务。根据自身需求，组建属于你自己的Workflow。下面开始介绍Workflow中丰富的深度学习工具。

### 强烈安利项目<a href="https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&backpath=/pjlab/projects/list#public">XEduHub实例代码-入门完整版</a>

<a href="https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&backpath=/pjlab/projects/list#public">https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&backpath=/pjlab/projects/list#public</a>

通过学习“XEduHub实例代码-入门完整版”，可以在项目实践中探索XEduHub的魅力，项目中通俗易懂的讲解和实例代码也能帮助初学者快速入门XEduHub。

## 1. 目标检测

目标检测是一种计算机视觉任务，其目标是在图像或视频中检测并定位物体的位置，并为每个物体分配类别标签。

实现目标检测通常包括特征提取、物体位置定位、物体类别分类等步骤。这一技术广泛应用于自动驾驶、安全监控、医学影像分析、图像搜索等各种领域，为实现自动化和智能化应用提供了关键支持。

XEduHub目标支持目标检测任务有：coco目标检测`det_coco`、人体检测`det_body`、人脸检测`det_face`和人手检测`det_hand`。

### coco目标检测

COCO（Common Objects in Context）是一个用于目标检测和图像分割任务的广泛使用的数据集和评估基准。它是计算机视觉领域中最重要的数据集之一，在XEduHub中的该模型能够检测出80类coco数据集中的物体：`det_coco`，以及加强版`det_coco_l`。

![](../images/xeduhub/new_coco.png)

若要查看coco目标检测中的所有类别可运行以下代码：

```python
wf.coco_class()
```

#### 代码样例

```python
from XEdu.hub import Workflow as wf
det_coco = wf(task='det_coco')
result,img_with_box = det_coco.inference(data='data/det_coco.jpg',img_type='pil') # 进行模型推理
format_result = det_coco.format_output(lang='zh')# 将推理结果进行格式化输出
det_coco.show(img_with_box)# 展示推理图片
det_coco.save(img_with_box,'img_with_box.jpg')# 保存推理图片
```

#### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
det_coco = wf(task='det_coco')
```

`wf()`中共有三个参数可以设置：

- `task`选择任务。coco目标检测的模型为`det_coco`, `det_coco_l`。`det_coco_l`相比`det_coco`模型规模较大，性能较强，但是推理的速度较慢。
- `checkpoints`指定模型的路径，默认在本地同级的checkpoints文件夹中寻找任务对应的模型，如`checkpoints='my_checkpoint/model.onnx'`。
- `download_path`指定模型的下载路径。默认是下载到同级的checkpoints文件夹中，如`download_path='my_checkpoint'`。

任务模型下载与存放请查看<a href="https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id122">XEduHub任务模型资源下载与存放</a>。

#### 2. 模型推理

```python
result,img_with_box = det_coco.inference(data='data/det_coco.jpg',img_type='pil') # 进行模型推理
```


模型推理`inference()`可传入参数：

- `data`(string|numpy.ndarray): 指定待目标检测的图片。

- `show`(flag): 可取值`[True,False]` ,如果取值为`True`，在推理完成后会直接输出目标检测完成后的图片，默认为`False`。

- `img_type`(string): 关键点识别完成后会返回含有目标检测框的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，即如果不传入值，则不会返回图。

- `thr`(float): 设置检测框阈值，取值范围为`[0,1]`超过该阈值的检测框被视为有效检测框，进行显示。

- `target_class`(string)：该参数在使用`cocodetect`的时候可以指定要检测的对象，如：`person`，`cake`等等。

模型推理返回结果：


- `result`：以二维数组的形式保存了检测框左上角顶点的坐标(x1,y1)和右下角顶点的坐标(x2,y2)（之所以是二维数组，是因为该模型能够检测多个对象，因此当检测到多个对象时，就会有多个[x1,y1,x2,y2]的一维数组，所以需要以二维数组形式保存），我们可以利用这四个数据计算出其他两个顶点的坐标，以及检测框的宽度和高度。

![](../images/xeduhub/det_res.png)

- `img_with_box：`：是个三维数组，以cv2格式保存了包含了检测框的图片。

#### 3. 结果输出

```python
format_result = det_coco.format_output(lang='zh')# 将推理结果进行格式化输出
```

![](../images/xeduhub/det_coco_format.png)

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(string) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_result`以字典形式存储了推理结果，共有三个键：`检测框`、`分数`和`类别`。检测框以二维数组形式保存了每个检测框的坐标信息[x1,y1,x2,y2]，而分数则是对应下标的检测框的置信度，以一维数组形式保存，类别则是检测框中对象所属的类别，以一维数组形式保存。

```python
det_coco.show(img_with_box)# 展示推理图片
```

`show()`能够输出带有检测框以及对应类别的结果图像。

![](../images/xeduhub/det_coco_show.png)

#### 4. 结果保存

```python
det_coco.save(img_with_box,'img_with_box.jpg')# 保存推理图片
```

`save()`方法能够保存带有检测框以及对应类别的图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

### 人体检测

人体目标检测的任务是在图像或视频中检测和定位人体的位置，并为每个检测到的人体分配一个相应的类别标签。

XEduHub提供了进行人体目标检测的模型：`det_body`，`det_body_l`，这两个模型能够进行单人的人体目标检测。

![](../images/xeduhub/det_body.png)

#### 代码样例

```python
from XEdu.hub import Workflow as wf
det_body = wf(task='det_body')
result,img_with_box = det_body.inference(data='data/det_body.jpg',img_type='pil') # 进行模型推理
format_result = det_body.format_output(lang='zh')# 将推理结果进行格式化输出
det_body.show(img_with_box)# 展示推理图片
det_body.save(img_with_box,'img_with_box.jpg')# 保存推理图片
```

#### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
det_body = wf(task='det_body')
```

`wf()`中共有三个参数可以设置：

- `task`选择任务。人体目标检测模型为`det_body`, `det_body_l`。`det_body_l`相比`det_body`模型规模较大，性能较强，但是推理的速度较慢。
- `checkpoints`指定模型的路径，默认在本地同级的checkpoints文件夹中寻找任务对应的模型，如`checkpoints='my_checkpoint/model.onnx'`。
- `download_path`指定模型的下载路径。默认是下载到同级的checkpoints文件夹中，如`download_path='my_checkpoint'`。

任务模型下载与存放请查看<a href="https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id122">XEduHub任务模型资源下载与存放</a>。

#### 2. 模型推理

```python
result,img_with_box = det_body.inference(data='data/det_body.jpg',img_type='pil') # 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`(string|numpy.ndarray): 指定待目标检测的图片。

- `show`(flag): 可取值`[True,False]` ,如果取值为`True`，在推理完成后会直接输出目标检测完成后的图片，默认为`False`。

- `img_type`(string): 关键点识别完成后会返回含有目标检测框的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，即如果不传入值，则不会返回图。

- `thr`(float): 设置检测框阈值，取值范围为`[0,1]`超过该阈值的检测框被视为有效检测框，进行显示。

- `bbox`(List|numpy.ndarray)：该参数指定了要识别某检测框中的目标，如输入bbox=[x0,y0,w0,h0]。

模型推理返回结果：

- `result`：变量`boxes`以二维数组的形式保存了检测框左上角顶点的坐标(x1,y1)和右下角顶点的坐标(x2,y2)（之所以是二维数组，是因为该模型能够检测多个人体，因此当检测到多个人体时，就会有多个[x1,y1,x2,y2]的一维数组，所以需要以二维数组形式保存），我们可以利用这四个数据计算出其他两个顶点的坐标，以及检测框的宽度和高度。

- `img_with_box：`：是个三维数组，以cv2格式保存了包含了检测框的图片。

#### 3. 结果输出

```python
format_result = det_body.format_output(lang='zh')# 将推理结果进行格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(string) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_result`以字典形式存储了推理结果，共有两个键：`检测框`、`分数`。检测框以二维数组形式保存了每个检测框的坐标信息[x1,y1,x2,y2]，而分数则是对应下标的检测框的置信度，以一维数组形式保存。

![](../images/xeduhub/det_body_format.png)

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

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

### 脸部检测

人脸检测指的是检测和定位一张图片中的人脸。XEduHub使用的是opencv的人脸检测模型，能够快速准确地检测出一张图片中所有的人脸。EduHub提供了进行人体目标检测的模型：`det_face`，能够快速准确地检测出图片中的所有人脸。

#### 代码样例

```python
from XEdu.hub import Workflow as wf
det_face = wf(task='det_face')
result,img_with_box = det_face.inference(data='data/det_face.jpg',img_type='pil') # 进行模型推理
format_result = det_face.format_output(lang='zh')# 将推理结果进行格式化输出
det_face.show(img_with_box)# 展示推理图片
det_face.save(img_with_box,'img_with_box.jpg')# 保存推理图片
```

#### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
det_face = wf(task='det_face')
```
`wf()`中共有三个参数可以设置：

- `task`选择任务。人脸目标检测模型为`det_face`。
- `checkpoints`指定模型的路径，默认在本地同级的checkpoints文件夹中寻找任务对应的模型，如`checkpoints='my_checkpoint/model.onnx'`。
- `download_path`指定模型的下载路径。默认是下载到同级的checkpoints文件夹中，如`download_path='my_checkpoint'`。

任务模型下载与存放请查看<a href="https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id122">XEduHub任务模型资源下载与存放</a>。

#### 2. 模型推理

```python
result,img_with_box = det_face.inference(data='data/det_face.jpg',img_type='pil') # 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`(string|numpy.ndarray): 指定待目标检测的图片。

- `show`(flag): 可取值`[True,False]` ,如果取值为`True`，在推理完成后会直接输出目标检测完成后的图片，默认为`False`。

- `img_type`(string): 关键点识别完成后会返回含有目标检测框的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，即如果不传入值，则不会返回图。

- `thr`(float): 设置检测框阈值，取值范围为`[0,1]`超过该阈值的检测框被视为有效检测框，进行显示。

- `bbox`(List|numpy.ndarray)：该参数指定了要识别某检测框中的目标，如输入bbox=[x0,y0,w0,h0]。

- `minSize`(tuple(int,int))：检测框的最小尺寸，小于该尺寸的目标会被过滤掉，默认为(50,50)。

- `maxSize`(tuple(int,int))：检测框的最大尺寸,大于该尺寸的目标会被过滤掉，默认为输入图像的大小。

- `scaleFactor`(float)：该参数用于缩放图像，以便在检测过程中使用不同大小的窗口来识别人脸。较小的值会导致检测速度加快，但可能会错过一些小的人脸；较大的值可以提高检测的准确性，但会减慢检测速度。通常，这个值会在1.1到1.5之间进行调整，默认为1.1。

- `minNeighbors`(int)：该参数定义了构成检测目标的最小邻域矩形个数。如果这个值设置得太高，可能会导致检测器过于严格，错过一些实际的人脸；如果设置得太低，检测器可能会变得过于宽松，错误地检测到非人脸区域。通常，这个值会在2到10之间进行调整，默认为5。

模型推理返回结果：

- `result`：以二维数组的形式保存了检测框左上角顶点的坐标(x1,y1)和右下角顶点的坐标(x2,y2)（之所以是二维数组，是因为该模型能够检测多个人脸，因此当检测到多个人脸时，就会有多个[x1,y1,x2,y2]的一维数组，所以需要以二维数组形式保存），我们可以利用这四个数据计算出其他两个顶点的坐标，以及检测框的宽度和高度。
- `img_with_box：`：是个三维数组，以cv2格式保存了包含了检测框的图片。

#### 3. 结果输出

```python
format_result = det_face.format_output(lang='zh')# 将推理结果进行格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(string) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_result`以字典形式存储了推理结果，只有一个键：`检测框`、`分数`。检测框以二维数组形式保存了每个检测框的坐标信息[x1,y1,x2,y2]。需要注意的是由于使用的为opencv的人脸检测模型，因此在`format_output`时缺少了分数这一指标。

![](../images/xeduhub/det_face_format.png)

```python
det_face.show(img_with_box)# 展示推理图片
```

`show()`能够输出带有检测框的结果图像。

![](../images/xeduhub/det_face_show.png)

#### 4. 结果保存

```python
det_face.save(img_with_box,'img_with_box.jpg')# 保存推理图片
```

`save()`方法能够保存带有检测框的图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

### 手部检测

手部检测指的是检测和定位一张图片中的人手。XEduHub提供了进行人体目标检测的模型：`det_hand`，能够快速准确地检测出图片中的所有人手。

#### 代码样例

```python
from XEdu.hub import Workflow as wf
det_hand = wf(task='det_hand')
result,img_with_box = det_body.inference(data='data/det_hand.jpg',img_type='pil',show=True) # 进行模型推理
format_result = det_hand.format_output(lang='zh')# 将推理结果进行格式化输出
det_hand.show(img_with_box)# 展示推理图片
det_hand.save(img_with_box,'img_with_box.jpg')# 保存推理图片
```

#### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
det_hand = wf(task='det_hand')
```

`wf()`中共有三个参数可以设置：

- `task`选择任务。全身关键点提取模型为`det_face`。
- `checkpoints`指定模型的路径，默认在本地同级的checkpoints文件夹中寻找任务对应的模型，如`checkpoints='my_checkpoint/model.onnx'`。
- `download_path`指定模型的下载路径。默认是下载到同级的checkpoints文件夹中，如`download_path='my_checkpoint'`。

任务模型下载与存放请查看<a href="https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id122">XEduHub任务模型资源下载与存放</a>。

#### 2. 模型推理

```python
result,img_with_box = det_hand.inference(data='data/det_hand.jpg',img_type='pil') # 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`：指定待检测的图片。
- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出目标检测完成后的图片。
- `img_type`：目标检测完成后会返回含有检测框的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。
- `thr`: 设置检测框阈值，取值范围为`[0,1]`超过该阈值的检测框被视为有效检测框，进行显示。

模型推理返回结果：

- `result`：以二维数组的形式保存了检测框左上角顶点的坐标(x1,y1)和右下角顶点的坐标(x2,y2)（之所以是二维数组，是因为该模型能够检测多个人手，因此当检测到多个人手时，就会有多个[x1,y1,x2,y2]的一维数组，所以需要以二维数组形式保存），我们可以利用这四个数据计算出其他两个顶点的坐标，以及检测框的宽度和高度。
- `img_with_box：`：是个三维数组，以cv2格式保存了包含了检测框的图片。

#### 3. 结果输出

```python
format_result = det_hand.format_output(lang='zh')# 将推理结果进行格式化输出
```

![](../images/xeduhub/det_hand_format.png)

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(string) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_result`以字典形式存储了推理结果，共有两个键：`检测框`、`分数`。检测框以二维数组形式保存了每个检测框的坐标信息[x1,y1,x2,y2]，而分数则是对应下标的检测框的置信度，以一维数组形式保存。

```python
det_hand.show(img_with_box)# 展示推理图片
```

`show()`能够输出带有检测框的结果图像。

![](../images/xeduhub/det_hand_show.png)

#### 4. 结果保存

```python
det_hand.save(img_with_box,'img_with_box.jpg')# 保存推理图片
```

`save()`方法能够保存带有检测框的图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

## 2. 关键点识别

关键点识别是深度学习中的一项关键任务，旨在检测图像或视频中的关键位置，通常代表物体或人体的重要部位。XEduHub支持的关键点识别任务有：人体关键点`pose_body`、人脸关键点`pose_face`、人手关键点`pose_hand`和所有人体关键点识别`pose_wholebody`。

**注意事项**：这里我们强烈建议提取关键点之前应**先进行目标检测**。

例如进行人体关键点检测`pose_body`之前，先使用`det_body`在图片中检测中人体目标，对每个人体目标进行更加精准的关键点检测。可参考项目<a href="https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&backpath=/pjlab/projects/list#public">XEduHub实例代码-入门完整版</a>中的 **“3-1 综合项目：目标检测+关键点检测”**。

当然关键点识别也可以单独用，但是效果并不保证。

![](../images/xeduhub/pose.png)

### 人体关键点识别 

人体关键点识别是一项计算机视觉任务，旨在检测和定位图像或视频中人体的关键位置，通常是关节、身体部位或特定的解剖结构。

这些关键点的检测可以用于人体姿态估计和分类、动作分析、手势识别等多种应用。

XEduHub提供了三个识别人体关键点的优质模型:`pose_body17`,`pose_body17_l`和`pose_body26`，能够在使用cpu推理的情况下，快速识别出身体的关键点。

 数字表示了识别出人体关键点的数量，l代表了large，表示规模较大的，性能较强的模型，但是缺点在于推理速度较慢。

`pose_body17`与`pose_body17_l`模型能识别出17个人体骨骼关键点，`pose_body26`模型能识别出26个人体骨骼关键点。

![](../images/xeduhub/body.png)

#### 代码样例

```python
from XEdu.hub import Workflow as wf
body = wf(task='pose_body') # 数字可省略，当省略时，默认为pose_body17
keypoints,img_with_keypoints = body.inference(data='data/body.jpg',img_type='pil') # 进行模型推理
format_result = body.format_output(lang='zh')# 将推理结果进行格式化输出
body.show(img_with_keypoints)# 展示推理图片
body.save(img_with_keypoints,'img_with_keypoints.jpg')# 保存推理图片
```

#### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
body = wf(task='pose_body') # 数字可省略，当省略时，默认为pose_body17
```

`wf()`中共有三个参数可以设置：

- `task`选择任务。在人体关键点识别模型中，`task`可选取值为：`[pose_body17,pose_body17_l,pose_body26]`。
- `checkpoints`指定模型的路径，默认在本地同级的checkpoints文件夹中寻找任务对应的模型，如`checkpoints='my_checkpoint/model.onnx'`。
- `download_path`指定模型的下载路径。默认是下载到同级的checkpoints文件夹中，如`download_path='my_checkpoint'`。

任务模型下载与存放请查看<a href="https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id122">XEduHub任务模型资源下载与存放</a>。

#### 2. 模型推理

```python
keypoints,img_with_keypoints = body.inference(data='data/body.jpg',img_type='pil') # 进行模型推理
```


模型推理`inference()`可传入参数：

- `data`: 指定待识别关键点的图片，可以是以图片路径形式传入，也可直接传入cv2或pil格式的图片。

- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出关键点识别完成后的图片。

- `img_type`: 关键点识别完成后会返回含有关键点的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。
- `bbox`：该参数可配合目标检测使用。在多人手关键点检测中，该参数指定了要识别哪个检测框中的关键点。

模型推理返回结果：

- `keypoints`以二维数组的形式保存了所有关键点的坐标，每个关键点(x,y)被表示为`[x,y]`根据前面的图示，要获取到某个特定序号`i`的关键点，只需要访问`keypoints[i]`即可。

- `img_with_keypoints`是个三维数组，以对应img_type格式保存了关键点识别完成后图片的像素点信息。

#### 3. 结果输出

```python
format_result = body.format_output(lang='zh')# 参数lang设置了输出结果的语言，默认为中文
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(string) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_result`以字典形式存储了推理结果，共有两个键：`关键点坐标`和`分数`。关键点坐标以二维数组形式保存了每个关键点的[x,y]坐标，而分数则是对应下标的关键点的分数，以一维数组形式保存。

![](../images/xeduhub/body-format.png)

结果可视化

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
format_result = face.format_output(lang='zh')# 将推理结果进行格式化输出
face.show(img_with_keypoints)# 展示推理图片
face.save(img_with_keypoints,'img_with_keypoints.jpg')# 保存推理图片
```

#### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
face = wf(task='pose_face') # 数字可省略，默认为face106
```

`wf()`中共有三个参数可以设置：

- `task`选择任务。人脸关键点识别模型为`pose_face106`（数字可省略，默认为face106）。
- `checkpoints`指定模型的路径，默认在本地同级的checkpoints文件夹中寻找任务对应的模型，如`checkpoints='my_checkpoint/model.onnx'`。
- `download_path`指定模型的下载路径。默认是下载到同级的checkpoints文件夹中，如`download_path='my_checkpoint'`。

任务模型下载与存放请查看<a href="https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id122">XEduHub任务模型资源下载与存放</a>。
#### 2. 模型推理

```python
keypoints,img_with_keypoints = face.inference(data='data/face.jpg',img_type='pil') # 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`: 指定待识别关键点的图片，可以是以图片路径形式传入，也可直接传入cv2或pil格式的图片。

- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出关键点识别完成后的图片。

- `img_type`: 关键点识别完成后会返回含有关键点的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。

- `bbox`：该参数可配合目标检测使用。在多人手关键点检测中，该参数指定了要识别哪个检测框中的关键点。

模型推理返回结果：

- `keypoints`以二维数组的形式保存了所有关键点的坐标，每个关键点(x,y)被表示为`[x,y]`根据前面的图示，要获取到某个特定序号`i`的关键点，只需要访问`keypoints[i]`即可。

- `img_with_keypoints`是个三维数组，以对应img_type格式保存了关键点识别完成后图片的像素点信息。

#### 3. 结果输出

```python
format_result = face.format_output(lang='zh')# 将推理结果进行格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(string) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_result`以字典形式存储了推理结果，共有两个键：`关键点坐标`和`分数`。关键点坐标以二维数组形式保存了每个关键点的[x,y]坐标，而分数则是对应下标的关键点的分数，以一维数组形式保存。

结果可视化

```python
face.show(img_with_keypoints)# 展示推理图片
```

`show()`能够输出带有关键点的结果图像。

![](../images/xeduhub/face_show.png)

**若此时发现关键点识别效果不佳**，关键点乱飞，我们可以果断采用在提取关键点之前**先进行目标检测**的方式。如当前任务`'pose_face'`，就可以在之前先进行`'det_face'`。详情可参考项目<a href="https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&backpath=/pjlab/projects/list#public">XEduHub实例代码-入门完整版</a>中的 **“3-1 综合项目：目标检测+关键点检测”**。

#### 4. 结果保存

```python
face.save(img_with_keypoints,'img_with_keypoints.jpg')# 保存推理图片
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
format_result = hand.format_output(lang='zh')# 将推理结果进行格式化输出
hand.show(img_with_keypoints)# 展示推理图片
hand.save(img_with_keypoints,'img_with_keypoints.jpg')# 保存推理图片
```

#### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
hand = wf(task='pose_hand') # 数字可省略，当省略时，默认为pose_hand21
```

`wf()`中共有三个参数可以设置：

- `task`选择任务。人手关键点识别模型为`pose_hand`。
- `checkpoints`指定模型的路径，默认在本地同级的checkpoints文件夹中寻找任务对应的模型，如`checkpoints='my_checkpoint/model.onnx'`。
- `download_path`指定模型的下载路径。默认是下载到同级的checkpoints文件夹中，如`download_path='my_checkpoint'`。

任务模型下载与存放请查看<a href="https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id122">XEduHub任务模型资源下载与存放</a>。

#### 2. 模型推理

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

- `img_with_keypoints`是个三维数组，以pil格式保存了关键点识别完成后的图片。

#### 3. 结果输出

```python
format_result = hand.format_output(lang='zh')# 将推理结果进行格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(string) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。
`format_result`以字典形式存储了推理结果，共有两个键：`关键点坐标`和`分数`。关键点坐标以二维数组形式保存了每个关键点的[x,y]坐标，而分数则是对应下标的关键点的分数，以一维数组形式保存。

```python
hand.show(img_with_keypoints)# 展示推理图片
```

`show()`能够输出带有关键点的结果图像。

![](../images/xeduhub/hand_show.png)

**若此时发现关键点识别效果不佳**，关键点乱飞，我们可以果断采用在提取关键点之前**先进行目标检测**的方式。如当前任务`'pose_hand'`，就可以在之前先进行`'det_hand'`。详情可参考项目<a href="https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&backpath=/pjlab/projects/list#public">XEduHub实例代码-入门完整版</a>中的 **“3-1 综合项目：目标检测+关键点检测”**。

#### 4. 结果保存

```python
hand.save(img_with_keypoints,'img_with_keypoints.jpg')# 保存推理图片
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
format_result = wholebody.format_output(lang='zh')# 将推理结果进行格式化输出
wholebody.show(img_with_keypoints)# 展示推理图片
wholebody.save(img_with_keypoints,'img_with_keypoints.jpg')# 保存推理图片
```

#### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
wholebody = wf(task='pose_wholebody') # 数字可省略，当省略时，默认为pose_wholebody133
```

`wf()`中共有三个参数可以设置：

- `task`选择任务。全身关键点提取模型为`pose_wholebody`。
- `checkpoints`指定模型的路径，默认在本地同级的checkpoints文件夹中寻找任务对应的模型，如`checkpoints='my_checkpoint/model.onnx'`。
- `download_path`指定模型的下载路径。默认是下载到同级的checkpoints文件夹中，如`download_path='my_checkpoint'`。

任务模型下载与存放请查看<a href="https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id122">XEduHub任务模型资源下载与存放</a>。

#### 2. 模型推理

```python
keypoints,img_with_keypoints = wholebody.inference(data='data/wholebody.jpg',img_type='pil') # 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`: 指定待识别关键点的图片，可以是以图片路径形式传入，也可直接传入cv2或pil格式的图片。

- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出关键点识别完成后的图片。

- `img_type`: 关键点识别完成后会返回含有关键点的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。
- `bbox`：该参数可配合目标检测使用。在多人手关键点检测中，该参数指定了要识别哪个检测框中的关键点。

模型推理返回结果：

- `keypoints`以二维数组的形式保存了所有关键点的坐标，每个关键点(x,y)被表示为`[x,y]`根据前面的图示，要获取到某个特定序号`i`的关键点，只需要访问`keypoints[i]`即可。

- `img_with_keypoints`是个三维数组，以pil格式保存了关键点识别完成后的图片。

#### 3. 结果输出

```python
format_result = wholebody.format_output(lang='zh')# 将推理结果进行格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(string) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_result`以字典形式存储了推理结果，共有两个键：`关键点坐标`和`分数`。关键点坐标以二维数组形式保存了每个关键点的[x,y]坐标，而分数则是对应下标的关键点的分数，以一维数组形式保存。

```python
wholebody.show(img_with_keypoints)# 展示推理图片
```

`show()`能够输出带有关键点的结果图像。

![](../images/xeduhub/wholebody_show.png)

**若此时发现关键点识别效果不佳**，关键点乱飞，我们可以果断采用在提取关键点之前**先进行目标检测**的方式。如当前任务`'pose_wholebody'`，就可以在之前先进行`'det_body'`。详情可参考项目<a href="https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&backpath=/pjlab/projects/list#public">XEduHub实例代码-入门完整版</a>中的 **“3-1 综合项目：目标检测+关键点检测”**。

#### 4. 结果保存

```python
wholebody.save(img_with_keypoints,'img_with_keypoints.jpg')# 保存推理图片
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
ocr = wf(task="ocr")
result,ocr_img = ocr.inference(data='data/ocr_img.png',img_type='cv2') # 进行模型推理
ocr_format_result = ocr.format_output(lang="zh")# 推理结果格式化输出
ocr.show(ocr_img)# 展示推理结果图片
ocr.save(ocr_img,'ocr_result.jpg')# 保存推理结果图片
```

### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
ocr = wf(task="ocr")
```

`wf()`中只有参数可以设置：

- `task`选择任务类型，光学字符识别（OCR）的模型为`ocr`。
  

**注意**：ocr的模型不是以onnx方式下载，而是以python库的形式下载和安装，因此不同于之前任务的下载方式，也无需指定下载路径。可以通过`pip install rapidocr_onnxruntime==1.3.7`预先下载库。

#### 2. 模型推理

```python
result,ocr_img = ocr.inference(data='data/ocr_img.png',img_type='cv2') # 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`: 指定待进行ocr的图片，可以是以图片路径形式传入，也可直接传入cv2或pil格式的图片。
- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出OCR完成后的图片。
- `img_type`：目标检测完成后会返回含有检测框的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`

`result`以一维数组的形式保存了识别出的文本及其检测框的四个顶点(x,y)坐标.

如图所示，数组中每个元素的形式为元组：（识别文本，检测框顶点坐标）。四个顶点坐标顺序分别为[左上，右上，左下，右下]。

![](../images/xeduhub/ocr_res.png)

`ocr_img`的格式为cv2，保存了ocr识别后的结果图片。

#### 3. 结果输出

```python
ocr_format_result = ocr.format_output(lang="zh")
```

![](../images/xeduhub/ocr_format.png)

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(string) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_output`的结果以字典形式存储了推理结果，共有三个键：`检测框`、`分数`和`文本`。检测框以三维数组形式保存了每个检测框的四个顶点的[x,y]坐标，而分数则是对应下标的检测框分数，以一维数组形式保存。文本则是每个检测框中识别出的文本，以一维数组形式保存。

```python
ocr.show(ocr_img)# 展示推理结果图片
```

显示结果图片：由两部分组成，左侧为原图片，右侧为经过ocr识别出的文本，并且该文本的位置与原图片中文本的位置保持对应。

![](../images/xeduhub/ocr_show.png)

#### 4. 结果保存

```python
ocr.save(ocr_img,'ocr_result.jpg')# 保存推理结果图片
```

`save()`方法能够保存ocr识别后的结果图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

## 4. 图像分类

图像分类是一个分类任务，它能够将不同的图像划分到指定的类别中，实现最小的分类误差和最高精度。XEduHub提供了进行图像分类的模型：`cls_imagenet`，该模型的分类类别取自ImageNet的一千个分类，这意味着该模型能够将输入的图像划分到这一千个分类中的类别上。

### 代码样例

```python
from XEdu.hub import Workflow as wf
cls = wf(task='cls_imagenet') # 模型声明
result = cls.inference(data='data/cat101.jpg')# 进行模型推理
format_result = cls.format_output(lang='zh')#推理结果格式化输出
```

### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
cls = wf(task="cls_imagenet") # 模型声明
```


`wf()`中共有三个参数可以设置：

- `task`选择任务。图像分类的模型为`cls_imagenet`。
- `checkpoints`指定模型的路径，默认在本地同级的checkpoints文件夹中寻找任务对应的模型，如`checkpoints='my_checkpoint/model.onnx'`。
- `download_path`指定模型的下载路径。默认是下载到同级的checkpoints文件夹中，如`download_path='my_checkpoint'`。

任务模型下载与存放请查看<a href="https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id122">XEduHub任务模型资源下载与存放</a>。

#### 2. 模型推理

```python
result = cls.inference(data='data/cat101.jpg')# 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`: 指定待分类的图片，可以是以图片路径形式传入，也可直接传入cv2或pil格式的图片。

推理结果`result`是一个二维数组，表示这个图片在ImageNet的一千个分类中，属于每个分类的概率。

![](../images/xeduhub/cls_result.png)

#### 3. 结果输出

```python
format_result = cls.format_output(lang='zh')#推理结果格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(string) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_result`是一个字典，以格式化的方式展示了这张图片最有可能的分类结果。预测值表示图片分类标签在所有一千个分类中的索引，分数是属于这个分类的概率，预测类别是分类标签的内容。

![](../images/xeduhub/cls_format.png)

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
result, img = style.inference(data='data/cat101.jpg',img_type='cv2')# 进行模型推理
style.show(img)# 展示推理图片
style.save(img,"style_cat.jpg")# 保存推理图片
```

### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
style = wf(task='gen_style',style='mosaic')
```

`wf()`中共有四个参数可以设置：

- `task`选择任务。图像分类的模型为`cls_imagenet`。
- `style`选择风格迁移所使用的风格。
- `checkpoints`指定模型的路径，默认在本地同级的checkpoints文件夹中寻找任务对应的模型，如`checkpoints='my_checkpoint/model.onnx'`。
- `download_path`指定模型的下载路径。默认是下载到同级的checkpoints文件夹中，如`download_path='my_checkpoint'`。

任务模型下载与存放请查看<a href="https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id122">XEduHub任务模型资源下载与存放</a>。

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
result, img = style.inference(data='data/cat101.jpg',img_type='cv2')# 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`: 待进行风格迁移的图片，可以是以图片路径形式传入，也可直接传入cv2或pil格式的图片。
- `show`: 可取值：`[true,false]` 默认为`false`。如果取值为`true`，在推理完成后会直接输出风格迁移完成后的图片。
- `img_type`: 推理完成后会直接输出风格迁移完成后的图片。该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。

模型推理返回结果：

- `result`和`img`都是三维数组，以cv2格式保存了风格迁移完成后的图片。

#### 3. 结果输出

```python
style.show(img)# 展示推理后的图片
```

`show()`能够输出风格迁移后的结果图像。

![](../images/xeduhub/style_show.png)

#### 4. 结果保存

```python
style.save(img,"style_cat.jpg")# 保存推理图片
```

`save()`方法能够保存风格迁移后的图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

### 实例讲解2：自定义风格迁移模型的使用

当我们看到喜欢的风格的图像，并想要迁移到其他图像上时，我们就可以使用XEduHub中的自定义风格迁移模型

例如我喜欢“my_style”这张图片，我想要将其风格迁移到我的风景照上，生成新的图像

**将图片的路径来自定义风格，style='demo/my_style.jpg'**

下面是实例自定义风格迁移模型的完整代码：

```python
from XEdu.hub import Workflow as wf # 导入库
style = wf(task='gen_style',style='demo/my_style.jpg') # 实例化模型
img_path = 'demo/ShangHai.jpg'  # 指定进行推理的图片路径
result, new_img = style.inference(data=img_path,img_type='cv2')# 进行模型推理
style.show(new_img) # 可视化结果
style.save(new_img, "demo/style_my_style_ShangHai.jpg") # 保存可视化结果
```

### 2. 图像着色模型的使用

图像着色模型是将灰度图像转换为彩色图像的模型，它根据图像的内容、场景和上下文等信息来推断合理的颜色分布，实现从灰度到彩色的映射。

当我们有一张黑白图片想要为它上色时，可以使用XEduHub提供的gen_color图像着色任务。通过调用基于卷积神经网络 (CNN)训练的模型进行推理，自动地为黑白图像添加颜色，实现了快速生成逼真的着色效果。

### 代码样例

```python
from XEdu.hub import Workflow as wf # 导入库
color = wf(task='gen_color') # 实例化模型
result, img = style.inference(data='demo/gray_img1.jpg',img_type='cv2')# 进行模型推
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
- `checkpoints`指定模型的路径，默认在本地同级的checkpoints文件夹中寻找任务对应的模型，如`checkpoints='my_checkpoint/model.onnx'`。
- `download_path`指定模型的下载路径。默认是下载到同级的checkpoints文件夹中，如`download_path='my_checkpoint'`。

任务模型下载与存放请查看<a href="https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id122">XEduHub任务模型资源下载与存放</a>。


#### 2. 模型推理

```python
result, img = style.inference(data='demo/gray_img1.jpg',img_type='cv2')# 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`: 待进行风格迁移的图片，可以是以图片路径形式传入，也可直接传入cv2或pil格式的图片。
- `show`: 可取值：`[true,false]` 默认为`false`。如果取值为`true`，在推理完成后会直接输出风格迁移完成后的图片。
- `img_type`: 推理完成后会直接输出图像着色完成后的图片。该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。

模型推理返回结果：

- `result`和`img`都是三维数组，以cv2格式保存了风格迁移完成后的图片。

#### 3. 结果输出

```python
style.show(img)# 展示推理后的图片
```

`show()`能够输出着色后的结果图像。

![](../images/xeduhub/color_show.png)

#### 4. 结果保存

```python
style.save(img,"color_img.jpg")# 保存推理图片
```

`save()`方法能够保存着色后的图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

## 6. 全景驾驶感知系统

全景驾驶感知系统是一种高效的多任务学习网络，“多任务”表示该模型可同时执行交通对象检测、可行驶道路区域分割和车道检测任务，能很好地帮助自动驾驶汽车通过摄像头全面了解周围环境。我们可以在实时自动驾驶的项目中组合运用不同的检测任务，来控制车辆的动作，以达到更好的效果。XEduHub提供了进行全景驾驶感知的任务：`drive_perception`。

### 代码样例

```python
from XEdu.hub import Workflow as wf
drive = wf(task='drive_perception') # 实例化模型
result,img = drive.inference(data="demo/drive.png",img_type='cv2') # 模型推理
drive.format_output(lang='zh') # 将推理结果进行格式化输出
drive.show(img) # 展示推理图片
drive.save(img,"img_perception.jpg") # 保存推理图片
```

### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
drive = wf(task='drive_perception') # 实例化模型
```

`wf()`中共有三个参数可以设置：

- `task`选择任务。全景驾驶感知系统的模型为`drive_perception`。
- `checkpoints`指定模型的路径，默认在本地同级的checkpoints文件夹中寻找任务对应的模型，如`checkpoints='my_checkpoint/model.onnx'`。
- `download_path`指定模型的下载路径。默认是下载到同级的checkpoints文件夹中，如`download_path='my_checkpoint'`。

任务模型下载与存放请查看<a href="https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#id122">XEduHub任务模型资源下载与存放</a>。

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

![](../images/xeduhub/drive_printresult.png)

- `result`：以三维数组的形式保存了车辆检测（红色框），车道线分割（蓝色色块），可行驶区域（绿色色块）。
- `车辆检测result[0]`：以二维数组保存了车辆目标检测框左上角顶点的坐标(x1,y1)和右下角顶点的坐标(x2,y2)（之所以是二维数组，是因为该模型能够检测多辆车，因此当检测到多辆车时，就会有多个[x1,y1,x2,y2]的一维数组，所以需要以二维数组形式保存），我们可以利用这四个数据计算出其他两个顶点的坐标，以及检测框的宽度和高度。
- `车道线分割result[1]`：以由0，1组成二维数组（w*h），保存图像中每个像素的mask，mask为1表示该像素为车道线分割目标，mask为0表示该像素是背景。
- `可行驶区域result[2]`：以由0，1组成二维数组（w*h），保存图像中每个像素的mask，mask为1表示该像素为可驾驶区域分割目标，mask为0表示该像素是背景。
- `img_with_box：`：是个三维数组，以cv2格式保存了包含了检测框与分割目标的图片。

#### 3. 结果输出

```python
format_result = drive.format_output(lang='zh')# 将推理结果进行格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(string) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_result`以字典形式存储了推理结果，有四个键：`检测框`、`分数`、`车道线掩码`、`可行驶区域掩码`。检测框以二维数组形式保存了每个检测框的坐标信息[x1,y1,x2,y2]。

![](../images/xeduhub/drive_format.png)

```python
drive.show(img)# 展示推理图片
```

`show()`能够输出带有检测框与分割目标的图片。

![](../images/xeduhub/drive_show.png)

#### 4. 结果保存

```python
drive.save(img,"img_perception.jpg") # 保存推理图片
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

#### 2. 模型推理

```python
image_embeddings = img_emb.inference(data='demo/cat.png') # 模型推理
```

模型推理`inference()`可传入参数：

- `data`：指定待特征提取的图片。可以直接传入图像路径`data='cat.jpg'` 或者多张图像路径列表`data= ['cat.jpg','dog.jpg'] `。


模型推理返回结果：

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

#### 2. 模型推理

```python
txt_embeddings = txt_emb.inference(data=['a black cat','a yellow cat']) # 模型推理
```

模型推理`inference()`可传入参数：

- `data`：指定待特征提取的文本。可以直接传入文本`data= 'cat' `或者多条文本列表`data= ['a black cat','a yellow cat']`。


模型推理返回结果：

![](../images/xeduhub/emb_show2.png)

- `result`：以二维数组的形式保存了每条文本特征提取后的512维向量。

### 提完了特征能干啥？

零样本分类！

什么是零样本分类呢？举个例子，现在我们想要分类图片中的猫是黑色的还是黄色的，按照图像分类的方式，我们需要收集数据集，并且标注数据集，再进行模型训练，最后才能使用训练出来的模型对图像进行分类。而现在，我们使用的“图像特征提取”和“文本特征提取”只需通过特征向量就可以进行分类，避免了大量的标注工作。

上文中我们已经通过图像特征提取和文本特征提取把`cat.jpg`,`'a black cat'`,`'a yellow cat'`分别变成了3堆数字（3个512维向量），但是很显然，我们看不懂这些数字，但是计算机可以！
通过让计算机将数字进行运算，即将图像和文本的特征向量作比较，就能看出很多信息，这也叫计算向量之间相似度。

为了方便大家计算向量之间的相似度，我们也提供了一系列数据处理函数，函数具体内容请见<a href="https://xedu.readthedocs.io/zh/master/about/functions.html#">XEdu的常见函数</a>。

下面就示范使用<a href="https://xedu.readthedocs.io/zh/master/about/functions.html#cosine-similarity">cosine_similarity</a>比较两个embedding序列的相似度。

```python
from XEdu.utils import get_similarity # 导入库
get_similarity(image_embeddings, txt_embeddings,method='cosine') # 计算相似度
```

该函数可以比较两个embedding序列的相似度，这里的相似度是以余弦相似度为计算指标的，其公式为：$$Cosine(x,y) = \frac{x \cdot y}{|x||y|}$$。

假设输入的待比较embedding序列尺度分别为(N, D)和(M, D)，则输出的结果尺度为(N, M)。

现在我们可以看到cat.jpg与'a black cat'向量的相似度为0.007789988070726395，而与'a yellow cat'向量的相似度为0.9922100305557251。显而易见，这张可爱的黄色猫咪图像与'a yellow cat'文本描述更为贴近。


## 8. MMEdu模型推理

XEduHub现在可以支持使用MMEdu导出的onnx模型进行推理啦！如果你想了解如何使用MMEdu训练模型，可以看这里：<a href="https://xedu.readthedocs.io/zh/master/mmedu/mmclassification.html">解锁图像分类模块：MMClassification</a>、<a href="https://xedu.readthedocs.io/zh/master/mmedu/mmdetection.html">揭秘目标检测模块：MMDetection</a>。

如果你想了解如何将使用[MMEdu](https://xedu.readthedocs.io/zh/master/mmedu.html)训练好的模型转换成ONNX格式，可以前往[最后一步：模型转换](https://xedu.readthedocs.io/zh/master/mmedu/mmedumodel_convert.html)。OK，准备好了ONNX模型，那么就开始使用XEduHub吧！

### MMClassification模型

#### 代码样例

```python
from XEdu.hub import Workflow as wf
mmcls = wf(task='mmedu',checkpoint='cats_dogs.onnx')# 指定使用的onnx模型
result, result_img =  mmcls.inference(data='data/cat101.jpg',img_type='pil')# 进行模型推理
format_result = mmcls.format_output(lang="zh")# 推理结果格式化输出
mmcls.show(result_img)# 展示推理结果图片
mmcls.save(result_img,'new_cat.jpg')# 保存推理结果图片
```

#### 代码解释

##### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
mmcls = wf(task='mmedu',checkpoint='cats_dogs.onnx')# 指定使用的onnx模型
```

`wf()`中共有两个参数可以设置

- `task`：只需要设置task为`mmedu` ，而不需要指定是哪种任务
- `checkpoint`：指定你的模型的路径

这里我们以猫狗分类模型为例，项目指路：[猫狗分类](https://www.openinnolab.org.cn/pjlab/project?id=63c756ad2cf359369451a617&sc=647b3880aac6f67c822a04f5#public)。

##### 2. 模型推理

```python
result, result_img =  mmcls.inference(data='data/cat101.jpg',img_type='pil',show=True)# 进行模型推理
```

![](../images/xeduhub/mmcls_result.png)

模型推理`inference()`可传入参数：

- `data`：指定待检测的图片。
- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出目标检测完成后的图片。
- `img_type`：分类完成后会返回含有分类标签的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。

`result`是一个字典，包含三个键：`标签`、`置信度`和`预测结果`。显然，这张图片为猫的置信度接近100%，自然这张图片被分类为猫。

![](../images/xeduhub/mmcls_res.png)

`result_img`以pil格式保存了模型推理完成后的图片

##### 3. 结果输出

```python
format_result = mmcls.format_output(lang="zh")# 推理结果格式化输出
```

![](../images/xeduhub/mmcls_format.png)

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(string) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_result`以字典形式保存了模型的推理结果，包括所属`标签`、`置信度`、以及`预测结果`。

```python
mmcls.show(result_img)# 展示推理结果图片
```

`show()`能够推理后的结果图像。与原图相比，结果图片在左上角多了`pred_label`, `pred_socre`和`pred_class`三个数据，对应着标签、置信度和预测结果。

![](../images/xeduhub/mmcls_show.png)

##### 4. 结果保存

```python
mmcls.save(img,'new_cat.jpg')# 保存推理结果图片
```

`save()`方法能够保存推理后的结果图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

### MMDetection模型

#### 代码样例

```python
from XEdu.hub import Workflow as wf
mmdet = wf(task='mmedu',checkpoint='plate.onnx')# 指定使用的onnx模型
result, result_img =  mmdet.inference(data='data/plate0.png',img_type='pil')# 进行模型推理
format_result = mmdet.format_output(lang="zh")# 推理结果格式化输出
mmdet.show(result_img)# 展示推理结果图片
mmdet.save(result_img,'new_plate.jpg')# 保存推理结果图片
```

#### 代码解释

##### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
mmdet = wf(task='mmedu',checkpoint='plate.onnx')# 指定使用的onnx模型
```

`wf()`中共有两个参数可以设置

- `task`：只需要设置task为`mmedu` ，而不需要指定是哪种任务
- `checkpoint`：指定你的模型的路径

这里以车牌识别为例进行说明。项目指路：[使用MMEdu实现车牌检测](https://www.openinnolab.org.cn/pjlab/project?id=641426fdcb63f030544017a2&backpath=/pjlab/projects/list#public)

##### 2. 模型推理

```python
result, result_img =  mmdet.inference(data='data/plate0.png',img_type='pil',show=True)# 进行模型推理
```

![](../images/xeduhub/mmdet_result.png)

模型推理`inference()`可传入参数：

- `data`：指定待检测的图片。
- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出目标检测完成后的图片。
- `img_type`：目标检测完成后会返回含有检测框的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。

`result`的结果是一个数组，里面保存了结果字典。该字典有四个键：`标签`、`置信度`、`坐标`以及`预测结果`。其中坐标表示了检测框的两个顶点：左上(x1,y1)和右下(x2,y2)。

![](../images/xeduhub/mmdet_res.png)

`result_img`以pil格式保存了模型推理完成后的图片

##### 3. 结果输出

```python
format_result = mmdet.format_output(lang="zh")# 推理结果格式化输出
```

![](../images/xeduhub/mmdet_format.png)

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(string) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_output`的结果是一个数组，里面保存了结果字典。该字典有四个键：`标签`、`置信度`、`坐标`以及`预测结果`。其中坐标表示了检测框的两个顶点：左上(x1,y1)和右下(x2,y2)。

```python
mmdet.show(result_img)# 展示推理结果图片
```

`show()`能够推理后的结果图像。与原图相比，结果图片还包含车牌周围的检测框以及结果信息。

![](../images/xeduhub/mmdet_show.png)

##### 4. 结果保存

```python
mmdet.save(img,'new_plate.jpg')# 保存推理结果图片
```

`save()`方法能够保存推理后的结果图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

## 9. BaseNN模型推理

XEduHub现在可以支持使用BaseNN导出的onnx模型进行推理啦！如果你想了解如何将使用[BaseNN](https://xedu.readthedocs.io/zh/master/basenn.html)训练好的模型转换成ONNX格式，可以看这里：[BaseNN模型文件格式转换](https://xedu.readthedocs.io/zh/master/basenn/introduction.html#id24)。OK，准备好了ONNX模型，那么就开始使用XEduHub吧！

#### 代码样例

```python
# 使用BaseNN训练的手写数字识别模型进行推理
from XEdu.hub import Workflow as wf
basenn = wf(task="basenn",checkpoint="basenn.onnx")# 指定使用的onnx模型
result = base.inference(data='data/6.jpg')# 进行模型推理
format_result = basenn.format_output()
```

#### 代码解释

##### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
basenn = wf(task="basenn",checkpoint="basenn.onnx")# 指定使用的onnx模型
```

`wf()`中共有两个参数可以设置

- `task`：只需要设置task为`basenn` ，而不需要指定是哪种任务
- `checkpoint`：指定你的模型的路径

##### 2. 模型推理

```python
result = base.inference(data='data/6.jpg')# 进行模型推理
```

模型推理`inference()`可传入参数：

- `data`：指定待检测的图片。

`result`的结果是一个二维数组，第一个元素表示这张图属于0-9的数字类别的概率，可以看到当为6时，概率接近，因此该手写数字是6。

![](../images/xeduhub/basenn_res.png)

##### 3. 结果输出

```python
format_result = basenn.format_output()
```

![](../images/xeduhub/basenn_format.png)

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(string) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

`format_output`的结果是一个结果字典，这个字典的第一个元素有两个键，`预测值`、`分数`，代表着该手写数字的分类标签以及属于该分类标签的概率。

## 10. BaseML模型推理

XEduHub现在可以支持使用BaseML导出的pkl模型文件进行推理啦！如果你想了解如何将使用[BaseML](https://xedu.readthedocs.io/zh/master/baseml.html)训练模型并保存成.pkl模型文件，可以看这里：[BaseML模型保存](https://xedu.readthedocs.io/zh/master/baseml/introduction.html#id16)。OK，准备好了pkl模型，那么就开始使用XEduHub吧！

#### 代码样例

```python
# 使用BaseML训练的鸢尾花聚类模型推理
from XEdu.hub import Workflow as wf
baseml = wf(task='baseml',checkpoint='baseml.pkl')# 指定使用的pkl模型
data = [[5.1,1.5],[7,4.7]] # 该项目中训练数据只有两维，因此推理时给出两维数据
result= baseml.inference(data=data)# 进行模型推理
format_output = baseml.format_output(lang='zh')# 推理结果格式化输出
```

#### 代码解释

##### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
baseml = wf(task='baseml',checkpoint='baseml.pkl')# 指定使用的pkl模型
```

`wf()`中共有两个参数可以设置

- `task`：只需要设置task为`baseml` ，而不需要指定是哪种任务
- `checkpoint`：指定你的模型的路径

##### 2. 模型推理

```python
data = [[5.1,1.5],[7,4.7]] # 该项目中训练数据只有两维，因此推理时给出两维数据
result= baseml.inference(data=data)# 进行模型推理
```

`mmdet.inference`可传入参数：

- `data`：指定待推理数据（数据类型和格式跟模型训练有关）。

**注意！**基于BaseML模型推理结果不包含图片！因为大部分使用BaseML解决的任务只需要输出分类标签、文本或者数组数据等。

![](../images/xeduhub/baseml_result.png)

##### 3. 结果输出

```python
format_output = baseml.format_output(lang='zh')# 推理结果格式化输出
```

![](../images/xeduhub/baseml_format.png)

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。

`format_output()`中共有两个参数可以设置：

- `lang`(string) - 可选参数，设置了输出结果的语言，可选取值为：[`'zh'`,`'en'`,`'ru'`,`'de'`,`'fr'`]，分别为中文、英文、俄语、德语、法语，默认为中文。
- `isprint`(bool) - 可选参数，设置了是否格式化输出，可选取值为：[`True`,`False`]，默认为True。

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

`wf()`中共有两个参数可以设置

- `task`：只需要设置task为`custom` ，而不需要指定是哪种任务
- `checkpoint`：指定你的模型的路径

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
print(result)
```

![](../images/xeduhub/custom_result.png)

模型推理`inference()`可传入参数：

- `data`：指定待检测的图片
- `preprocess`: 指定前处理函数
- `postprocess`：指定后处理函数

## XEduHub任务模型文件获取与存放

XEduHub提供了大量优秀的任务模型，我们不仅可以通过`wf()`代码的运行实现模型的获取，还可以在网站上进行获取。

只要进入<a href="https://openxlab.org.cn/models/detail/xedu/hub-model">模型仓库</a>，在Model File里就可以看到各种任务模型。网址：<a href="https://openxlab.org.cn/models/detail/xedu/hub-model">https://openxlab.org.cn/models/detail/xedu/hub-model</a>

![](../images/xeduhub/downloadmodel.png)

没有网络，如何让代码`wf()`运行时找到找到模型文件呢？

在这里悄悄透露代码`wf()`运行时的工作流程，在没有指定模型路径`checkpoints`参数的情况下，会先检查是否已下载了对应任务的模型，检查的顺序如下：

1. 本地的同级目录的checkpoints文件夹中
2. 本地缓存中

如果都没有，就会到网络上下载。

因此，无论是网络下载还是自己训练的模型使用，有两种解决思路：

- 在本地同级目录中新建checkpoints文件夹，将模型存放在该文件夹中
  
- 使用参数`checkpoints`，指定模型路径，如`model=wf(task='det_body',checkpoints='my_path/body17.onnx')`

最后提醒一下，自己到网络下载或自己训练的模型需要是ONNX格式。

## 报错专栏

你是否在使用XEduHub时遇到过报错？是否遇到ERROR时感到无所适从，甚至有点慌张？
没有关系！报错专栏将为你可能在使用过程中出现的错误提供解决方案！
这里收录着使用XEduHub中的常见报错并呈现对应的解决方案，如果你遇到了问题，查看这个专栏，找到类似报错，并且解决它！

正在努力收录中……敬请期待！
