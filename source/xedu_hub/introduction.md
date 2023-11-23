# XEduHub功能详解

## 什么是Workflow？

我们在使用XEduHub时都需要执行这段代码`from XEdu.hub import Workflow as wf`。Workflow的基本逻辑是使用训练好的模型对数据进行推理。

那什么是Workflow呢？在使用XEduHub里的单个模型时，Workflow就是模型推理的推理流，从数据，到输入模型，再到输出推理结果。在使用XEduHub里多个模型进行联动时，Workflow可以看做不同模型之间的数据流动，例如首先进行多人的目标检测，将检测到的数据传入关键点识别模型从而对每个人体进行关键点识别。

总之，Workflow里有丰富的深度学习工具，你可以灵活地使用这些工具，根据自身需求，组建属于你自己的Workflow。下面开始介绍Workflow中丰富的深度学习工具。

### 强烈安利项目<a href="https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&backpath=/pjlab/projects/list#public">XEduHub实例代码-入门完整版</a>

<a href="https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&backpath=/pjlab/projects/list#public">https://www.openinnolab.org.cn/pjlab/project?id=65518e1ae79a38197e449843&backpath=/pjlab/projects/list#public</a>

通过学习“XEduHub实例代码-入门完整版”，可以在项目实践中探索XEduHub的魅力，项目中通俗易懂的讲解和实例代码也能帮助初学者快速入门XEduHub。

## 1. 关键点识别

关键点识别是深度学习中的一项关键任务，旨在检测图像或视频中的关键位置，通常代表物体或人体的重要部位。XEduHub支持的关键点识别任务有：人体关键点`pose_body`、人脸关键点`pose_face`、人手关键点`pose_hand`和所有人体关键点识别`pose_wholebody`。

### 人体关键点识别 

人体关键点识别是一项计算机视觉任务，旨在检测和定位图像或视频中人体的关键位置，通常是关节、身体部位或特定的解剖结构。

这些关键点的检测可以用于人体姿态估计和分类、动作分析、手势识别等多种应用。

XEduHub提供了两个识别人体关键点的优质模型:`pose_body17`和`pose_body26`，能够在使用cpu推理的情况下，快速识别出身体的关键点。

 数字表示了识别出人体关键点的数量。

`pose_body17`模型能识别出17个人体骨骼关键点，`pose_body26`模型能识别出26个人体骨骼关键点。

![](../images/xeduhub/body.png)

#### 代码样例

```python
from XEdu.hub import Workflow as wf
body = wf(task='pose_body') # 数字可省略，当省略时，默认为pose_body17
keypoints,img_with_keypoints = body.inference(data='data/body.jpg',img_type='pil',show=True) # 进行模型推理
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

`wf()`中共有两个参数可以设置

- `task`决定了使用哪个关键点识别模型，可选取值为：`[pose_body17,pose_body26]`
- `download_path`参数决定了预训练模型下载的路径。默认是下载到同级的checkpoints文件夹中，当代码运行时，会先在本地的同级目录中寻找是否有已下载的预训练模型，如果没有，到本地缓存中寻找，如果本地缓存没有，查看是不是指定了模型的路径，如果都没有，到网络下载。用户也可指定模型的下载路径，如`dowload_path='my_checkpoint'`。

#### 2. 模型推理

```python
keypoints,img_with_keypoints = body.inference(data='data/body.jpg',img_type='pil',show=True) # 进行模型推理
```

![](../images/xeduhub/body_result.png)

模型推理`inference()`可传入参数：

- `data`: 指定待识别关键点的图片，可以是以图片路径形式传入，也可直接传入cv2或pil格式的图片。

- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出关键点识别完成后的图片。

- `img_type`: 关键点识别完成后会返回含有关键点的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。
- `bbox`：该参数可配合目标检测使用。在多人关键点检测中，该参数指定了要识别哪个检测框中的关键点。

模型推理返回结果：

- `keypoints`以二维数组的形式保存了所有关键点的坐标，每个关键点(x,y)被表示为`[x,y]`根据前面的图示，要获取到某个特定序号`i`的关键点，只需要访问`keypoints[i]`即可。

- `img_with_keypoints`是个三维数组，以pil格式保存了关键点识别完成后的图片。

#### 3. 结果输出

```python
format_result = body.format_output(lang='zh')# 参数lang设置了输出结果的语言，默认为中文
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。参数`lang`设置了输出结果的语言，如果不指定默认为中文。

`format_result`以字典形式存储了推理结果，共有两个键：`关键点坐标`和`分数`。关键点坐标以二维数组形式保存了每个关键点的[x,y]坐标，而分数则是对应下标的关键点的分数，以一维数组形式保存。

![](../images/xeduhub/body-format.png)

输出图

```python
body.show(img_with_keypoints)
```

`show()`能够输出带有关键点和关键点连线的结果图像。

![](../images/xeduhub/body_show.png)

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
keypoints,img_with_keypoints = face.inference(data='data/face.jpg',img_type='pil',show=True) # 进行模型推理
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

`wf()`中共有两个参数可以设置

- `task`决定了使用哪个关键点识别模型，人脸关键点识别模型为`pose_face`。
- `download_path`参数决定了预训练模型下载的路径。默认是下载到同级的checkpoints文件夹中，当代码运行时，会先在本地的同级目录中寻找是否有已下载的预训练模型，如果没有，到本地缓存中寻找，如果本地缓存没有，查看是不是指定了模型的路径，如果都没有，到网络下载。用户也可指定模型的下载路径，如`dowload_path='my_checkpoint'`。

#### 2. 模型推理

```python
keypoints,img_with_keypoints = face.inference(data='data/face.jpg',img_type='pil',show=True) # 进行模型推理
```

![](../images/xeduhub/face_result.png)

模型推理`inference()`可传入参数：

- `data`: 指定待识别关键点的图片，可以是以图片路径形式传入，也可直接传入cv2或pil格式的图片。

- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出关键点识别完成后的图片。

- `img_type`: 关键点识别完成后会返回含有关键点的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。
- `bbox`：该参数可配合目标检测使用。在多人脸关键点检测中，该参数指定了要识别哪个检测框中的关键点。

模型推理返回结果：

- `keypoints`以二维数组的形式保存了所有关键点的坐标，每个关键点(x,y)被表示为`[x,y]`根据前面的图示，要获取到某个特定序号`i`的关键点，只需要访问`keypoints[i]`即可。

- `img_with_keypoints`是个三维数组，以pil格式保存了关键点识别完成后的图片。

#### 3. 结果输出

```python
format_result = face.format_output(lang='zh')# 将推理结果进行格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。参数`lang`设置了输出结果的语言，如果不指定默认为中文。

`format_result`以字典形式存储了推理结果，共有两个键：`关键点坐标`和`分数`。关键点坐标以二维数组形式保存了每个关键点的[x,y]坐标，而分数则是对应下标的关键点的分数，以一维数组形式保存。

```python
face.show(img_with_keypoints)# 展示推理图片
```

`show()`能够输出带有关键点的结果图像。

![](../images/xeduhub/face_show.png)

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
keypoints,img_with_keypoints = hand.inference(data='data/hand.jpg',img_type='pil',show=True) # 进行模型推理
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

`wf()`中共有两个参数可以设置

- `task`决定了使用哪个关键点识别模型，人手关键点识别模型为`pose_hand`
- `download_path`参数决定了预训练模型下载的路径。默认是下载到同级的checkpoints文件夹中，当代码运行时，会先在本地的同级目录中寻找是否有已下载的预训练模型，如果没有，到本地缓存中寻找，如果本地缓存没有，查看是不是指定了模型的路径，如果都没有，到网络下载。用户也可指定模型的下载路径，如`dowload_path='my_checkpoint'`。

#### 2. 模型推理

```python
keypoints,img_with_keypoints = hand.inference(data='data/hand.jpg',img_type='pil',show=True) # 进行模型推理
```

![](../images/xeduhub/hand_result.png)

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

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。参数`lang`设置了输出结果的语言，如果不指定默认为中文。

`format_result`以字典形式存储了推理结果，共有两个键：`关键点坐标`和`分数`。关键点坐标以二维数组形式保存了每个关键点的[x,y]坐标，而分数则是对应下标的关键点的分数，以一维数组形式保存。

```python
hand.show(img_with_keypoints)# 展示推理图片
```

`show()`能够输出带有关键点的结果图像。

![](../images/xeduhub/hand_show.png)

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
keypoints,img_with_keypoints = wholebody.inference(data='data/wholebody.jpg',img_type='pil',show=True) # 进行模型推理
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

`wf()`中共有两个参数可以设置

- `task`决定了使用哪个关键点识别模型，全身关键点提取模型为`pose_wholebody`
- `download_path`参数决定了预训练模型下载的路径。默认是下载到同级的checkpoints文件夹中，当代码运行时，会先在本地的同级目录中寻找是否有已下载的预训练模型，如果没有，到本地缓存中寻找，如果本地缓存没有，查看是不是指定了模型的路径，如果都没有，到网络下载。用户也可指定模型的下载路径，如`dowload_path='my_checkpoint'`。

#### 2. 模型推理

```python
keypoints,img_with_keypoints = wholebody.inference(data='data/wholebody.jpg',img_type='pil',show=True) # 进行模型推理
```

![](../images/xeduhub/wholebody_result.png)

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

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。参数`lang`设置了输出结果的语言，如果不指定默认为中文。

`format_result`以字典形式存储了推理结果，共有两个键：`关键点坐标`和`分数`。关键点坐标以二维数组形式保存了每个关键点的[x,y]坐标，而分数则是对应下标的关键点的分数，以一维数组形式保存。

```python
wholebody.show(img_with_keypoints)# 展示推理图片
```

`show()`能够输出带有关键点的结果图像。

![](../images/xeduhub/wholebody_show.png)

#### 4. 结果保存

```python
wholebody.save(img_with_keypoints,'img_with_keypoints.jpg')# 保存推理图片
```

`save()`方法能够保存带有关键点的图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

## 2. 目标检测

目标检测是一种计算机视觉任务，其目标是在图像或视频中检测并定位物体的位置，并为每个物体分配类别标签。

实现目标检测通常包括特征提取、物体位置定位、物体类别分类等步骤。这一技术广泛应用于自动驾驶、安全监控、医学影像分析、图像搜索等各种领域，为实现自动化和智能化应用提供了关键支持。

XEduHub目标支持目标检测任务有：coco目标检测`det_coco`、人体检测`det_body`、人脸检测`det_face`和人手检测`det_hand`。

### coco目标检测

COCO（Common Objects in Context）是一个用于目标检测和图像分割任务的广泛使用的数据集和评估基准。它是计算机视觉领域中最重要的数据集之一，在XEduHub中的该模型能够检测出80类coco数据集中的物体：`det_coco`。

若要查看coco目标检测中的所有类别可运行以下代码：

```python
wf.coco_class()
```

#### 代码样例

```python
from XEdu.hub import Workflow as wf
det_coco = wf(task='det_coco')
result,img_with_box = det_coco.inference(data='data/det_coco.jpg',img_type='pil',show=True) # 进行模型推理
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

`wf()`中共有两个参数可以设置

- `task`决定了使用哪个检测模型，coco目标检测的模型为`det_coco`。
- `download_path`参数决定了预训练模型下载的路径。默认是下载到同级的checkpoints文件夹中，当代码运行时，会先在本地的同级目录中寻找是否有已下载的预训练模型，如果没有，到本地缓存中寻找，如果本地缓存没有，查看是不是指定了模型的路径，如果都没有，到网络下载。用户也可指定模型的下载路径，如`dowload_path='my_checkpoint'`。

#### 2. 模型推理

```python
result,img_with_box = det_coco.inference(data='data/det_coco.jpg',img_type='pil',show=True) # 进行模型推理
```

![](../images/xeduhub/det_coco_result.png)

模型推理`inference()`可传入参数：

- `data`：指定待检测的图片。
- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出目标检测完成后的图片。
- `img_type`：目标检测完成后会返回含有检测框的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。
- `target_class`：该参数在使用`cocodetect`的时候可以指定要检测的对象，如：`person`，`cake`等等。
- `thr`: 设置检测框阈值，取值范围为`[0,1]`超过该阈值的检测框被视为有效检测框，进行显示。

模型推理返回结果：

- `result`：以二维数组的形式保存了检测框左上角顶点的(x,y)坐标以及检测框的宽度w和高度h（之所以是二维数组，是因为该模型能够检测多个物体，因此当检测到多个物体时，就会有多个[x,y,w,h]的一维数组，所以需要以二维数组形式保存），我们可以利用这四个数据计算出其他三个顶点的坐标。

![](../images/xeduhub/det_res.png)

- `img_with_box：`：是个三维数组，以cv2格式保存了包含了检测框的图片。

#### 3. 结果输出

```python
format_result = det_coco.format_output(lang='zh')# 将推理结果进行格式化输出
```

![](../images/xeduhub/det_coco_format.png)

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。参数`lang`设置了输出结果的语言，如果不指定默认为中文。

`format_result`以字典形式存储了推理结果，共有三个键：`检测框`、`分数`和`类别`。检测框以二维数组形式保存了每个检测框的坐标信息[x,y,w,h]，而分数则是对应下标的检测框的置信度，以一维数组形式保存，类别则是检测框中对象所属的类别，以一维数组形式保存。

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

XEduHub提供了进行人体目标检测的模型：`det_body`，该模型能够进行单人的人体目标检测。

#### 代码样例

```python
from XEdu.hub import Workflow as wf
det_body = wf(task='det_body')
result,img_with_box = det_body.inference(data='data/det_body.jpg',img_type='pil',show=True) # 进行模型推理
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

`wf()`中共有两个参数可以设置

- `task`决定了使用哪个检测模型，人体目标检测模型为`det_body`
- `download_path`参数决定了预训练模型下载的路径。默认是下载到同级的checkpoints文件夹中，当代码运行时，会先在本地的同级目录中寻找是否有已下载的预训练模型，如果没有，到本地缓存中寻找，如果本地缓存没有，查看是不是指定了模型的路径，如果都没有，到网络下载。用户也可指定模型的下载路径，如`dowload_path='my_checkpoint'`。

#### 2. 模型推理

```python
result,img_with_box = det_body.inference(data='data/det_body.jpg',img_type='pil',show=True) # 进行模型推理
```

![](../images/xeduhub/det_result.png)

模型推理`inference()`可传入参数：

- `data`：指定待检测的图片。
- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出目标检测完成后的图片。
- `img_type`：目标检测完成后会返回含有检测框的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。
- `thr`: 设置检测框阈值，取值范围为`[0,1]`超过该阈值的检测框被视为有效检测框，进行显示。

模型推理返回结果：

- `result`：以二维数组的形式保存了检测框左上角顶点的(x,y)坐标以及检测框的宽度w和高度h（之所以是二维数组，是因为该模型能够检测多个人体，因此当检测到多个人体时，就会有多个[x,y,w,h]的一维数组，所以需要以二维数组形式保存），我们可以利用这四个数据计算出其他三个顶点的坐标。

- `img_with_box：`：是个三维数组，以cv2格式保存了包含了检测框的图片。

#### 3. 结果输出

```python
format_result = det_body.format_output(lang='zh')# 将推理结果进行格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。参数`lang`设置了输出结果的语言，如果不指定默认为中文。

`format_result`以字典形式存储了推理结果，共有两个键：`检测框`、`分数`。检测框以二维数组形式保存了每个检测框的坐标信息[x,y,w,h]，而分数则是对应下标的检测框的置信度，以一维数组形式保存。

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
result,img_with_box = det_face.inference(data='data/det_face.jpg',img_type='pil',show=True) # 进行模型推理
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

`wf()`中共有两个参数可以设置

- `task`决定了使用哪个检测模型，人脸目标检测模型为`det_face`
- `download_path`参数决定了预训练模型下载的路径。默认是下载到同级的checkpoints文件夹中，当代码运行时，会先在本地的同级目录中寻找是否有已下载的预训练模型，如果没有，到本地缓存中寻找，如果本地缓存没有，查看是不是指定了模型的路径，如果都没有，到网络下载。用户也可指定模型的下载路径，如`dowload_path='my_checkpoint'`。

#### 2. 模型推理

```python
result,img_with_box = det_face.inference(data='data/det_face.jpg',img_type='pil',show=True) # 进行模型推理
```

![](../images/xeduhub/det_face_result.png)

模型推理`inference()`可传入参数：

- `data`：指定待检测的图片。
- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出目标检测完成后的图片。
- `img_type`：目标检测完成后会返回含有检测框的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。
- `thr`: 设置检测框阈值，取值范围为`[0,1]`超过该阈值的检测框被视为有效检测框，进行显示。

模型推理返回结果：

- `result`：以二维数组的形式保存了检测框左上角顶点的(x,y)坐标以及检测框的宽度w和高度h（之所以是二维数组，是因为该模型能够检测多个人脸，因此当检测到多个人脸时，就会有多个[x,y,w,h]的一维数组，所以需要以二维数组形式保存），我们可以利用这四个数据计算出其他三个顶点的坐标。
- `img_with_box：`：是个三维数组，以cv2格式保存了包含了检测框的图片。

#### 3. 结果输出

```python
format_result = det_face.format_output(lang='zh')# 将推理结果进行格式化输出
```

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。参数`lang`设置了输出结果的语言，如果不指定默认为中文。

`format_result`以字典形式存储了推理结果，只有一个键：`检测框`、`分数`。检测框以二维数组形式保存了每个检测框的坐标信息[x,y,w,h]。需要注意的是由于使用的为opencv的人脸检测模型，因此在`format_output`时缺少了分数这一指标。

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

`wf()`中共有两个参数可以设置

- `task`决定了使用哪个检测模型，全身关键点提取模型为`det_face`
- `download_path`参数决定了预训练模型下载的路径。默认是下载到同级的checkpoints文件夹中，当代码运行时，会先在本地的同级目录中寻找是否有已下载的预训练模型，如果没有，到本地缓存中寻找，如果本地缓存没有，查看是不是指定了模型的路径，如果都没有，到网络下载。用户也可指定模型的下载路径，如`dowload_path='my_checkpoint'`。

#### 2. 模型推理

![](../images/xeduhub/det_hand_result.png)

模型推理`inference()`可传入参数：

- `data`：指定待检测的图片。
- `show`: 可取值：`[True,False]` 默认为`False`。如果取值为`True`，在推理完成后会直接输出目标检测完成后的图片。
- `img_type`：目标检测完成后会返回含有检测框的图片，该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。
- `thr`: 设置检测框阈值，取值范围为`[0,1]`超过该阈值的检测框被视为有效检测框，进行显示。

模型推理返回结果：

- `result`：以二维数组的形式保存了检测框左上角顶点的(x,y)坐标以及检测框的宽度w和高度h（之所以是二维数组，是因为该模型能够检测多个人手，因此当检测到多个人手时，就会有多个[x,y,w,h]的一维数组，所以需要以二维数组形式保存），我们可以利用这四个数据计算出其他三个顶点的坐标。
- `img_with_box：`：是个三维数组，以cv2格式保存了包含了检测框的图片。

#### 3. 结果输出

```python
format_result = det_hand.format_output(lang='zh')# 将推理结果进行格式化输出
```

![](../images/xeduhub/det_hand_format.png)

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。参数`lang`设置了输出结果的语言，如果不指定默认为中文。

`format_result`以字典形式存储了推理结果，共有两个键：`检测框`、`分数`。检测框以二维数组形式保存了每个检测框的坐标信息[x,y,w,h]，而分数则是对应下标的检测框的置信度，以一维数组形式保存。

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

## 3. 图像分类

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

`wf()`中共有两个参数可以设置

- `task`决定了进行图像分类的模型，目前支持`cls_imagenet`
- `download_path`参数决定了预训练模型下载的路径。默认是下载到同级的checkpoints文件夹中，当代码运行时，会先在本地的同级目录中寻找是否有已下载的预训练模型，如果没有，到本地缓存中寻找，如果本地缓存没有，查看是不是指定了模型的路径，如果都没有，到网络下载。用户也可指定模型的下载路径，如`dowload_path='my_checkpoint'`

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

`format_result`是一个字典，以格式化的方式展示了这张图片最有可能的分类结果。预测值表示图片分类标签在所有一千个分类中的索引，分数是属于这个分类的概率，预测类别是分类标签的内容。

![](../images/xeduhub/cls_format.png)

## 4. 风格迁移

风格迁移，这里主要指的是图像风格迁移，指的是一种计算机视觉和图像处理技术。它允许将一个图像的艺术风格应用到另一个图像上，从而创建出一个新的图像，同时保留了原始图像的内容，但采用了第二个图像的风格。XEduHub提供了进行图像风格迁移的模型：`gen_style`，它预设了5种图像风格，并支持用户输入自定义的图片进行自定义风格迁移。

### 代码样例

```python
from XEdu.hub import Workflow as wf
style = wf(task='gen_style',style='mosaic')
result = style.inference(data='data/cat101.jpg',img_type='cv2',show=True)# 进行模型推理
style.show(result)# 展示推理图片
style.save(result,"style_cat.jpg")# 保存推理图片
```

### 代码解释

#### 1. 模型声明

```python
from XEdu.hub import Workflow as wf
style = wf(task='gen_style',style='mosaic')
```

`wf()`中共有三个参数可以设置

- `task`决定了任务类型
- `download_path`参数决定了预训练模型下载的路径。默认是下载到同级的checkpoints文件夹中，当代码运行时，会先在本地的同级目录中寻找是否有已下载的预训练模型，如果没有，到本地缓存中寻找，如果本地缓存没有，查看是不是指定了模型的路径，如果都没有，到网络下载。用户也可指定模型的下载路径，如`dowload_path='my_checkpoint'`
- `style`决定了风格迁移所使用的风格。

运行代码`wf.support_style()`可查看当前预设的风格。当前预设风格共有五种，如下图所示。

![](../images/xeduhub/style_all.png)

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
result = style.inference(data='data/cat101.jpg',img_type='cv2',show=True)# 进行模型推理
```

![](../images/xeduhub/style_result.png)

模型推理`inference()`可传入参数：

- `data`: 待进行风格迁移的图片，可以是以图片路径形式传入，也可直接传入cv2或pil格式的图片。
- `show`: 可取值：`[true,false]` 默认为`false`。如果取值为`true`，在推理完成后会直接输出风格迁移完成后的图片。
- `img_type`: 推理完成后会直接输出风格迁移完成后的图片。该参数指定了返回图片的格式，可选有:`['cv2','pil']`，默认值为`None`，如果不传入值，则不会返回图。

模型推理返回结果：

- `result`是个三维数组，以cv2格式保存了风格迁移完成后的图片。

#### 3. 结果输出

```python
style.show(result)# 展示推理后的图片
```

`show()`能够输出风格迁移后的结果图像。

![](../images/xeduhub/style_show.png)

#### 4. 结果保存

```python
style.save(result,"style_cat.jpg")# 保存推理图片
```

`save()`方法能够保存风格迁移后的图像

该方法接收两个参数，一个是图像数据，另一个是图像的保存路径。

## 5. 光学字符识别（OCR）

光学字符识别（Optical Character Recognition, OCR）是一项用于将图像或扫描的文档转换为可编辑的文本格式的技术。OCR技术能够自动识别和提取图像或扫描文档中的文本，并将其转化为计算机可处理的文本格式。OCR技术在车牌识别、证件识别、文档扫描、拍照搜题等多个场景有着广泛应用。

XEduHub使用的OCR模型是来自百度的开源免费的OCR模型：rapidocr，这个模型运行速度快，性能优越，小巧灵活，并且能支持超过6000种字符的识别，如简体中文、繁体中文、英文、数字和其他艺术字等等。

注意：你可以在当前项目中找到名为**font**的文件夹，里面的FZVTK.TTF文件是一种字体文件，为了显示识别出的文字而使用。

### 代码样例

```python
from XEdu.hub import Workflow as wf
ocr = wf(task="ocr")
result,ocr_img = ocr.inference(data='data/ocr_img.png',img_type='cv2',show=True) # 进行模型推理
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

`wf()`中共有一个参数可以设置，这里的模型不是以onnx方式下载，而是以python库的形式下载和安装，因此不同于之前任务的下载方式，也无需指定下载路径。

- `task`决定了任务类型

#### 2. 模型推理

```python
result,ocr_img = ocr.inference(data='data/ocr_img.png',img_type='cv2',show=True) # 进行模型推理
```

![](../images/xeduhub/ocr_result.png)

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

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。参数`lang`设置了输出结果的语言，如果不指定默认为中文。

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

## 6. MMEdu模型推理

XEduHub现在可以支持使用MMEdu导出的onnx模型进行推理啦！如果你想了解如何使用MMEdu训练模型，可以看这里：[解锁图像分类模块：MMClassification](https://xedu.readthedocs.io/zh/master/mmedu/mmclassification.html)、[揭秘目标检测模块：MMDetection](https://xedu.readthedocs.io/zh/master/mmedu/mmdetection.html)。

如果你想了解如何将使用[MMEdu](https://xedu.readthedocs.io/zh/master/mmedu.html)训练好的模型转换成ONNX格式，可以看这里[最后一步：AI模型转换](https://xedu.readthedocs.io/zh/master/mmedu/model_convert.html)。OK，准备好了ONNX模型，那么就开始使用XEduHub吧！

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

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。参数`lang`设置了输出结果的语言，如果不指定默认为中文。

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

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。参数`lang`设置了输出结果的语言，如果不指定默认为中文。

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

## 7. BaseNN模型推理

XEduHub现在可以支持使用BaseNN导出的onnx模型进行推理啦！如果你想了解如何将使用[BaseNN](https://xedu.readthedocs.io/zh/master/basenn.html)训练好的模型转换成ONNX格式，可以看这里：[BaseNN模型文件格式转换](https://xedu.readthedocs.io/zh/master/basenn/introduction.html#id29)。OK，准备好了ONNX模型，那么就开始使用XEduHub吧！

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

`format_output()`能够将模型推理结果以标准美观的方式进行输出。输出结果与`format_result`保存的内容一致。参数`lang`设置了输出结果的语言，如果不指定默认为中文。

`format_output`的结果是一个结果字典，这个字典的第一个元素有两个键，`预测值`、`分数`，代表着该手写数字的分类标签以及属于该分类标签的概率。

## 8. BaseML模型推理

XEduHub现在可以支持使用BaseML导出的pkl模型文件进行推理啦！如果你想了解如何将使用[BaseML](https://xedu.readthedocs.io/zh/master/baseml.html)训练模型并保存成.pkl模型文件，可以看这里：[BaseML模型保存](https://xedu.readthedocs.io/zh/master/baseml/introduction.html#id10)。OK，准备好了pkl模型，那么就开始使用XEduHub吧！

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

`format_result`以字典形式保存了模型的推理结果，由于使用的是聚类模型，输出结果为这两个特征数据所对应的聚类标签。

如果此时你有冲动去使用BaseML完成模型训练到推理，再到转换与应用，快去下文学习[BaseML的相关使用](https://xedu.readthedocs.io/zh/master/baseml.html)吧！

## 9. 其他onnx模型推理

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

## 报错专栏

你是否在使用XEduHub时遇到过报错？是否遇到ERROR时感到无所适从，甚至有点慌张？
没有关系！报错专栏将为你可能在使用过程中出现的错误提供解决方案！
这里收录着使用XEduHub中的常见报错并呈现对应的解决方案，如果你遇到了问题，查看这个专栏，找到类似报错，并且解决它！

正在努力收录中……敬请期待！
