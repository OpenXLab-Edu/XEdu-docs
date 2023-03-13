# 揭秘BaseDT的强大功能

数据处理是机器学习和深度学习中不可或缺的一环，它直接影响着模型的性能和效果。然而，在实际应用中，我们常常面临着数据处理过程中的各种困难和挑战，比如数据格式不统一、数据质量低下、数据处理流程复杂等。为了解决这些问题，我们开发了BaseDT库。BaseDT是一个用Python编写的库，它可以让你用一行代码就完成数据的各种操作，比如resize、crop、normalize、转换格式等。无论你是处理图片数据、文本数据、语音数据，还是其他类型的数据，BaseDT都可以帮助你轻松地准备好你的数据集，并且保持部署模型时和推理时的pipeline一致。BaseDT还支持数据集的处理，例如数据集的下载、检查和转换。此外，BaseDT还支持数据的可视化和I/O设备的调用。BaseDT不仅是一个功能强大的库，也是一个易于使用和扩展的库。它可以让你专注于模型的构建和训练，而不用担心数据处理的问题。BaseDT是一个值得你尝试和信赖的库。

## 板块1：数据的处理

BaseDT提供了一个data模块，它包含了多个子模块和类，分别针对不同类型和格式的数据提供了处理功能。例如，ImageData子模块可以让你对图片数据进行resize、crop、normalize等操作；TextData子模块可以让你对文本数据进行分词、词向量化等操作；AudioData子模块可以让你对语音数据进行重采样、裁剪、增强等操作。

### （1）图片数据处理

图片数据处理是指对数字图像进行各种操作以改变其属性或提取其中的信息。图片数据处理在许多领域都有广泛地应用，比如计算机视觉、医学成像、生物识别、数字艺术等。通过对图片进行合适地处理，我们可以提高图片地质量、增强图片地特征、分析图片地内容、生成新地图片等。

BaseDT库可进行一些常见的图片数据处理操作，比如resize、crop和normalize。这些操作可以帮助我们调整图片地大小和形状以适应不同地需求，以及将图片的数值标准化以方便后续地计算。

**示例代码：**

```Python
from BaseDT.data import ImageData
#图片路径，修改为对应路径
img = r"D:\PythonProject\OpenDataLab-Edu\dataset\cat2.jpg" 
data = ImageData(img, size=(256, 256), crop_size=(224,224), normalize=True)#
data.show()
```

**参数详解：**

`img`：一个np.ndarray对象或者一个图像文件的路径。

`size`: 一个元组，表示要调整的图像大小，例如(256, 256)。

`crop_size`: 一个元组，表示要裁剪的图像区域大小，例如(224,224)。

`normalize`: 一个布尔值，表示是否要对图像数据进行归一化处理。

### （2）文本数据处理

文本数据处理是指对自然语言形式的数据进行各种操作以改变其属性或提取其中的信息。文本数据处理在许多领域都有广泛地应用，比如自然语言理解、机器翻译、情感分析、信息检索等。通过对文本进行合适地处理，我们可以提高文本地质量、增强文本地特征、分析文本地内容、生成新地文本等。

BaseDT库来进行一些常见的文本数据处理操作，比如向量化。向量化是指将文字转换为数字形式以方便后续地计算和分析。向量化可以帮助我们度量文字之间地相似性、分类文字地类别、预测文字地情感等。

**示例代码：**

```Python
from BaseDT.data import TextData
# 文本数据，字典类型
texts = {'city': 'Dubai', 'temperature': 33}
data = TextData(texts, vectorize = True)
print(data.value)
```

**参数详解：**

`texts`：一个字符串、一个字典或者一个列表，表示要处理的文本数据。如果是字符串，那么表示单个文本；如果是字典，那么表示多个文本，每个键值对代表一个特征和它的取值；如果是列表，那么表示多个文本，每个元素代表一个文本。

`vectorize`：一个布尔值，表示是否要对文本数据进行向量化。如果为True，那么会调用vectorizer方法，根据文本类型（str或dict）选择合适的向量化器（CountVectorizer或DictVectorizer），并返回一个词向量矩阵；如果为False，那么不会进行向量化。

### （3）语音数据处理

敬请期待。

### （4）通用数据处理

敬请期待。

### （5）模型部署数据处理

由于网络模型在训练时会对输入数据进行一系列预处理操作以提高模型性能，在部署时也需要对输入设备采集到的原始数据进行相同或类似地预处理操作以保证模型效果。然而，在不同设备上进行相同或类似地预处理操作并不容易实现，并且可能会增加部署成本和时间。为了简化这个过程，并且保证预处理操作与训练时相同或类似地执行方式与顺序，在BaseDT库中我们提供了一个简单而强大地功能：只需指定网络模型所使用地backbone名称（例如"MobileNet"），就可以自动完成与该backbone对应地预处理操作，并将原始输入转换为网络模型可接受地张量格式。

**示例代码：**

```Python
from BaseDT.data import ImageData
img = r"D:\PythonProject\OpenDataLab-Edu\dataset\cat2.jpg" #修改为对应路径
data = ImageData(img, backbone = "MobileNet")
tensor_value = data.to_tensor()
```

**参数详解：**

`img`: 一个np.ndarray对象或者一个图像文件的路径。

`backbone`: 网络模型所使用地backbone名称。

## 板块2：数据集的处理

数据集的好坏决定了模型的训练结果，数据集的制作是深度学习模型训练的第一步。在实现一个深度学习项目时，除了搭建模型的网络结构，更重要的一点是处理好项目需要的数据集，那么就需要进行数据集处理。BaseDT提供了的DataSet类可支持对不同类型和格式的数据集进行处理。

### （1）常用数据集下载

敬请期待。

### （2）数据集格式检查

敬请期待。

### （3）数据集格式转换

针对网上下载的常见格式的数据集和自己标注或整理的数据集**（目前支持IMAGENET、VOC、COCO和OpenInnoLab在线标注格式）**，均可使用BaseDT完成格式转换。

**示例代码：**

```Python
from BaseDT.dataset import DataSet
ds = DataSet(r"my_dataset") # 指定为新数据集路径
# 默认比例为train_ratio = 0.7, test_ratio = 0.1, val_ratio = 0.2
ds.make_dataset(r"G:\\测试数据集\\fruit_voc", src_format="VOC",train_ratio = 0.8, test_ratio = 0.1, val_ratio = 0.1) # 指定待转格式的原始数据集路径，原始数据集格式，划分比例
```

**参数详解：**

`source`: 原始数据集路径。

`src_format`: 原始数据集格式，目前支持"IMAGENET"、“VOC”、“COCO"、"INNOLAB"（OpenInnoLab平台在线标注格式）。

`train_ratio , test_ratio, val_ratio`：训练集、测试集、验证集划分比例，默认比例为train_ratio = 0.7, test_ratio = 0.1, val_ratio = 0.2。

## 板块3：数据的可视化

BaseDT提供了一个plot模块，它可以让你对不同任务的数据进行可视化，例如绘制分类任务的混淆矩阵、目标检测任务的目标框、分割任务的掩膜等。plot模块支持多种显示方式，让你可以方便地查看和分析图片、文本、语音等不同类型的数据。

### （1）绘制分类任务混淆矩阵

敬请期待。

### （2）绘制目标检测任务的检测框

针对目标检测任务目标框的绘制，BaseDT可根据网络模型的输出数据出检测出相应目标框。

**示例代码：**

```Python
from BaseDT.plot import imshow_det_bboxes
img = 'test.jpg'
# imshow_det_bboxes(图像， 框和得分，标签， 类别， 得分阈值)
imshow_det_bboxes(img, bboxes = [[3,25,170,263,0.9]],labels = [0], class_names = ["cat"], score_thr = 0.8)
```

**参数详解：**

`img`: 一个np.ndarray对象或者一个图像文件的路径。

`bboxes`: 边界框（带分数），这里只有一个边界框，坐标为 [3,25,170,263] ，分数为 0.9。

`labels`: 边界框的类别，这里只有一个类别，编号为 0。

`class_names`: 每个类别的名称，这里只有一个名称，就是 “cat”。

`score_thr`: 显示边界框的最小分数，这里是 0.8 ，表示只显示分数大于等于 0.8 的边界框。

## 板块4：I/O设备

BaseDT提供的io模块可支持数据的I/O设备的调用，它可以让你方便地从不同的来源获取和输出数据。例如，你可以用io模块中的MicroPhone类来调用麦克风录音，并且直接将录音转换成AudioData对象，进行后续的处理和分析。此外，BaseDT还支持其他多种I/O设备，比如摄像头、扫描仪、打印机等，让你可以轻松地处理各种类型和格式的数据。

### （1） 调用麦克风

针对麦克风设备，BaseDT可调用麦克风，并用其录音。

**示例代码：**

```Python
from BaseDT.io import MicroPhone
# 创建麦克风对象
microphone = MicroPhone()
# 录音两秒
audio_data = microphone.record_audio(time = 2)
```

**参数详解：**

`time`：录音时间，单位是秒，这里是录音两秒。
