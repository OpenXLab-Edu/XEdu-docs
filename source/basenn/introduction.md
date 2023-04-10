# BaseNN功能详解

## BaseNN是什么？

BaseNN是神经网络库，能够使用类似Keras却比Keras门槛更低的的语法搭建神经网络模型。可支持逐层搭建神经网路，深入探究网络原理。如果有如下需求，可以优先选择BaseNN：

a）简易和快速地搭建神经网络

b）支持搭建[CNN和RNN](https://xedu.readthedocs.io/zh/latest/basenn/introduction.html#rnncnn)，或二者的结合

c）同时支持CPU和GPU

## 解锁BaseNN使用方法

### 0. 引入包

```python
from BaseNN import nn
```

### 1. 声明模型

```python
model = nn()
```

### 2. 载入数据

```
model.load_dataset(x, y)
```

此处采用lvis鸢尾花数据集和MNIST手写体数字图像数据集作为示例。

读取并载入鸢尾花数据集（鸢尾花数据集以鸢尾花的特征作为数据来源，数据集包含150个数据集，有4维，分为3类（setosa、versicolour、virginica），每类50个数据，每个数据包含4个属性，花萼长度、宽度和花瓣长度、宽度）：

```python
# 训练数据
train_path = '../dataset/iris/iris_training.csv' 
x = np.loadtxt(train_path, dtype=float, delimiter=',',skiprows=1,usecols=range(0,4)) # 读取前四列，特征
y = np.loadtxt(train_path, dtype=int, delimiter=',',skiprows=1,usecols=4) # 读取第五列，标签
# 测试数据
test_path = '../dataset/iris/iris_test.csv'
test_x = np.loadtxt(test_path, dtype=float, delimiter=',',skiprows=1,usecols=range(0,4)) # 读取前四列，特征
test_y = np.loadtxt(test_path, dtype=int, delimiter=',',skiprows=1,usecols=4) # 读取第五列，标签
# 将数据载入
model.load_dataset(x, y)
```

读取并载入手写体图像数据集（数据集包含了0-9共10类手写数字图片，都是28x28大小的灰度图）：

```python
# 定义读取训练数据的函数
def read_data(path):
    data = []
    label = []
    dir_list = os.listdir(path)

    # 将顺序读取的文件保存到该list中
    for item in dir_list:
        tpath = os.path.join(path,item)

        # print(tpath)
        for i in os.listdir(tpath):
            # print(item)
            img = cv2.imread(os.path.join(tpath,i))
            imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # print(img)
            data.append(imGray)
            label.append(int(item))
    x = np.array(data)
    y = np.array(label)

    x = np.expand_dims(x, axis=1)
    return x, y
    
# 读取训练数据
train_x, train_y = read_data('../dataset/mnist/training_set')
# 载入数据
model.load_dataset(train_x, train_y) 
```

### 3. 搭建模型

逐层添加，搭建起模型结构，支持CNN（卷积神经网络）和RNN（循环神经网络）。注释标明了数据经过各层的尺寸变化。

```python
model.add(layer='Linear',size=(4, 10),activation='ReLU') # [120, 10]
model.add(layer='Linear',size=(10, 5), activation='ReLU') # [120, 5]
model.add(layer='Linear', size=(5, 3), activation='Softmax') # [120, 3]

model.add('LSTM',size=(128,256),num_layers=2)

model.add('Conv2D', size=(1, 3),kernel_size=( 3, 3), activation='ReLU') # [100, 3, 18, 18]
```

以上使用`add()`方法添加层，参数`layer='Linear'`表示添加的层是线性层，`size=(4,10)`表示该层输入维度为4，输出维度为10，`activation='ReLU'`表示使用ReLU激活函数。更详细[`add()`方法使用可见[附录1](https://xedu.readthedocs.io/zh/latest/basenn/introduction.html#add)。

### 4. 模型训练

模型训练可以采用以下函数：

```python
model.train(lr=0.01, epochs=500)
```

参数`lr`为学习率，`epochs`为训练轮数。

从训练类型的角度，可以分为正常训练和继续训练。

#### 4.1 正常训练

```python
model = nn() 
model.add(layer='Linear',size=(4, 10),activation='ReLU') # [120, 10]
model.add(layer='Linear',size=(10, 5), activation='ReLU') # [120, 5]
model.add(layer='Linear', size=(5, 3), activation='Softmax') # [120, 3]
model.load_dataset(x, y)
model.save_fold = 'checkpoints' # 指定模型保存路径
model.train(lr=0.01, epochs=1000)
```

`model.save_fold`表示训练出的模型文件保存的文件夹。

#### 4.2 继续训练

```python
model = nn()
model.load_dataset(x, y)
model.save_fold = 'checkpoints/new_train' # 指定模型保存路径
checkpoint = 'checkpoints/basenn.pth' # 指定已有模型的权重文件路径
model.train(lr=0.01, epochs=1000, checkpoint=checkpoint)
```

`checkpoint`为现有模型路径，当使用`checkpoint`参数时，模型基于一个已有的模型继续训练，不使用`checkpoint`参数时，模型从零开始训练。

### 5. 从数据类型看训练代码

针对不同类型的数据类型，载入数据、搭建模型和模型训练的代码会略有不同。深度学习常见的数据类型介绍详见[附录4](https://xedu.readthedocs.io/zh/latest/basenn/introduction.html#id23)。

#### 5.1 文本

在做文本识别等NLP（自然语言处理）领域项目时，一般搭建[RNN网络](https://xedu.readthedocs.io/zh/latest/basenn/introduction.html#rnncnn)训练模型，训练数据是文本数据，模型训练的示例代码如下：

```
model = nn()
model.load_dataset(x,y,word2idx=word2idx) # word2idx是词表（字典）
model.add('LSTM',size=(128,256),num_layers=2)
model.train(lr=0.001,epochs=1)
```

#### 5.2 图像

针对图像数据可增加classes参数设置，模型训练的示例代码如下：

```
model = nn()
model.load_dataset(x,y,classes=classes) # classes是类别列表（列表） //字典
model.add('Conv2D',...)
model.train(lr=0.01,epochs=1)
```

classes可传参数兼容列表，字典形式(以下三种形式均可)。

```
classes = ['cat','dog']
classes = {0:'cat',1:'dog'}
classes = {'cat':0, 'dog':1} # 与词表形式统一
```

注意：索引是数值类型（int)，类别名称是字符串（str)，即哪怕类别名也是数字0,1,...字典的键和值也有区别，例如：

```
# 正确示例
classes = {0:'0',1:'1'} # 索引to类别
classes = {'0':0, '1':1} # 类别to索引

# 错误示例
classes = {0:0,1:1} 
classes = {'0':'0', '1':'1'} 
```

#### 5.3 特征

针对特征数据，使用BaseNN各模块的示例代码即可。

```
model = nn()
model.load_dataset(x,y)
model.add('Linear',...)
model.train(lr=0.01,epochs=1)
```

### 6. 模型推理

可使用以下函数进行推理：

```python
model = nn() # 声明模型
checkpoint = 'checkpoints/iris_ckpt/basenn.pth' # 现有模型路径
result = model.inference(data=test_x, checkpoint=checkpoint) # 直接推理
model.print_result(result) # 输出字典格式结果
```

参数`data`为待推理的测试数据数据，该参数必须传入值；

`checkpoint`为已有模型路径，即使用现有的模型进行推理。

直接推理的输出结果数据类型为`numpy`的二维数组，表示各个样本的各个特征的置信度。

输出字典格式结果的数据类型为字典，格式为{样本编号：{预测值：x，置信度：y}}。`print_result()`函数调用即输出，但也有返回值。

针对文本数据的推理：

```python
model = nn()
data = '长'
checkpoint = 'xxx.pth'
result = model.inference(data=data, checkpoint=checkpoint)
index = np.argmax(result[0]) # 取得概率最大的字的索引，当然也可以取别的，自行选择即可
word = model.idx2word[index] # 根据词表获得对应的字
```

result为列表包含两个变量：[output, hidden]

output为numpy数组，里面是一系列概率值，对应每个字的概率。

hidden为高维向量，存储上下文信息，代表“记忆”，所以生成单个字可以不传入hidden，但写诗需要循环传入之前输出的hidden。

### 7. 模型的保存与加载

```python
# 保存
model.save_fold = 'mn_ckpt'
# 加载
model.load("basenn.pth")
```

参数为模型保存的路径，模型权重文件格式为`.pth`文件格式。

注：`train()`，`inference()`函数中也可通过参数控制模型的保存与加载，但这里也列出单独保存与加载模型的方法，以确保灵活性。

### 8. 查看模型结构

```python
model.print_model()
```

无参数。

### 9. 网络中特征可视化

BaseNN内置`visual_feature`函数可查看数据在网络中传递。

如输入数据为图片，指定图片和已经训练好的模型，可生成一张展示逐层网络特征传递的图片。

```
import cv2
from BaseNN import nn
model = nn()
model.load('mn_ckpt/basenn.pth')          # 保存的已训练模型载入
path = 'test_IMG/single_data.jpg'
img = cv2.imread(path,flags = 0)          # 图片数据读取
model.visual_feature(img,in1img = True)   # 特征的可视化
```

如输入数据为一维数据，指定数据和已经训练好的模型，可生成一个txt文件展示经过各层后的输出。

```
import numpy as np
from BaseNN import nn
model = nn()
model.load('checkpoints/iris_ckpt/basenn.pth')          # 保存的已训练模型载入
data = np.array(test_x[0]) # 指定数据,如测试数据的一行
model.visual_feature(data)   # 特征的可视化
```

### 10. 自定义随机数种子

默认初始化是随机的，每次训练结果都不一样。可以可使用`set_seed()`函数设定随机数种子，使得训练结果可被其他人复现。一旦指定，则每次训练结果一致。使用方法如下：

```Shell
model = nn()
model.set_seed(1235)
model.add(...)
...
model.train(...)
```

注：设定随机数种子`set_seed()`应当在搭建网络`add()`之前。

### 11. 自定义损失函数

损失函数（或称目标函数、优化评分函数）是编译模型时所需的参数之一。在机器学习和深度学习中，模型的训练通常涉及到一个优化过程，即通过不断调整模型的参数，使得模型在训练数据上的预测结果与实际结果的差距最小化。这个差距通常使用一个称为"损失函数"的指标来衡量。损失函数通常是一个关于模型参数的函数，用于度量模型预测结果与实际结果之间的差异。在模型训练过程中，模型会根据损失函数的值来调整自己的参数，以减小损失函数的值。

默认的损失函数是交叉熵损失函数，允许选择不同的损失函数，支持的损失函数见[附录](https://xedu.readthedocs.io/zh/latest/basenn/introduction.html#id22)。自选损失函数方法如下：

```
model.train(...,loss="CrossEntropyLoss")
```

### 12. 自定义评价指标

评价指标用于评估当前训练模型的性能。当模型编译后，评价指标应该作为 `metrics` 的参数来输入。默认的默认为准确率，允许选择其他的评价指标。支持的评价指标：acc（准确率），mae（平均绝对误差），mse（均方误差）。

自选评价指标方法如下：

```
model.train(...,metrics=["mse"])
```

因此针对不同的分类或回归任务，可指定不同的损失函数和评价指标。

例：

回归：`model.train(...,loss="SmoothL1Loss", metrics=["mae"])` 

分类：`model.train(...,loss="CrossEntropyLoss",metrics=["acc"])`

## 附录

### 1. add()详细介绍

此处以典型的LeNet5网络结构为例。注释标明了数据经过各层的尺寸变化。

```python
model.add('Conv2D', size=(1, 3),kernel_size=( 3, 3), activation='ReLU') # [100, 3, 18, 18]
model.add('MaxPool', kernel_size=(2,2)) # [100, 3, 9, 9]
model.add('Conv2D', size=(3, 10), kernel_size=(3, 3), activation='ReLU') # [100, 10, 7, 7]
model.add('AvgPool', kernel_size=(2,2)) # [100, 10, 3, 3]
model.add('Linear', size=(90, 10), activation='ReLU') # [100, 10]
model.add('Linear', size=(10, 2), activation='Softmax') # [100,2]
model.add(optimizer='SGD') # 设定优化器
```

添加层的方法为`add(self, layer=None, activation=None, optimizer='SGD', **kw)`，

参数:

​	layer：层的类型，可选值包括Conv2D, MaxPool, AvgPool, Linear。

​	activation：激活函数类型，可选值包括ReLU，Softmax。

​	optimizer：为优化器类型，默认值为SGD，可选值包括SGD，Adam，Adagrad，ASGD。

​	kw：关键字参数，包括与size相关的各种参数，常用的如size=(x,y)，x为输入维度，y为输出维度；					                     			kernel_size=(a,b)， (a,b)表示核的尺寸。

以下具体讲述各种层：

Conv2D：卷积层（二维），需给定size，kernel_size。

MaxPool：最大池化层，需给定kernel_size。

AvgPool：平均池化层，需给定kernel_size。

Linear：线性层，需给定size。

搭建RNN模型（循环神经网络）：

```
model.add('LSTM',size=(128,256),num_layers=2)
```

LSTM（Long Short-Term Memory，长短时记忆）是一种特殊的RNN（Recurrent Neural Network，循环神经网络）模型，主要用于处理序列数据。LSTM模型在自然语言处理、语音识别、时间序列预测等任务中被广泛应用，特别是在需要处理长序列数据时，LSTM模型可以更好地捕捉序列中的长程依赖关系。

size的两个值：

第一个为嵌入层维度（embedding_dim)，即每一个字用多少维的向量来表示。

第二个为隐藏层维度（hidden_dim)，即lstm隐藏层中神经元数量。

### 2. 支持的损失函数

| 序号 | 损失函数                                                     |
| ---- | :----------------------------------------------------------- |
| 1    | [nn.L1Loss](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss) |
| 2    | [nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss) |
| 3    | [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) |
| 4    | [nn.CTCLoss](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html#torch.nn.CTCLoss) |
| 5    | [nn.NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) |
| 6    | [nn.PoissonNLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.PoissonNLLLoss.html#torch.nn.PoissonNLLLoss) |
| 7    | [nn.GaussianNLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html#torch.nn.GaussianNLLLoss) |
| 8    | [nn.KLDivLoss](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html#torch.nn.KLDivLoss) |
| 9    | [nn.BCELoss](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss) |
| 10   | [nn.BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss) |
| 11   | [nn.MarginRankingLoss](https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html#torch.nn.MarginRankingLoss) |
| 12   | [nn.HingeEmbeddingLoss](https://pytorch.org/docs/stable/generated/torch.nn.HingeEmbeddingLoss.html#torch.nn.HingeEmbeddingLoss) |
| 13   | [nn.MultiLabelMarginLoss](https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelMarginLoss.html#torch.nn.MultiLabelMarginLoss) |
| 14   | [nn.HuberLoss](https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html#torch.nn.HuberLoss) |
| 15   | [nn.SmoothL1Loss](https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html#torch.nn.SmoothL1Loss) |
| 16   | [nn.SoftMarginLoss](https://pytorch.org/docs/stable/generated/torch.nn.SoftMarginLoss.html#torch.nn.SoftMarginLoss) |
| 17   | [nn.MultiLabelSoftMarginLoss](https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelSoftMarginLoss.html#torch.nn.MultiLabelSoftMarginLoss) |
| 18   | [nn.CosineEmbeddingLoss](https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html#torch.nn.CosineEmbeddingLoss) |
| 19   | [nn.MultiMarginLoss](https://pytorch.org/docs/stable/generated/torch.nn.MultiMarginLoss.html#torch.nn.MultiMarginLoss) |
| 20   | [nn.TripletMarginLoss](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html#torch.nn.TripletMarginLoss) |
| 21   | [nn.TripletMarginWithDistanceLoss](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginWithDistanceLoss.html#torch.nn.TripletMarginWithDistanceLoss) |

### 3. RNN和CNN

RNN（Recurrent Neural Network，循环神经网络）和CNN（Convolutional Neural Network，卷积神经网络）是深度学习中两个非常重要的神经网络模型。

RNN是一种用于处理序列数据的神经网络模型。它的特点是可以将前面的输入信息保存下来，并在后面的计算中进行利用，从而实现对序列数据的建模。RNN在自然语言处理、语音识别、股票预测等任务中广泛应用。

一些常见的序列数据：

- 文本数据：一段话或一篇文章中的单词或字符序列
- 时间序列数据：股票价格、气温、交通流量等随时间变化的数据
- 语音数据：音频信号中的时域或频域特征序列
- 生物信息学数据：DNA或RNA序列、蛋白质序列等
- 符号序列：编码信息的二进制序列、信号编码序列等

在这些序列数据中，每个数据点（单词、股票价格、音频帧等）都与序列中的其他数据点密切相关，因此需要使用序列模型（如RNN、LSTM等）进行处理和分析。

CNN是一种用于处理图像和空间数据的神经网络模型。它的主要特点是利用卷积操作提取图像中的特征，并通过池化操作减小特征图的大小，最终通过全连接层进行分类或回归。CNN在图像分类、目标检测、图像分割等任务中表现出色。

简单来说，RNN适用于序列数据处理，而CNN适用于图像和空间数据处理。但实际上，它们也可以互相组合使用，例如在图像描述生成任务中，可以使用CNN提取图像特征，然后使用RNN生成对应的文字描述。使用BaseNN搭建RNN和CNN模型的方式详见[add()详细](https://xedu.readthedocs.io/zh/latest/basenn/introduction.html#add)介绍。

### 4. 深度学习常见的数据类型

图像数据：图像数据是深度学习应用中最常见的数据类型之一。图像数据通常表示为多维数组，每个数组元素代表一个像素的值。深度学习应用中常使用的图像数据格式包括JPEG、PNG、BMP等。

文本数据：文本数据是指由字符组成的序列数据。在深度学习应用中，文本数据通常被表示为词向量或字符向量，用于输入到文本处理模型中。

特征数据：特征数据指的是表示对象或事物的特征的数据，通常用于机器学习和数据挖掘。特征数据可以是数值型、离散型或者是二进制的，用于描述对象或事物的各种属性和特征。特征数据可以是手动设计的、自动提取的或者是混合的。在机器学习中，特征数据通常作为模型的输入，用于预测目标变量或者分类。
