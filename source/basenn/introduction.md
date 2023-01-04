# BaseNN功能详解

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

此处采用lvis鸢尾花数据集和MNIST手写体数据集作为示例。

读取并载入鸢尾花数据：

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

读取并载入手写体数据：

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

逐层添加，搭建起模型结构。注释标明了数据经过各层的尺寸变化。

```python
model.add(layer='Linear',size=(4, 10),activation='ReLU') # [120, 10]
model.add(layer='Linear',size=(10, 5), activation='ReLU') # [120, 5]
model.add(layer='Linear', size=(5, 3), activation='Softmax') # [120, 3]
```

以上使用`add()`方法添加层，参数`layer='Linear'`表示添加的层是线性层，`size=(4,10)`表示该层输入维度为4，输出维度为10，`activation='ReLU'`表示使用ReLU激活函数。更详细`add()`方法使用可见附录1。

### 4. 模型训练

模型训练可以采用以下函数：

```python
model.train(lr=0.01, epochs=500)
```

参数`lr`为学习率，`epochs`为训练轮数。

#### 4.1 正常训练

```python
model = nn() 
model.add(layer='Linear',size=(4, 10),activation='ReLU') # [120, 10]
model.add(layer='Linear',size=(10, 5), activation='ReLU') # [120, 5]
model.add(layer='Linear', size=(5, 3), activation='Softmax') # [120, 3]
model.load_dataset(x, y)
model.save_fold = 'checkpoints'
model.train(lr=0.01, epochs=1000)
```

`model.save_fold`表示训练出的模型文件保存的文件夹。

#### 4.2 继续训练

```python
model = nn()
model.load_dataset(x, y)
model.save_fold = 'checkpoints'
checkpoint = 'checkpoints/basenn.pkl'
model.train(lr=0.01, epochs=1000, checkpoint=checkpoint)
```

`checkpoint`为现有模型路径，当使用`checkpoint`参数时，模型基于一个已有的模型继续训练，不使用`checkpoint`参数时，模型从零开始训练。

### 5. 使用现有模型直接推理

可使用以下函数进行推理：

```python
model.inference(data=test_x, checkpoint=checkpoint)
```

参数`data`为待推理的测试数据数据，该参数必须传入值；

`checkpoint`为已有模型路径，即使用现有的模型进行推理，该参数可以不传入值，即直接使用训练出的模型做推理。

```python
model = nn() # 声明模型
checkpoint = 'checkpoints/basenn.pkl' # 现有模型路径
result = model.inference(data=test_x, checkpoint=checkpoint) # 直接推理
model.print_result() # 输出结果
```

### 6. 输出推理结果

```python
res = model.inference(test_x)
```

输出结果数据类型为`numpy`的二维数组，表示各个样本的各个特征的置信度。

```python
model.print_result() # 输出字典格式结果
```

输出结果数据类型为字典，格式为{样本编号：{预测值：x，置信度：y}}。该函数调用即输出，但也有返回值。

### 7. 模型的保存与加载

```python
# 保存
model.save_fold = 'mn_ckpt'
# 加载
model.load("basenn.pkl")
```

参数为模型保存的路径，模型权重文件格式为`.pkl`文件格式，此格式可以理解为将python中的数组、列表等持久化地存储在硬盘上的一种方式。

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
model.load('mn_ckpt/basenn.pkl')          # 保存的已训练模型载入
path = 'test_IMG/single_data.jpg'
img = cv2.imread(path,flags = 0)          # 图片数据读取
model.visual_feature(img,in1img = True)   # 特征的可视化
```

如输入数据为一维数据，指定数据和已经训练好的模型，可生成一个txt文件展示经过各层后的输出。

```
import numpy as np
from BaseNN import nn
model = nn()
model.load('checkpoints/iris_ckpt/basenn.pkl')          # 保存的已训练模型载入
data = np.array(test_x[0]) # 指定数据,如测试数据的一行
model.visual_feature(data)   # 特征的可视化
```

### 10. 指定随机数种子（选）

默认初始化是随机的，每次训练结果都不一样。可以可使用`set_seed()`函数设定随机数种子，使得训练结果可被其他人复现。一旦指定，则每次训练结果一致。使用方法如下：

```Shell
model = nn()
model.set_seed(1235)
model.add(...)
...
model.train(...)
```

注：设定随机数种子`set_seed()`应当在搭建网络`add()`之前。

### 11. 指定损失函数（选）

默认的损失函数是交叉熵损失函数，允许选择不同的损失函数，支持的损失函数见附录。自选损失函数方法如下：

```
model.train(...,loss="CrossEntropyLoss")
```

### 12. 指定评价指标（选）

默认的默认为准确率，允许选择其他的评价指标。支持的评价指标：acc（准确率），mae（平均绝对误差），mse（均方误差）。

自选评价指标方法如下：

```
model.train(...,metrics=["mse"])
```

因此针对不同的分类或回归任务，可指定不同的损失函数和评价指标。

例：

回归：`model.train(...,loss="SmoothL1Loss", metrics=["mae"])` 

分类：`model.train(...,loss="CrossEntropyLoss",metrics=["acc"])`

## 附录

#### 1. add()详细介绍

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

#### 2. 支持的损失函数

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
