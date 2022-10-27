# BaseNN快速入门

## 简介

BaseNN可以方便地逐层搭建神经网路，深入探究网络原理。

## 安装

`pip install basenn` 或 `pip install BaseNN`


## 体验

运行demo/BaseNN_demo.py。

可以在命令行输入BaseNN查看安装的路径，在安装路径内，可以查看提供的更多demo案例。

## 训练

### 0.引入包

```python
from BaseNN import nn
```

### 1.声明模型

```python
model = nn()
```

### 2.载入数据

此处采用IRIS鸢尾花数据集作为示例。

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

### 3.搭建模型

逐层添加，搭建起模型结构。注释标明了数据经过各层的尺寸变化。

```python
model.add(layer='Linear',size=(4, 10),activation='ReLU') # [120, 10]
model.add(layer='Linear',size=(10, 5), activation='ReLU') # [120, 5]
model.add(layer='Linear', size=(5, 3), activation='Softmax') # [120, 3]
```

以上使用`add()`方法添加层，参数`layer='Linear'`表示添加的层是线性层，`size=(4,10)`表示该层输入维度为4，输出维度为10，`activation='ReLU'`表示使用ReLU激活函数。更详细`add()`方法使用可见附录1。

### 4.模型训练

模型训练可以采用以下函数：

```python
model.train(lr=0.01, epochs=500,checkpoint=checkpoint)
```

参数lr为学习率， epochs为训练轮数，checkpoint为现有模型路径，当使用checkpoint参数时，模型基于一个已有的模型继续训练，不使用checkpoint参数时，模型从零开始训练。

#### 4.1正常训练

```python
model = nn() 
model.add(layer='Linear',size=(4, 10),activation='ReLU') # [120, 10]
model.add(layer='Linear',size=(10, 5), activation='ReLU') # [120, 5]
model.add(layer='Linear', size=(5, 3), activation='Softmax') # [120, 3]
model.load_dataset(x, y)
model.save_fold = 'checkpoints'
model.train(lr=0.01, epochs=1000)
```

model.save_fold表示训练出的模型文件保存的文件夹。

#### 4.2 继续训练

```python
model = MMBase()
model.load_dataset(x, y)
model.save_fold = 'checkpoints'
checkpoint = 'checkpoints/mmbase_net.pkl'
model.train(lr=0.01, epochs=1000, checkpoint=checkpoint)
```

## 推理

### 使用现有模型直接推理

可使用以下函数进行推理：

```python
model.inference(data=test_x, checkpoint=checkpoint)
```

参数data为待推理的测试数据数据，该参数必须传入值；

checkpoint为已有模型路径，即使用现有的模型进行推理，该参数可以不传入值，即直接使用训练出的模型做推理。

```python
model = MMBase() # 声明模型
checkpoint = 'checkpoints/mmbase_net.pkl' # 现有模型路径
result = model.inference(data=test_x, checkpoint=checkpoint) # 直接推理
model.print_result() # 输出结果
```



### 输出推理结果

```python
res = model.inference(test_x)
```

输出结果数据类型为numpy的二维数组，表示各个样本的各个特征的置信度。

```python
model.print_result() # 输出字典格式结果
```

输出结果数据类型为字典，格式为{样本编号：{预测值：x，置信度：y}}。该函数调用即输出，但也有返回值。

### 模型的保存与加载

```python
# 保存
model.save("mmbase_net.pkl")
# 加载
model.load("mmbase_net.pkl")
```

参数为模型保存的路径，`.pkl`文件格式可以理解为将python中的数组、列表等持久化地存储在硬盘上的一种方式。

注：train()，inference()函数中也可通过参数控制模型的保存与加载，但这里也列出单独保存与加载模型的方法，以确保灵活性。

### 查看模型结构

```python
model.print_model()
```

无参数。



完整测试用例可见BaseNN_demo.py文件。

## 附录

#### 1. add()详细介绍

此处以典型的LeNet5网络结构为例。注释标明了数据经过各层的尺寸变化。

```python
model.add('Conv2D', size=(1, 3),kernel_size=( 3, 3)) # [100, 3, 18, 18]
model.add('MaxPool', kernel_size=(2,2), activation='ReLU') # [100, 3, 9, 9]
model.add('Conv2D', size=(3, 10), kernel_size=(3, 3)) # [100, 10, 7, 7]
model.add('AvgPool', kernel_size=(2,2), activation='ReLU') # [100, 10, 3, 3]
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

