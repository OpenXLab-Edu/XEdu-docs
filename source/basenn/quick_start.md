# 快速体验BaseNN，开始！

## 简介

BaseNN可以方便地逐层搭建神经网络，深入探究神经网络的原理。

## 安装

`pip install basenn` 或 `pip install BaseNN`

更新库文件：`pip install --upgrade BaseNN`


## 体验

运行demo/BaseNN_demo.py。

可以在命令行输入BaseNN查看安装的路径，在安装路径内，可以查看提供的更多demo案例。同时可查看附录。

如果在使用中出现类似报错：`**AttributeError**: partially initialized module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline' (most likely due to a circular import)` 

可尝试通过运行`pip install --upgrade opencv-python`解决

## 第一个BaseNN项目：搭建搭建鸢尾花分类模型

### 第0步 引入包

```python
# 导入BaseNN库、numpy库，numpy库用于数据处理
from BaseNN import nn
import numpy as np
```

### 第1步 声明模型

```python
model = nn()
```

### 第2步 载入数据

```python
train_path = 'data/iris_training.csv'
model.load_tab_data(train_path, batch_size=120)
```

### 第3步 搭建模型

逐层添加，搭建起模型结构。注释标明了数据经过各层的维度变化。

```python
model.add(layer='linear',size=(4, 10),activation='relu') # [120, 10]
model.add(layer='linear',size=(10, 5), activation='relu') # [120, 5]
model.add(layer='linear', size=(5, 3), activation='softmax') # [120, 3]
```

以上使用`add()`方法添加层，参数`layer='linear'`表示添加的层是线性层，`size=(4,10)`表示该层输入维度为4，输出维度为10，`activation='relu'`表示使用ReLU激活函数。

### 第4步 模型训练

模型训练可以采用以下函数：

```python
# 设置模型保存的路径
model.save_fold = 'checkpoints/iris_ckpt'
# 模型训练
model.train(lr=0.01, epochs=1000)
```

也可以使用继续训练：

```
checkpoint = 'checkpoints/basenn.pth'
model.train(lr=0.01, epochs=1000, checkpoint=checkpoint)
```

参数`lr`为学习率， `epochs`为训练轮数，`checkpoint`为现有模型路径，当使用`checkpoint`参数时，模型基于一个已有的模型继续训练，不使用`checkpoint`参数时，模型从零开始训练。

### 第5步 模型测试

用测试数据查看模型效果。

```python
# 用测试数据查看模型效果
model2 = nn()
test_path = 'data/iris_test.csv'
test_x = np.loadtxt(test_path, dtype=float, delimiter=',',skiprows=1,usecols=range(0,4)) 
res = model2.inference(test_x, checkpoint="checkpoints/iris_ckpt/basenn.pth")
model2.print_result(res)

# 获取最后一列的真实值
test_y = np.loadtxt(test_path, dtype=float, delimiter=',',skiprows=1,usecols=4) 
# 定义一个计算分类正确率的函数
def cal_accuracy(y, pred_y):
    res = pred_y.argmax(axis=1)
    tp = np.array(y)==np.array(res)
    acc = np.sum(tp)/ y.shape[0]
    return acc

# 计算分类正确率
print("分类正确率为：",cal_accuracy(test_y, res))
```

用某组测试数据查看模型效果。

```python
# 用某组测试数据查看模型效果
data = np.array([test_x[0]])
checkpoint = 'checkpoints/iris_ckpt/basenn.pth'
res = model.inference(data=data, checkpoint=checkpoint)
model.print_result(res) # 输出字典格式结果
```

参数`data`为待推理的测试数据数据，该参数必须传入值；

`checkpoint`为已有模型路径，即使用现有的模型进行推理。

## 快速体验

体验BaseNN的最快速方式是通过OpenInnoLab平台。

OpenInnoLab平台为上海人工智能实验室推出的青少年AI学习平台，满足青少年的AI学习和创作需求，支持在线编程。在“项目”中查看更多，搜索”BaseNN“，即可找到所有与BaseNN相关的体验项目。

AI项目工坊：[https://www.openinnolab.org.cn/pjlab/projects/list?backpath=/pjlab/ai/projects](https://www.openinnolab.org.cn/pjlab/projects/list?backpath=/pjlab/ai/projects)

（用Chrome浏览器打开效果最佳）

用BaseNN库搭建搭建鸢尾花分类模型项目地址：[https://www.openinnolab.org.cn/pjlab/project?id=641bc2359c0eb14f22fdbbb1&sc=635638d69ed68060c638f979#public](https://www.openinnolab.org.cn/pjlab/project?id=641bc2359c0eb14f22fdbbb1&sc=635638d69ed68060c638f979#public)

## 挑战使用BaseNN完成第一个回归项目：波士顿房价预测

波士顿房价数据集（Boston Housing Dataset）是一个著名的数据集，经常用于机器学习和统计分析中。该数据集包含波士顿郊区房屋的各种信息，包括房价和与房价可能相关的各种属性。选择了四个与房价关系较大的特征：RM (每栋住宅的平均房间数)、LSTAT (人口中较低地位的百分比)、PTRATIO (师生比例)、NOX (一氧化氮浓度) 。进行数据预处理后生成了已提取出只有这四列特征和预测值且做了归一化处理的训练集（house_price_data_norm_train.csv）、验证集（house_price_data_norm_val.csv），搭建模型进行训练，数据预处理的代码可参考原项目。

项目地址：

[https://www.openinnolab.org.cn/pjlab/project?id=656d99e87e42e551fa5f89bd&sc=62f34141bf4f550f3e926e0e#public](https://www.openinnolab.org.cn/pjlab/project?id=656d99e87e42e551fa5f89bd&sc=62f34141bf4f550f3e926e0e#public)

（用Chrome浏览器打开效果最佳）

### 第0步 引入包

```
# 导入库
from BaseNN import nn
```

### 第1步 声明模型

```python
# 声明模型，选择回归任务
model = nn('reg') 
```

### 第2步 载入数据

```
model.load_tab_data('house_price_data_norm_train.csv',batch_size=1024) # 载入数据
```

### 第3步 搭建一个3层的全连接神经网络

```
model.add('Linear', size=(4, 64),activation='ReLU')  
model.add('Linear', size=(64, 4), activation='ReLU') 
model.add('Linear', size=(4, 1))
model.add(optimizer='Adam')
```

### 第4步 模型训练

```
# 设置模型保存的路径
model.save_fold = 'checkpoints/ckpt'
model.train(lr=0.008, epochs=5000,loss='MSELoss') # 训练
```

### 第5步 模型测试

此步骤可以借助验证集完成。

读取数据。

```
import numpy as np
# 读取验证集
val_path = 'house_price_data_norm_val.csv'
val_x = np.loadtxt(val_path, dtype=float, delimiter=',',skiprows=1,usecols=range(0,4)) # 读取特征列
val_y = np.loadtxt(val_path, dtype=float, delimiter=',',skiprows=1,usecols=4) # 读取第4列
```

模型推理。

```
# 导入库
from BaseNN import nn
# 声明模型
model = nn('reg') 
y_pred = model.inference(val_x,checkpoint = 'checkpoints/ckpt2/basenn.pth')  # 对该数据进行预测
```

绘制曲线图。

```
# 绘制真实数据和预测比较曲线
import matplotlib.pyplot as plt
plt.plot(val_y, label='val')
plt.plot(y_pred, label='pred')
plt.legend()
plt.show()
```

对比输出，查看回归的效果，觉得效果还是很不错的。

![](../images/basenn/huigui.png)

## 挑战使用BaseNN完成第一个自然语言处理项目：自动写诗机

### 第0步 引入包

```python
# 导入BaseNN库、numpy库，numpy库用于数据处理
from BaseNN import nn
import numpy as np
```

### 第1步 声明模型

```python
model = nn()
```

### 第2步 载入数据

tangccc.npz是本项目的文本数据，源于互联网，包括57580首唐诗。npz是一种用于存储NumPy数组数据的文件格式。

npz文件是一种方便的方式来保存和加载NumPy数组，通常用于在不同的Python程序之间或不同的计算环境中共享数据。

在该项目中可以使用`load_npz_data()`方法直接读取npz格式的数据到模型中

```python
model.load_npz_data('tangccc.npz')
```

### 第3步 搭建LSTM模型

搭建模型只需加入em_lstm层即可，其他层会自适应补充，其中num_layers参数为循环神经网络循环的次数。

em_LSTM由包括embedding层，LSTM层和线性层组成，因为有embedding层的加入，所以em_LSTM可以专门处理文本数据。

```python
model.add('em_lstm', size=(128,256),num_layers=2) 
```

### 第4步 模型训练

为了节省训练时间，可以选择继续训练。

```python
checkpoint = 'model.pth'
model.save_fold = 'checkpoints'
model.train(lr=0.005, epochs=1,batch_size=16, checkpoint=checkpoint)
```

### 第5步 模型测试

可以输入一个字输出下一个字。

```
input = '长'
checkpoint = 'model.pth'
result = model.inference(data=input,checkpoint=checkpoint) # output是多维向量，接下来转化为汉字
output = result[0]
print("output: ",output)
index = np.argmax(output) # 找到概率最大的字的索引
w = model.ix2word[index] # 根据索引从词表中找到字
print("word:",w)
```

### 拓展

可以使用训练好的模型生成唐诗，生成藏头诗，做各种有意思的应用。

更多内容详见用BaseNN实现自动写诗机项目，项目地址：[https://www.openinnolab.org.cn/pjlab/project?id=641c00bbba932064ea962783&sc=635638d69ed68060c638f979#public](https://www.openinnolab.org.cn/pjlab/project?id=641c00bbba932064ea962783&sc=635638d69ed68060c638f979#public)

## 附录

### 体验案例1. 搭建卷积神经网络实现手写体分类

本案例来源于《人工智能初步》人教地图72页。

项目地址：[https://www.openinnolab.org.cn/pjlab/project?id=641d17e67c99492cf16d706f&sc=635638d69ed68060c638f979#public](https://www.openinnolab.org.cn/pjlab/project?id=641d17e67c99492cf16d706f&sc=635638d69ed68060c638f979#public)

实现效果：

![](../images/basenn/six.png)

#### 实现步骤：

##### 1）网络搭建和模型训练

导入库：

```
# 导入BaseNN库
from BaseNN import nn
```

读取数据：

```
# 模型载入数据
model.load_img_data("/data/MELLBZ/mnist/training_set",color="grayscale",batch_size=10000)
```

搭建网络开始训练：

```
# 声明模型
model = nn()
# 自己搭建网络（我们搭建的是LeNet网络，可改变参数搭建自己的网络）
model.add('Conv2D', size=(1, 6),kernel_size=(5, 5), activation='ReLU') 
model.add('MaxPool', kernel_size=(2,2)) 
model.add('Conv2D', size=(6, 16), kernel_size=(5, 5), activation='ReLU')
model.add('MaxPool', kernel_size=(2,2)) 
model.add('Linear', size=(256, 120), activation='ReLU') 
model.add('Linear', size=(120, 84), activation='ReLU') 
model.add('Linear', size=(84, 10), activation='Softmax') 

# 模型超参数设置和网络训练
model.optimizer = 'Adam' #'SGD' , 'Adam' , 'Adagrad' , 'ASGD' 内置不同优化器
learn_rate = 0.001 #学习率
max_epoch = 100 # 最大迭代次数
model.save_fold = 'mn_ckpt' # 模型保存路径
model.train(lr=learn_rate, epochs=max_epoch) # 直接训练
```

##### 2）模型推理

读取某张图片进行推理：

```
# 单张图片的推理
path = 'test_IMG/single_data.jpg'
checkpoint = 'mn_ckpt/basenn.pth' # 现有模型路径
y_pred = model.inference(data=path, checkpoint=checkpoint)
model.print_result()

# 输出结果
res = y_pred.argmax(axis=1)
print('此手写体的数字是：',res[0])
```

定义一个准确率计算函数，读取测试集所有图片进行推理并计算准确率。

```
# 计算准确率函数
def cal_accuracy(y, pred_y):
    res = pred_y.argmax(axis=1)
    tp = np.array(y)==np.array(res)
    acc = np.sum(tp)/ y.shape[0]
    return acc

import torch
from BaseNN import nn
import numpy as np
# 推理验证集
m = nn()
val_data = m.load_img_data('/data/MELLBZ/mnist/val_set',color="grayscale",batch_size=20000)
checkpoint_path = 'mn_ckpt/basenn.pth' # 载入模型

for x, y in val_data:
    res = m.inference(x, checkpoint=checkpoint_path)
    acc=cal_accuracy(y,res)
    print('验证集准确率: {:.2f}%'.format(100.0 * acc))
```



### 体验案例2. 一维卷积神经网络文本情感识别

本案例来源于《人工智能初步》人教地图版72-76页。

项目地址：[https://www.openinnolab.org.cn/pjlab/project?id=638d8bd8be5e9c6ce28ad033&sc=635638d69ed68060c638f979#public](https://www.openinnolab.org.cn/pjlab/project?id=638d8bd8be5e9c6ce28ad033&sc=635638d69ed68060c638f979#public)

#### 项目核心功能：

完成了搭建一维卷积神经网络实现文本感情识别分类，代码使用BaseNN库实现，同时结合了Embedding层对单词文本进行向量化。

数据集是imdb电影评论和情感分类数据集，来自斯坦福AI实验室平台，[http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/)。

**注意**：新版本BaseNN（>==0.1.6）已不支持项目中部分代码的写法或，如添加Embedding层。可模仿下列代码进行

#### 实现步骤：

##### 1）网络搭建和模型训练

导入库：

```
# 导入BaseNN库、numpy库用于数据处理
from BaseNN import nn
import numpy as np
```

读取数据并载入：

```
# 读取训练集数据
train_data = np.loadtxt('imdb/train_data.csv', delimiter=",")
train_label = np.loadtxt('imdb/train_label.csv', delimiter=",")
# 模型载入数据
model.load_dataset(train_data, train_label) 
```

搭建模型并开始训练：

```
# 声明模型
model = nn() # 有Embedding层
# 搭建模型
model.add('Embedding', vocab_size = 10000, embedding_dim = 32)  # Embedding层，对实现文本任务十分重要，将one-hot编码转化为相关向量 输入大小（batch_size,512）输出大小（batch_size,32,510）
model.add('conv1d', size=(32, 32),kernel_size=3, activation='relu') #一维卷积 输入大小（batch_size,32,510） 输出大小（batch_size,32,508）
model.add('conv1d', size=(32, 64),kernel_size=3, activation='relu') #一维卷积 输入大小（batch_size,32,508） 输出大小（batch_size,64,506）
model.add('mean') #全局池化 输入大小（batch_size,64,508）输出大小（batch_size,64）
model.add('linear', size=(64, 128), activation='relu') #全连接层 输入大小（batch_size,64）输出大小（batch_size,128）
model.add('linear', size=(128, 2), activation='softmax') #全连接层 输入大小（batch_size,128）输出大小（batch_size,2）

# 模型超参数设置和网络训练（训练时间较长, 可调整最大迭代次数减少训练时间）
model.add(optimizer='Adam') #'SGD' , 'Adam' , 'Adagrad' , 'ASGD' 内置不同优化器
learn_rate = 0.001 #学习率
max_epoch = 150 # 最大迭代次数
model.save_fold = 'mn_ckpt' # 模型保存路径
checkpoint = 'mn_ckpt/cov_basenn.pkl' 
model.train(lr=learn_rate, epochs=max_epoch) # 直接训练
```

##### 2）模型推理

读取测试集所有数据进行推理：

```
#读取测试集数据
test_data = np.loadtxt('imdb/test_data.csv', delimiter=",")
test_label = np.loadtxt('imdb/test_label.csv', delimiter=",")
y_pred = model.inference(data=train_data)
```

用单个数据进行推理：

```
# 用测试集单个数据查看模型效果
single_data = np.loadtxt('imdb/test_data.csv', delimiter=",", max_rows = 1)
single_label = np.loadtxt('imdb/test_label.csv', delimiter=",", max_rows = 1)
label = ['差评','好评']
single_data = single_data.reshape(1,512) 
res = model.inference(data=single_data)
res = res.argmax(axis=1)
print('评论对电影的评价是：', label[res[0]]) # 该评论文本数据可见single_data.txt
```

### 体验案例3. 用神经网络计算前方障碍物方向

本案例是一个跨学科项目，用神经网络来拟合三角函数。案例发表于2023年的《中国信息技术教育》杂志。

项目地址：[https://www.openinnolab.org.cn/pjlab/project?id=6444992a06618727bed5a67c&backpath=/pjlab/projects/list#public](https://www.openinnolab.org.cn/pjlab/project?id=6444992a06618727bed5a67c&backpath=/pjlab/projects/list#public)

#### 项目核心功能：

用两个超声波传感器测量前方的障碍物距离，然后计算出障碍物所在的方向。这是一个跨学科项目，用神经网络来拟合三角函数。训练一个可以通过距离计算出坐标的神经网络模型，掌握使用BaseNN库搭建神经网络完成“回归”任务的流程。

#### 实现步骤：

##### 1）数据采集

我们有多种方式来采集数据。第一种是最真实的，即在障碍物和中点之间拉一条 线，然后读取两个超声波传感器的数据，同时测量角度并记录。另一种是拉三条线， 因为超声波传感器的数值和真实长度误差是很小的。 当然，因为这一角度是可以用三角函数计算的，那么最方面的数据采集方式莫过于是用Python写一段代码，然后将一组数据输出到CSV 文件中。或者使用Excel的公式来计算，再导出关键数据，如图所示。

![](../images/basenn/excel.png)

##### 2）数据预处理

首先读取数据，0-2为输入，3-9是各种输出的数据。

```
import numpy as np
train_path = './data/train-full.csv'
x = np.loadtxt(train_path, dtype=float, delimiter=',',skiprows=1,usecols=[0,1,2]) # 读取前3列
y = np.loadtxt(train_path, dtype=float, delimiter=',',skiprows=1,usecols=[8]) # 读取9列
```

将y映射到0-1之间。

```
from sklearn.preprocessing import MinMaxScaler
y = y.reshape(-1, 1)
scaler = MinMaxScaler()
scaler.fit(y)
y = scaler.transform(y)  # 0~1
```

生成新的数据集。

```
norm_data = np.concatenate((x,y),axis=1)
np.savetxt('./data/train_norm.csv',norm_data,delimiter=',')
```

##### 3）网络搭建和模型训练

搭建一个3层的神经网络并开始训练，输入维度是3（3列数据），最后输出维度是1（1列数据），激活函数使用ReLU。

```
from BaseNN import nn
model = nn('reg') #声明模型 
model.load_tab_data('./data/train_norm.csv',batch_size=1024) # 载入数据
model.add('Linear', size=(3, 60),activation='ReLU')  
model.add('Linear', size=(60, 6), activation='ReLU') 
model.add('Linear', size=(6, 1))
model.add(optimizer='Adam')

# 设置模型保存的路径
model.save_fold = 'checkpoints/ckpt'
# 模型训练
model.train(lr=0.001, epochs=300,loss='MSELoss') 
```

##### 4）模型推理

读取测试数据进行模型推理，测试数据同样来自随机数。

```
# 测试数据
test_path = './data/test-full.csv'
test_x = np.loadtxt(test_path, dtype=float, delimiter=',',skiprows=1,usecols=[0,1,2]) # 读取前3列
test_y = np.loadtxt(test_path, dtype=float, delimiter=',',skiprows=1,usecols=[8]) # 读取第9列
y_pred = model.inference(test_x,checkpoint = 'checkpoints/ckpt/basenn.pth')  # 对该数据进行预测
```

