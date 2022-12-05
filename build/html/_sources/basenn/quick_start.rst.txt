BaseNN快速入门
==============

简介
----

BaseNN可以方便地逐层搭建神经网路，深入探究网络原理。

安装
----

``pip install basenn`` 或 ``pip install BaseNN``

体验
----

运行demo/BaseNN_demo.py。

可以在命令行输入BaseNN查看安装的路径，在安装路径内，可以查看提供的更多demo案例。同时可查看附录。

训练
----

0.引入包
~~~~~~~~

.. code:: python

   from BaseNN import nn

1.声明模型
~~~~~~~~~~

.. code:: python

   model = nn()

2.载入数据
~~~~~~~~~~

此处采用lvis鸢尾花数据集和MNIST手写体数据集作为示例。

读取并载入鸢尾花数据：

.. code:: python

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

读取并载入手写体数据：

.. code:: python

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

3.搭建模型
~~~~~~~~~~

逐层添加，搭建起模型结构。注释标明了数据经过各层的尺寸变化。

.. code:: python

   model.add(layer='Linear',size=(4, 10),activation='ReLU') # [120, 10]
   model.add(layer='Linear',size=(10, 5), activation='ReLU') # [120, 5]
   model.add(layer='Linear', size=(5, 3), activation='Softmax') # [120, 3]

以上使用\ ``add()``\ 方法添加层，参数\ ``layer='Linear'``\ 表示添加的层是线性层，\ ``size=(4,10)``\ 表示该层输入维度为4，输出维度为10，\ ``activation='ReLU'``\ 表示使用ReLU激活函数。

4.模型训练
~~~~~~~~~~

模型训练可以采用以下函数：

.. code:: python

   model.train(lr=0.01, epochs=500,checkpoint=checkpoint)

参数\ ``lr``\ 为学习率，
``epochs``\ 为训练轮数，\ ``checkpoint``\ 为现有模型路径，当使用\ ``checkpoint``\ 参数时，模型基于一个已有的模型继续训练，不使用\ ``checkpoint``\ 参数时，模型从零开始训练。

4.1正常训练
^^^^^^^^^^^

.. code:: python

   model = nn() 
   model.add(layer='Linear',size=(4, 10),activation='ReLU') # [120, 10]
   model.add(layer='Linear',size=(10, 5), activation='ReLU') # [120, 5]
   model.add(layer='Linear', size=(5, 3), activation='Softmax') # [120, 3]
   model.load_dataset(x, y)
   model.save_fold = 'checkpoints'
   model.train(lr=0.01, epochs=1000)

``model.save_fold``\ 表示训练出的模型文件保存的文件夹。

4.2 继续训练
^^^^^^^^^^^^

.. code:: python

   model = nn()
   model.load_dataset(x, y)
   model.save_fold = 'checkpoints'
   checkpoint = 'checkpoints/basenn.pkl'
   model.train(lr=0.01, epochs=1000, checkpoint=checkpoint)

推理
----

使用现有模型直接推理
~~~~~~~~~~~~~~~~~~~~

可使用以下函数进行推理：

.. code:: python

   model.inference(data=test_x, checkpoint=checkpoint)

参数\ ``data``\ 为待推理的测试数据数据，该参数必须传入值；

``checkpoint``\ 为已有模型路径，即使用现有的模型进行推理，该参数可以不传入值，即直接使用训练出的模型做推理。

.. code:: python

   model = nn() # 声明模型
   checkpoint = 'checkpoints/basenn.pkl' # 现有模型路径
   result = model.inference(data=test_x, checkpoint=checkpoint) # 直接推理
   model.print_result() # 输出结果

输出推理结果
~~~~~~~~~~~~

.. code:: python

   res = model.inference(test_x)

输出结果数据类型为\ ``numpy``\ 的二维数组，表示各个样本的各个特征的置信度。

.. code:: python

   model.print_result() # 输出字典格式结果

输出结果数据类型为字典，格式为{样本编号：{预测值：x，置信度：y}}。该函数调用即输出，但也有返回值。

模型的保存与加载
~~~~~~~~~~~~~~~~

.. code:: python

   # 保存
   model.save_fold = 'mn_ckpt'
   # 加载
   model.load("basenn.pkl")

参数为模型保存的路径，模型权重文件格式为\ ``.pkl``\ 文件格式，此格式可以理解为将python中的数组、列表等持久化地存储在硬盘上的一种方式。

注：\ ``train()``\ ，\ ``inference()``\ 函数中也可通过参数控制模型的保存与加载，但这里也列出单独保存与加载模型的方法，以确保灵活性。

查看模型结构
~~~~~~~~~~~~

.. code:: python

   model.print_model()

无参数。

完整测试用例可见BaseNN_demo.py文件。

快速体验
--------

体验BaseNN的最快速方式是通过OpenInnoLab平台。

OpenInnoLab平台为上海人工智能实验室推出的青少年AI学习平台，满足青少年的AI学习和创作需求，支持在线编程。在“AI项目工坊
- 人工智能工坊”中，查找”BaseNN“，即可找到所有与BaseNN相关的体验项目。

AI项目工坊：https://www.openinnolab.org.cn/pjLab/projects/channel（用Chorm浏览器打开效果最佳）

附录
----

案例1. 搭建卷积神经网络实现手写体分类
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

本案例来源于《人工智能初步》人教地图72页。

**项目核心功能和实现效果展示**\ ：

使用BaseNN库实现卷积神经网络搭建，完成手写图分类，数据集为MNIST数据集。

.. figure:: https://www.openinnolab.org.cn/webdav/635638d69ed68060c638f979/638028ff777c254264da4e6f/current/assets/%E7%94%A8%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%AE%9E%E7%8E%B0%E6%89%8B%E5%86%99%E4%BD%93%E5%88%86%E7%B1%BB%E9%A1%B9%E7%9B%AE%E6%95%88%E6%9E%9C%E5%9B%BE%E7%89%87.PNG
   :alt: 用卷积神经网络实现手写体分类项目效果图片.PNG



**实现步骤：**

.. _模型训练-1:

1）模型训练
'''''''''''

从零开始训练

::

   # 导入BaseNN库、os、cv2、numpy库，os、cv2、numpy库用于数据处理
   from BaseNN import nn
   import os
   import cv2
   import numpy as np

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
               img = cv2.resize(img,(32,32))
               imGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
               # print(img)
               data.append(imGray)
               label.append(int(item))
       x = np.array(data)
       y = np.array(label)
    
       x = np.expand_dims(x, axis=1)
       return x, y

   # 读取训练数据
   train_x, train_y = read_data('/data/QX8UBM/mnist_sample/training_set')
   # 声明模型
   model = nn()
   # 载入数据
   model.load_dataset(train_x, train_y) 

   # 搭建模型
   model.add('Conv2D', size=(1, 6),kernel_size=( 5, 5), activation='ReLU') 
   model.add('AvgPool', kernel_size=(2,2)) 
   model.add('Conv2D', size=(6, 16), kernel_size=(5, 5), activation='ReLU')
   model.add('AvgPool', kernel_size=(2,2)) 
   model.add('Linear', size=(400, 120), activation='ReLU') 
   model.add('Linear', size=(120, 84), activation='ReLU') 
   model.add('Linear', size=(84, 10), activation='Softmax')
   model.add(optimizer='SGD') # 设定优化器

   # 设置模型保存的路径
   model.save_fold = 'checkpoints/mn_ckpt1'
   # 模型训练
   model.train(lr=0.01, epochs=30)

继续训练：

::

   # 继续训练
   model = nn()
   model.load_dataset(train_x, train_y) 
   model.save_fold = 'checkpoints/mn_ckpt2' # 设置模型保存的新路径
   checkpoint = 'checkpoints/mn_ckpt1/basenn.pkl'
   model.train(lr=0.01, epochs=20, checkpoint=checkpoint)

2）模型推理
'''''''''''

读取测试集所有图片进行推理：

::

   # 用测试集查看模型效果
   test_x, test_y = read_data('/data/QX8UBM/mnist_sample/test_set') # 读取测试集数据
   res = model.inference(data=test_x)
   model.print_result(res) # 输出字典格式结果

读取某张图片进行推理：

::

   # 用测试集某张图片查看模型效果
   img = '/data/QX8UBM/mnist_sample/test_set/0/0.jpg' # 指定一张图片
   data = []
   im = cv2.imread(img)
   im = cv2.resize(im,(32,32))
   imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
   data.append(imGray)
   x = np.array(data)
   x = np.expand_dims(x, axis=1)
   result = model.inference(data=x)
   model.print_result(result) # 输出字典格式结果

案例2. 一维卷积神经网络文本情感识别
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

本案例来源于《人工智能初步》人教地图版72-76页。

**项目核心功能**\ ：

完成了搭建一维卷积神经网络实现文本感情识别分类，代码使用BaseNN库实现，同时结合了Embedding层对单词文本进行向量化。

数据集是imdb电影评论和情感分类数据集，来自斯坦福AI实验室平台，http://ai.stanford.edu/~amaas/data/sentiment/。

**实现步骤：**

.. _模型训练-2:

1）模型训练
'''''''''''

::

   # 导入BaseNN库、numpy库用于数据处理
   from BaseNN import nn
   import numpy as np
   # 声明模型
   model = nn() # 有Embedding层
   # 读取训练集数据
   train_data = np.loadtxt('imdb/train_data.csv', delimiter=",")
   train_label = np.loadtxt('imdb/train_label.csv', delimiter=",")
   # 模型载入数据
   model.load_dataset(train_data, train_label) 

   # 搭建模型
   model.add('Embedding', vocab_size = 10000, embedding_dim = 32)  # Embedding层，对实现文本任务十分重要，将one-hot编码转化为相关向量 输入大小（batch_size,512）输出大小（batch_size,32,510）
   model.add('Conv1D', size=(32, 32),kernel_size=3, activation='ReLU') #一维卷积 输入大小（batch_size,32,510） 输出大小（batch_size,32,508）
   model.add('Conv1D', size=(32, 64),kernel_size=3, activation='ReLU') #一维卷积 输入大小（batch_size,32,508） 输出大小（batch_size,64,506）
   model.add('Mean') #全局池化 输入大小（batch_size,64,508）输出大小（batch_size,64）
   model.add('Linear', size=(64, 128), activation='ReLU') #全连接层 输入大小（batch_size,64）输出大小（batch_size,128）
   model.add('Linear', size=(128, 2), activation='softmax') #全连接层 输入大小（batch_size,128）输出大小（batch_size,2）

   # 模型超参数设置和网络训练（训练时间较长, 可调整最大迭代次数减少训练时间）
   model.add(optimizer='Adam') #'SGD' , 'Adam' , 'Adagrad' , 'ASGD' 内置不同优化器
   learn_rate = 0.001 #学习率
   max_epoch = 150 # 最大迭代次数
   model.save_fold = 'mn_ckpt' # 模型保存路径
   checkpoint = 'mn_ckpt/cov_basenn.pkl' 
   model.train(lr=learn_rate, epochs=max_epoch) # 直接训练

.. _模型推理-1:

2）模型推理
'''''''''''

读取测试集所有数据进行推理：

::

   #读取测试集数据
   test_data = np.loadtxt('imdb/test_data.csv', delimiter=",")
   test_label = np.loadtxt('imdb/test_label.csv', delimiter=",")
   y_pred = model.inference(data=train_data)

用单个数据进行推理：

::

   # 用测试集单个数据查看模型效果
   single_data = np.loadtxt('imdb/test_data.csv', delimiter=",", max_rows = 1)
   single_label = np.loadtxt('imdb/test_label.csv', delimiter=",", max_rows = 1)
   label = ['差评','好评']
   single_data = single_data.reshape(1,512) 
   res = model.inference(data=single_data)
   res = res.argmax(axis=1)
   print('评论对电影的评价是：', label[res[0]]) # 该评论文本数据可见single_data.txt
