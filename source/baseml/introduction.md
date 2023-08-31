# BaseML功能详解

我们的传统机器学习（Mechine Learning）有很多算法，但总的来说，可以分为三大类：分类、回归和聚类。BaseML和sklearn不同之处，也就体现于此，sklearn尽管在拟合、预测等函数上对各模块做了统一，但并没有明确指出这样的三大类划分方式。这三类也有着特有的数据输入格式。除此之外，BaseLM还提供了`DimentionReduction`数据降维模块，用以对数据进行降维处理。

## 分类任务

### 0. 引入包

```
from BaseML import Classification as cls
```

### 1. 实例化

#### 贝叶斯分类

贝叶斯分类算法常用于解决不确定问题，如人们普遍认为夜里下雨，第二天早晨草地会湿，实际到了早上草地可能就干了，也许是因为风的因素，解决这类问题往往需要根据人类已有的经验来计算某种状态出现的概率，这种方式叫做贝叶斯推理。 贝叶斯分类算法是基于贝叶斯定理的一种算法，即“简单”地假设每对特征之间相互独立。

贝叶斯定理：P(A|B)表示事件B发生的条件下事件A发生的概率，P(A|B)等于事件A发生的条件下事件B发生的概率乘以事件A发生的概率P(A)，再除以事件B发生的概率P(B)。

```
# 实例化模型，模型名称选择NaiveBayes
model=cls('NaiveBayes')
```

#### 决策树分类

决策树（Decision Trees）是一种用来分类（Classification）和 回归（Regression）的算法。其目的是创建一种模型从数据特征中学习简单的决策规则来预测一个目标变量的值。优势在于便于理解和解释。树的结构可以可视化出来，训练需要的数据少。

```
# 实例化模型，模型名称选择CART（Classification and Regression Trees）
model=cls('CART')
```

#### k近邻

k近邻（K Nearest Neighbors）算法，简称为kNN，该算法以每一个测试数据为中心，根据在特征空间中与测试数据相邻的训练数据的标签来确定测试数据的标签。给定测试样本，基于特定的某种距离度量方式找到与训练集中最接近的k个样本，然后基于这k个样本的类别进行预测。

```
# 实例化模型，模型名称选择KNN(K Nearest Neighbor)
model = cls(algorithm = 'KNN',n_neighbors=3)
```

`n_neighbors`表示k的值，参数需设置为整数，默认值为5。

#### SVM

支持向量机（support vector machine），简称SVM，能够同时最小化经验误差与最大化几何边缘区，因此支持向量机也被称为最大边缘区分类器。可用于分类和回归。此处使用支持向量机（SVM）完成分类任务。

```
#实例化模型，模型名称选择SVM
model=cls('SVM')
```

#### 多层感知机（MLP）

感知机（perceptrons）是最简单的一种神经网络，由单个神经元构成。感知机组成的网络就是多层感知机（Multilayer Perceptron，简称MLP），多层感知机又称为前馈神经网络。神经元以层级结构组织在一起，层数一般是二三层，但是理论上层数是无限的。

```
# 实例化模型，模型名称选择MLP（Multilayer Perceptron），n_hidden = (100,100)表示2层神经元数量为100的隐藏层
model=cls(algorithm = 'MLP',n_hidden = (100,100))
```

`n_hidden`表示隐藏层，参数值设置为一个元组，元组的元素数表示隐藏层数，元素的值依次表示隐藏层的神经元数。

#### Adaboost分类

AdaBoost是Adaptive Boosting的缩写，Adaptive是自适应的意思，Boosting(提升)是一族可将弱学习器提升为强学习器的算法。AdaBoost算法核心思想是针对同一个训练集训练不同的分类器（弱分类器），然后把这些弱分类器集合起来，构成一个更强的最终分类器（强分类器）。

```
# 实例化模型，模型名称选择AdaBoost（Adaptive Boosting）
model=cls(algorithm = 'AdaBoost'，n_estimators = 50)
```

`n_estimators`表示弱学习器数量，默认值为100。

...

#### 查看拥有的算法以及类注释

```
cls.__doc__
```



### 2. 数据载入

BaseML库支持各种形式载入数据。

1. 从路径载入数据：

```
# 载入数据集，并说明特征列和标签列
model.load_dataset('./lenses.csv', type ='csv', x_column = [1,2,3,4],y_column=[5])
```

`x_column`表示特征列，`y_column`表示标签列。

2. 读取图像数据转换为Numpy数组后直接从变量载入数据：

```
# 载入数据，并说明特征列和标签列
model.load_dataset(X=train_x, y=train_y,type ='numpy')
```

`X`表示数据特征，`y`表示标签。可再设置`x_column`和`y_column`参数，不设置则默认所有列。同时BaseML内置了图片读取处理模块`ImageLoader`。

`ImageLoader`是BaseML内置的图片读取处理模块，用于进行图像数字化处理，读取图片并提取其中的图像特征，如HOG特征和LBP特征，用以进行后续的机器学习任务。

使用示例：

以读取ImageNet格式的MNIST数据集为例，

```
from BaseML import IMGLoader
# 指定数据集路径
train_path = '/data/QX8UBM/mnist_sample/training_set'
test_path = '/data/QX8UBM/mnist_sample/test_set'
# 初始化图片加载器并载入数据集
img_set = IMGLoader.ImageLoader(train_path, test_path,size=28)
# 图像数字化处理
X_train, y_train, X_test, y_test = img_set.get_data(method='flatten')
```

```
# 载入数据，从变量载入
model.load_dataset(X=X_train, y=y_train,type ='numpy')
```

### 3. 模型训练

```
# 模型训练
model.train()
```

### 4. 模型测试

对一组数进行推理：

```
# 给定一组数据，推理查看效果
y=model.inference([[1,  1,  1,  1]])
```

读入测试集所有数据进行推理：

```
# 用测试集数据测试模型效果
test_x = np.loadtxt(test_path, dtype=float, delimiter=',',skiprows=1,usecols=range(1,5) # 读取2-5列，特征列
y_pred=m.inference(test_x) 
```

### 5. 模型的保存与加载

```python
# 保存模型
model.save('my_CART_model.pkl')
# 加载模型
model.load("my_CART_model.pkl")
```

参数为模型保存的路径。

## 回归任务

### 0. 引入包

```
from BaseML import Regression as reg
```

### 1. 实例化

#### 线性回归

线性回归（Linear Regression）是一种数据分析技术，它通过使用另一个相关的已知数据值来预测未知数据的值，类似于一次函数。回归算法一般用于确定两种或两种以上变量间的定量关系。按照自变量的数量多少，可以分为一元回归和多元回归。

```
# 实例化模型，模型名称选择'LinearRegression'
model = reg(algorithm = 'LinearRegression')
```

#### 决策树回归

决策树（Decision Trees）是一种用来分类（Classification）和 回归（Regression）的算法。其目的是创建一种模型从数据特征中学习简单的决策规则来预测一个目标变量的值。优势在于便于理解和解释。树的结构可以可视化出来，训练需要的数据少。

```
# 实例化模型，模型名称选择'DecisionTree'
model = reg(algorithm = 'DecisionTree')
```

#### 随机森林回归

随机森林回归算法（Random Forest Regression）通过随机抽取样本和特征，建立多棵相互不关联的决策树，通过并行的方式获得预测结果。每棵决策树都能通过抽取的样本和特征得出一个预测结果，通过综合所有树的结果取平均值，得到整个森林的回归预测结果。

```
# 实例化模型，模型名称选择'RandomForest'
model = reg(algorithm = 'RandomForest')
```

...

#### 查看拥有的算法以及类注释

```
reg.__doc__
```



### 2. 数据载入

与分类任务一致（见上文）。

### 3. 模型训练

与分类任务一致（见上文）。

### 4. 模型测试

与分类任务一致（见上文）。

### 5. 模型的保存与加载

与分类任务一致（见上文）。

## 聚类任务

### 0. 引入包

```
from BaseML import Cluster as clt
```

### 1. 实例化

#### k均值

k均值（k-means）算法是一种基于数据间距离迭代求解的聚类算法，通过分析数据之间的距离，发现数据之间的内在联系和相关性，将看似没有关联的事物聚合在一起，并将数据划分为若干个集合，方便为数据打上标签，从而进行后续的分析和处理。k代表划分的集合个数，means代表子集内数据对象的均值。

```
# 实例化模型，模型名称选择'KMeans',N_CLUSTERS设置为3
model = clt(algorithm='KMeans',N_CLUSTERS=3)
```

`N_CLUSTERS`表示k的值，默认值为5。

#### 谱聚类

谱聚类（spectral clustering）算法主要思想是把所有的数据看做空间中的点，这些点之间可以用边连接起来。将聚类问题转为图分割问题：距离较远（或者相似度较低）的两个点之间的边权重值较低，而距离较近（或者相似度较高）的两个点之间的边权重值较高，将所有数据点组成的图分割成若干个子图，让不同的子图间边权重和尽可能的低，而子图内的边权重和尽可能的高，从而达到聚类的目的。

```
# 实例化模型，模型名称选择'SpectralClustering',N_CLUSTERS设置为3
model = clt(algorithm='SpectralClustering',N_CLUSTERS=3)
```

`N_CLUSTERS`表示子图的数量，默认值为5。

#### Agglomerative clustering

Agglomerative clutsering 是一种自底而上的层次聚类方法，它能够根据指定的相似度或距离定义计算出类之间的距离。首先将每个样本都视为一个簇，然后开始按一定规则，将相似度高的簇进行合并，直到所有的元素都归为同一类。

```
# 实例化模型，模型名称选择'Agglomerative clustering',N_CLUSTERS设置为3
model = clt(algorithm='Agglomerative clustering',N_CLUSTERS=3)
```

`N_CLUSTERS`表示聚类的数量，默认值为5。

...

#### 查看拥有的算法以及类注释

```
clt.__doc__
```

### 2. 数据载入

与分类任务一致（见上文）。

### 3. 模型训练

与分类任务一致（见上文）。

### 4. 模型测试

与分类任务一致（见上文）。

### 5. 模型的保存与加载

与分类任务一致（见上文）。

## 附录

### 1. 分类、回归和聚类

如果预测任务是为了将观察值分类到有限的标签集合中，换句话说，就是给观察对象命名，那任务就被称为 **分类** 任务。另外，如果任务是为了预测一个连续的目标变量，那就被称为 **回归** 任务。

所谓 **聚类** ，即根据相似性原则，将具有较高相似度的数据对象划分至同一类簇，将具有较高相异度的数据对象划分至不同类簇。聚类与分类最大的区别在于，聚类过程为无监督过程，即待处理数据对象没有任何先验知识，而分类过程为有监督过程，即存在有先验知识的训练数据集。
