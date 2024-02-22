# BaseML功能详解

我们的传统机器学习（Mechine Learning）有很多算法，但总的来说，可以分为三大类：分类、回归和聚类。BaseML和sklearn不同之处，也就体现于此，sklearn尽管在拟合、预测等函数上对各模块做了统一，但并没有明确指出这样的三大类划分方式。这三类也有着特有的数据输入格式。除此之外，BaseML还提供了`DimentionReduction`数据降维模块，用以对数据进行降维处理。

文档涉及的部分代码见XEdu帮助文档配套项目集：[https://www.openinnolab.org.cn/pjlab/project?id=64f54348e71e656a521b0cb5&sc=645caab8a8efa334b3f0eb24#public](https://www.openinnolab.org.cn/pjlab/project?id=64f54348e71e656a521b0cb5&sc=645caab8a8efa334b3f0eb24#public)

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

#### 决策树分类（CART）

决策树（Decision Trees）是一种用来分类（Classification）和 回归（Regression）的算法。决策树算法将数据看作树的若干片叶子。在每一个树杈位置，决策树都根据特征的不同而划分，将不同的叶子分在不同的枝干上，算法根据最优划分方法，将误差降到最低。该算法解释性强，在解决各种问题时都有良好表现。此处使用决策树分类（CART）完成分类任务。

```
# 实例化模型，模型名称选择CART（Classification and Regression Trees）
model=cls('CART')
```

#### 最近邻分类（KNN）

最近邻分类（K Nearest Neighbors）算法，简称为kNN，属于监督学习中的分类算法，核心思想是“近朱者赤”。如果要分析一个新数据点的类别，我们会寻找离它最近的K个邻居，哪类邻居多，就认为新数据点也属于该类。适用于数据集较小等情况，分类结果直观。

```
# 实例化模型，模型名称选择KNN(K Nearest Neighbor)
model = cls(algorithm = 'KNN',n_neighbors=3)
```

`n_neighbors`表示k的值，参数需设置为整数，默认值为5。

#### 支持向量机（SVM）

支持向量机（support vector machine），简称SVM，支持向量机算法在一个高次元空间来思考问题，尤其适合处理多特征、非线性和少样本的学习问题。此外，它能够很好地适应干扰数据和异常值带来的模型误差。可用于分类和回归。此处使用支持向量机（SVM）完成分类任务。

```
#实例化模型，模型名称选择SVM
model=cls('SVM')
```

#### 自适应增强分类（AdaBoost）

自适应增强算法（Adaptive Boosting，AdaBoost）是一种迭代算法，需要经历多个阶段，在每个阶段都增加一个新的智能体帮助判断，直到达到足够小的错误率。这种算法在各领域都表现出超凡的能力。

```
# 实例化模型，模型名称选择AdaBoost（Adaptive Boosting）
model=cls(algorithm = 'AdaBoost'，n_estimators = 50)
```

`n_estimators`表示弱学习器数量，默认值为100。

#### 随机森林分类（RandomForest）

随机森林算法是一种基于集成学习的算法，通过构建多棵决策树并将它们的预测结果进行集成，从而降低风险。它能够处理多特征数据，并自动选择最相关特征，从而提升模型准确率。

```
# 实例化模型，模型名称选择RandomForest
model=cls(algorithm = 'RandomForest')
```

#### 多层感知机（MLP）

多层感知机算法是一种深度学习算法。它通过模拟大脑的神经元系统，将信号通过突触传递到与之相关的神经元中，如果传递正确，这样的传递就会被强化，从而逐渐构成模型。它可以自动学习到输入特征之间非常复杂的关系。但是，它的训练时间可能会较长，且依赖大量训练数据。

```
# 实例化模型，模型名称选择MLP（Multilayer Perceptron），n_hidden = (100,100)表示2层神经元数量为100的隐藏层
model=cls(algorithm = 'MLP',n_hidden = (100,100))
```

`n_hidden`表示隐藏层，参数值设置为一个元组，元组的元素数表示隐藏层数，元素的值依次表示隐藏层的神经元数。

...

#### 查看拥有的算法以及类注释

```
cls.__doc__
```



### 2. 数据载入

BaseML库支持各种形式载入数据。

#### （1）从路径载入数据：

```
# 载入数据集，并说明特征列和标签列
model.load_dataset('./lenses.csv', type ='csv', x_column = [1,2,3,4],y_column=[5])
```

`x_column`表示特征列，`y_column`表示标签列。

#### （2）读取图像数据转换为Numpy数组后直接从变量载入数据：

```
# 载入数据，并说明特征列和标签列
model.load_dataset(X=train_x, y=train_y,type ='numpy')
```

`X`表示数据特征，`y`表示标签。可再设置`x_column`和`y_column`参数，不设置则默认所有列。同时BaseML内置了图片读取处理模块`ImageLoader`。

#### （3）图片处理

`ImageLoader`是BaseML内置的图片处理模块，用于进行图像数字化处理，读取图片并提取其中的图像特征，如HOG特征和LBP特征，用以进行后续的机器学习任务。

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

模型保存后可加载模型进行模型测试，参考代码如下：

```
# 加载模型
model.load("my_CART_model.pkl")
# 给定一组数据，推理查看效果
y=model.inference(data)
```

也可以[借助XEduHub库完成推理](https://xedu.readthedocs.io/zh/master/xedu_hub/introduction.html#baseml)和应用，更多模型转换与应用的介绍详见[后文](https://xedu.readthedocs.io/zh/master/support_resources/model_convert.html)。

## 回归任务

### 0. 引入包

```
from BaseML import Regression as reg
```

### 1. 实例化

#### 线性回归（Linear Regression）

线性回归（Linear Regression）线性回归算法的核心思想是找到一条直线，使得这条直线能够最好地代表和预测数据。通常适用于连续值的预测，例如房价、贷款额度等。

多元线性回归是线性回归的进一步发展，它考虑了多个因素对预测的影响。想象一下，房屋价格不仅取决于大小，还取决于卧室数量、卫生间数量、地理位置等多个因素。多元线性回归会尝试找到一个更复杂的方程，其中包括所有这些因素，以更准确地预测房屋价格。这就像在画一个多维的图形，而不仅仅是一条直线，以考虑所有这些因素对结果的影响。多元线性回归的模型则可以表示为：y = b0 + b1x1 + b2x2 + ... + bn*xn，y是输出目标，x1、x2、...、xn是不同的输入特征，b0、b1、b2、...、bn是回归系数，表示每个特征对输出的影响。

```
# 实例化模型，模型名称选择'LinearRegression'
model = reg(algorithm = 'LinearRegression')
```

#### 决策树回归（CART）

算法介绍见分类任务的决策树算法。

```
# 实例化模型，模型名称选择'DecisionTree'
model = reg(algorithm = 'DecisionTree')
```

#### 随机森林回归（RandomForest）

算法介绍见分类任务的随机森林算法。

```
# 实例化模型，模型名称选择'RandomForest'
model = reg(algorithm = 'RandomForest')
```

#### 支持向量机回归（SVM）

算法介绍见分类任务的支持向量机算法。

```
# 实例化模型，模型名称选择'SVM'
model = reg(algorithm = 'SVM')
```

#### 自适应增强回归（AdaBoost）

算法介绍见分类任务的自适应增强回归算法。

```
# 实例化模型，模型名称选择'AdaBoost'
model = reg(algorithm = 'AdaBoost'，n_estimators = 50)
```

`n_estimators`表示弱学习器数量，默认值为100。

#### 多层感知机（MLP）

算法介绍见分类任务的多层感知机算法。

```
# 实例化模型，模型名称选择MLP（Multilayer Perceptron），n_hidden = (100,100)表示2层神经元数量为100的隐藏层
model=reg(algorithm = 'MLP',n_hidden = (100,100))
```

`n_hidden`表示隐藏层，参数值设置为一个元组，元组的元素数表示隐藏层数，元素的值依次表示隐藏层的神经元数。

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
