# BaseML功能详解

我们的传统机器学习（Mechine Learning）有很多算法，但总的来说，可以分为三大类：分类、回归和聚类。BaseML和sklearn不同之处，也就体现于此，sklearn尽管在拟合、预测等函数上对各模块做了统一，但并没有明确指出这样的三大类划分方式。这三类也有着特有的数据输入格式。

## 分类任务

### 实例化

#### 贝叶斯分类

```
# 实例化模型，模型名称选择NaiveBayes
model=cls('NaiveBayes')
```

#### 决策树

```
# 实例化模型，模型名称选择CART（Classification and Regression Trees）
model=cls('CART')
```

#### k近邻

k近邻（K Nearest Neighbors）算法，简称为knn，该算法以每一个测试数据为中心，根据在特征空间中与测试数据相邻的训练数据的标签来确定测试数据的标签。给定测试样本，基于特定的某种距离度量方式找到与训练集中最接近的k个样本，然后基于这k个样本的类别进行预测。

```
# 实例化模型，模型名称选择KNN(K-Nearest Neighbor)
model = cls('KNN',n_neighbors=5)
```

`n_neighbors`表示k的值，参数需设置为整数，默认值为5。

#### SVM

使用支持向量机（SVM）完成分类任务。

```
#实例化模型，模型名称选择SVM
model=cls('SVM')
```

#### 多层感知机（MLP）

```
# 实例化模型，模型名称选择MLP（Multilayer Perceptron），n_hidden = (100,100)表示2层神经元数量为100的隐藏层
model=cls('MLP',n_hidden = (100,100))
```

`n_hidden`表示隐藏层，参数值设置为一个元组，元组的元素数表示隐藏层数，元素的值依次表示隐藏层的神经元数

### 数据载入

BaseML库支持各种形式载入数据。

从路径载入数据：

```
# 载入数据集，并说明特征列和标签列
model.load_dataset('./lenses.csv', type ='csv', x_column = [1,2,3,4],y_column=[5])
```

`x_column`表示特征列，`y_column`表示标签列。

从变量载入数据：

```
# 载入数据，并说明特征列和标签列
model.load_dataset(X=train_x, y=train_y,type ='numpy')
```

`X`表示数据特征，`y`表示标签。可再设置`x_column`和`y_column`参数，不设置则默认所有列。

### 模型训练

```
# 模型训练
model.train()
```

### 模型测试

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

## 回归任务

### 实例化

#### 线性回归

线性回归（Linear Regression）是一种数据分析技术，它通过使用另一个相关的已知数据值来预测未知数据的值，类似于一次函数。回归算法一般用于确定两种或两种以上变量间的定量关系。按照自变量的数量多少，可以分为一元回归和多元回归。

```
# 实例化模型，模型名称选择'LinearRegression'
model = reg(algorithm = 'LinearRegression')
```

#### SVM

...

### 数据载入

### 模型训练

### 模型测试

## 聚类任务

### 实例化

#### k均值

k均值（k-means）算法,是一种基于数据间距离的聚类算法，通过分析数据之间的距离，发现数据之间的内在联系和相关性，将看似没有关联的事物聚合在一起，并将数据划分为k个集合，即k个类，方便为数据打上标签，从而进行后续的分析和处理。

```
# 实例化模型，模型名称选择'KMeans',N_CLUSTERS设置为3
model=cls(algorithm='KMeans',N_CLUSTERS=3)
```

`N_CLUSTERS`表示k的值。

...

### 数据载入

```
# 载入数据集，并说明特征列和标签列
model.load_dataset('./lenses.csv', type ='csv', x_column = [1,2,3,4],y_column=[5])
```

### 模型训练

```
# 模型训练
model.train()
```

### 模型测试
