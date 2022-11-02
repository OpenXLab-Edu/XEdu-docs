# BaseML功能详解

我们的传统机器学习（Mechine Learning）有很多算法，但总的来说，可以分为三大类：分类、回归和聚类。BaseML和sklearn不同之处，也就体现于此，sklearn尽管在拟合、预测等函数上对各模块做了统一，但并没有明确指出这样的三大类划分方式。这三类也有着特有的数据输入格式。

## 分类任务

### 实例化

```
# 实例化模型
model=cls()
```

#### 贝叶斯分类



#### 决策树

```
# 实例化模型，模型名称选择CART（Classification and Regression Trees）
model=cls('CART')
```

#### k近邻

```
# 实例化模型，模型名称选择KNN(K-Nearest Neighbor)
model = cls('KNN',n_neighbors=5)
```

k近邻算法(k-NearestNeighbor，KNN)，顾名思义，即由某样本k个邻居的类别来推断出该样本的类别。给定测试样本，基于特定的某种距离度量方式找到与训练集中最接近的k个样本，然后基于这k个样本的类别进行预测。`n_neighbors`参数需设置为整数，表示k的值。

#### SVC

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

```
# 给定一组数据，推理查看效果
y=model.inference([[1,  1,  1,  1]])
```



```
# 用测试集数据测试模型效果
test_x = np.loadtxt(test_path, dtype=float, delimiter=',',skiprows=1,usecols=range(1,5) # 读取2-5列，特征列
test_y = np.loadtxt(test_path, dtype=int, delimiter=',',skiprows=1,usecols=5) # 读取第6列，标签列
y_pred=m.inference(test_x) 
```



## 回归任务

### 实例化

#### 线性回归

#### SVM

...

### 数据载入

### 模型训练

### 模型测试

## 聚类任务

### 实例化

#### K均值

...

### 数据载入

### 模型训练

### 模型测试
