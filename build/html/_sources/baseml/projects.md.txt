# BaseML项目案例集

## 探秘BaseML之MLP（多层感知机）

本案例选用了多套教材中的数据集，并都使用MLP算法对数据集进行训练，实现了**分类**和**回归**两大任务。

项目地址：[https://openinnolab.org.cn/pjlab/project?id=65f69017ace40851ae424258&sc=635638d69ed68060c638f979#public](https://openinnolab.org.cn/pjlab/project?id=65f69017ace40851ae424258&sc=635638d69ed68060c638f979#public)

### 分类任务实现代码举例
```python
from BaseML import Classification as cls # 从库文件中导入分类任务模块
model = cls('MLP') # 实例化MLP模型
model.set_para(hidden_layer_sizes=(10,10)) # 设定模型参数
                                             # 这里的输入和输出层神经元数量是自动识别的
                                             # 只需要设定隐藏层的神经元数量即可
data = model.load_tab_data('data/road-accessibility-status-analysis.csv',train_val_ratio=0.6) # 载入训练数据
print(data)
model.train(lr=0.01,epochs=100) # 训练模型
model.valid(metrics='acc') # 载入验证数据并验证
model.metricplot() 
```
输出如下：
```
Setting hidden_layer_sizes to (10, 10)
(array([[ 2., 83.],
       [ 1., 80.],
       [ 1., 90.],
       [ 1., 71.],
       [ 1., 87.],
       [ 2., 29.],
       [ 1., 47.]]), array([1., 1., 1., 2., 1., 1., 2.]), array([[ 1., 73.],
       [ 2., 75.],
       [ 2., 48.],
       [ 1., 68.],
       [ 1., 78.]]), array([2., 1., 1., 2., 2.]))
验证准确率为：80.0%
```
上面的代码通过metrics='acc'，计算了分类任务的准确性，并可以通过metricplot()将结果可视化。
### 回归任务实现代码举例
```python
from BaseML import Regression as reg # 从库文件中导入回归任务模块
model = reg('MLP') # 实例化MLP模型
model.set_para(hidden_layer_sizes=(10,10)) # 设定模型参数
                                             # 这里的输入和输出层神经元数量是自动识别的
                                             # 只需要设定隐藏层的神经元数量即可
data = model.load_tab_data('data/cake-size-to-price-prediction.csv',train_val_ratio=0.6) # 载入训练数据
print(data)
model.train(lr=0.01,epochs=100) # 训练模型
model.valid(metrics='r2') # 载入验证数据并验证
model.metricplot() 
```
输出如下：
```
Setting hidden_layer_sizes to (10, 10)
(array([[ 9.],
       [ 6.],
       [10.]]), array([69., 40., 77.]), array([[ 8.],
       [12.]]), array([56., 96.]))
验证r2-score为：98.95081251824811%
```
上面的代码通过metrics='r2'，计算了回归任务的R平方指标的值，并可以通过metricplot()将结果可视化。
## 基于决策树的道路智能决策

本案例来源于上海科教版《人工智能初步》人教地图56-58页。

数据集来源：上海科教版《人工智能初步》人教地图56-58页。

项目地址：[https://www.openinnolab.org.cn/pjlab/project?id=64140719ba932064ea956a3e&sc=635638d69ed68060c638f979#public](https://www.openinnolab.org.cn/pjlab/project?id=64140719ba932064ea956a3e&sc=635638d69ed68060c638f979#public)

### 项目核心功能

借助决策树算法完成道路智能决策，可通过学习和实验了解决策树的工作原理，掌握决策树分类任务编程的流程。

### 数据说明：

第0列：序号；

第1列：道路施工状况：(1) 未施工, (2) 施工；

第2列：预计车流量 ；

第3列：分类结果（道路能否通行）：(1) 不可通行, (2) 可通行。

### 实现步骤：

##### 1）模型训练

```python
# 导入库，从BaseML导入分类模块
from BaseML import Classification as cls
# 实例化模型，模型名称选则CART（Classification and Regression Trees）
model=cls('CART')
# 载入数据集，并说明特征列和标签列
model.load_dataset('./道路是否可通行历史数据f.csv', type ='csv', x_column = [1,2],y_column=[3])
# 模型训练
model.train()
```

##### 2）模型评估

```python
# 模型评估,使用载入数据时默认拆分出的验证集进行评估
model.valid()
# 模型评价指标可视化
model.metricplot()
```

##### 3）模型保存

```python
# 保存模型
model.save('my_CART_model.pkl')
```

##### 4）模型推理

```python
# 使用载入功能，复现效果
m=cls('CART')
m.load('my_CART_model.pkl') # 模型保存路径
y=m.inference([[2,  10]]) # 2代表施工中，10代表预计车流量为10
print(y)
print(label[y[0]-1])
```

## 用多层感知机算法实现手写体数字分类

本案例来源于《人工智能初步》广东教育出版社版75-80页。

项目地址：[https://openinnolab.org.cn/pjlab/project?id=6440e64606618727bee5c1ce&backpath=/pjlab/projects/list#public](https://openinnolab.org.cn/pjlab/project?id=6440e64606618727bee5c1ce&backpath=/pjlab/projects/list#public)

### 项目核心功能：

阿拉伯数字的字形信息量很小,不同数字写法字形相差又不大，使得准确区分某些数字相当困难。本项目解决的核心问题是如何利用计算机自动识别人手写在纸张上的阿拉伯数字。使用的数据集MNIST数据集包含 0~9 共10种数字的手写图片，每种数字有7000张图片，采集自不同书写风格的真实手写图片，整个数据集一共70000张图片。70000张手写数字图片使用`train_test_split`方法划分为60000张训练集（Training Set）和10000张测试集（Test Set）。项目核心功能是使用BaseML库搭建多层感知机实现手写数字识别。

### 实现步骤：

首先需对MNIST数据集进行图像数字化处理，使用BaseML自带的IMGLoader库。

```python
from BaseML import IMGLoader
# 指定数据集路径
train_path = '/data/QX8UBM/mnist_sample/training_set'
test_path = '/data/QX8UBM/mnist_sample/test_set'
# 初始化图片加载器并载入数据集
img_set = IMGLoader.ImageLoader(train_path, test_path,size=28)
# 图像数字化处理
X_train, y_train, X_test, y_test = img_set.get_data(method='flatten')
```

##### 1）模型训练

```python
# 导入库，从BaseML导入分类模块
from BaseML import Classification as cls
# 搭建模型，模型名称选择MLP（Multilayer Perceptron）
model=cls('MLP')
# 设置参数，hidden_layer_sizes":(100,100)表示2层神经元数量为100的隐藏层
model.para = {"hidden_layer_sizes":(100,100)}
# 载入数据，从变量载入
model.load_dataset(X=X_train, y=y_train,type ='numpy')
# 模型训练
model.train()
```

##### 2）模型评估

```python
# 读取验证数据进行评估
model.valid(x=X_val, y=y_val,metrics='acc')
# 评价指标可视化
model.metricplot(X_val,y_val)
```

##### 3）模型保存

```python
# 保存模型
model.save('checkpoints/mymodel.pkl')
```

##### 4）模型推理

```python
# 给定一张图片，推理查看效果
img = '/data/QX8UBM/mnist_sample/test_set/0/0.jpg' # 指定一张图片
img_cast = img_set.pre_process(img)
data = img_set.get_feature(img_cast,method = 'flatten')
print(data)
y = model.inference(data) #图片推理
print(y)
# 输出结果
label=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print(label[y[0]])
```

## 用k近邻为参观者推荐场馆

本案例来源于华师大出版社《人工智能初步》56-57页。

项目地址：[https://www.openinnolab.org.cn/pjlab/project?id=6417d0477c99492cf1aa8ba6&sc=635638d69ed68060c638f979#public](https://www.openinnolab.org.cn/pjlab/project?id=6417d0477c99492cf1aa8ba6&sc=635638d69ed68060c638f979#public)

### 项目核心功能：

使用BaseML来实现k近邻（knn）分类算法，为旅行者们推荐最适合他们的场馆。在项目实践中了解k近邻的工作原理，掌握使用BaseML进行k近邻分类的方法。

数据集来源：华师大出版社《人工智能初步》38页。

### 实现步骤：

首先导入库并进行文本特征数字化。

```python
# 导入需要的各类库，numpy和pandas用来读入数据和处理数据，BaseML是主要的算法库
import numpy as np
import pandas as pd
from BaseML import Classification as cls

# 构建字典键值对
yesno_dict = {'是':1,'否':0}
number_dict = {'多':1,'少':0}
weather_dict = {'雨':-1, '阴':0, '晴':1}

# 采用map进行值的映射
df['首次参观'] = df['首次参观'].map(yesno_dict)
df['参观人数'] = df['参观人数'].map(number_dict)
df['天气'] = df['天气'].map(weather_dict)
df['专业人士'] = df['专业人士'].map(yesno_dict)
```

##### 1）模型训练

```python
# 实例化模型，KNN默认值为k=5
model=cls('KNN')
# 载入数据集，并说明特征列和标签列
model.load_dataset(X = df, y = df, type ='pandas', x_column = [1,2,3,4],y_column=[5])
# 开始训练
model.train()
```

##### 2）模型评估

```python
# 模型评估,使用载入数据时默认拆分出的验证集进行评估
model.valid()
# 模型评价指标可视化
model.metricplot()
```

![](../images/baseml/knnvis.png)

根据可视化生成的图例可以清晰呈现哪些类别预测错误以及预测的结果。

如上图，正确答案是类别0，全部预测正确；

而正确答案是类别1时有一半预测结果为2，一半预测正确，另一半预测错误；

正确答案是类别2的则全部预测错误。

##### 3）模型推理 

```python
# 给定一组数据，查看模型推理结果
test_data = [[0,1,0,1]]
test_y = model.inference(test_data)
print(test_y)
print(loc.inverse_transform(test_y))
```

拓展-修改k值进行训练：

```python
# 使用k = 3进行训练
model1=cls('KNN')
model1.para = {"n_neighbors":3}
model1.load_dataset(X = df, y = df, type ='pandas', x_column = [1,2,3,4],y_column=[5])
model1.train()
```



## 用线性回归预测蛋糕价格

本案例来源于人教地图版《人工智能初步》39-41页。

项目地址：[https://openinnolab.org.cn/pjlab/project?id=64141e08cb63f030543bffff&backpath=/pjlab/projects/list#public](https://openinnolab.org.cn/pjlab/project?id=64141e08cb63f030543bffff&backpath=/pjlab/projects/list#public)

### 项目核心功能：

使用线性回归预测蛋糕价格，案例场景贴近生活，可通过学习和实验了解线性回归的工作原理，掌握使用BaseML中的线性回归进行预测的方法。

数据集来源：人教地图版《人工智能初步》39-41页。

### 实现步骤：

##### 1）模型训练

```python
# 导入需要的各类库，numpy和pandas用来读入数据和处理数据，BaseML是主要的算法库
import numpy as np
import pandas as pd
from BaseML import Regression as reg
# 实例化模型
model = reg('LinearRegression')
# 载入训练数据
model.load_tab_data( '蛋糕尺寸与价格.csv') 
# 开始训练
model.train()
```

##### 2）模型评估

```python
# 计算R值进行评估
model.valid('蛋糕尺寸与价格.csv',metrics='r2')
```

##### 3）模型保存

```python
# 模型保存
model.save('mymodel.pkl')
```

##### 4）模型应用

```python
# 指定数据
df = pd.read_csv("蛋糕尺寸与价格.csv")
# 输出模型对于数据的预测结果
result = model.inference(df.values[:,0].reshape(-1,1))

# 可视化线性回归
import matplotlib.pyplot as plt
# 画真实的点
plt.scatter(df['蛋糕尺寸/英寸'], df['价格/元'], color = 'blue')
# 画拟合的直线
plt.plot(df.values[:,0].reshape(-1,1), result, color = 'red', linewidth = 4)
plt.xlabel('size')
plt.ylabel('value')
plt.show()
```



## 用k均值实现城市聚类分析

本案例来源于人民教育出版社《人工智能初步》（中国地图出版社）56-59页。

项目地址：[https://openinnolab.org.cn/pjlab/project?id=6440ce55d73dd91bcbcbb934&backpath=/pjedu/userprofile?slideKey=project#public](https://openinnolab.org.cn/pjlab/project?id=6440ce55d73dd91bcbcbb934&backpath=/pjedu/userprofile?slideKey=project#public)

### 项目核心功能：

使用BaseML中的Cluster模块进行聚类，使用matplotlib库对聚类结果进行可视化。该项目可根据同学所在位置，解决聚集点设定问题。可通过学习和实验了解KMeans的工作原理，掌握使用BaseML进行k均值（KMeans）聚类的方法。

数据集来源：自定义数据集。

### 实现步骤：

首先完成数据读取。

```python
import pandas as pd
# 观察数据情况
df = pd.read_csv("2016地区GDP.csv")
```

##### 1）模型训练

```python
# 实例化模型
model = clt('Kmeans')
model.set_para(N_CLUSTERS=5) 
# 指定数据集，需要显式指定类型. show可以显示前5条数据，scale表示要进行归一化。数量级相差大的特征需要进行归一化。
model.load_dataset(X = df, type='pandas', x_column=[1,2], shuffle=True,scale=True)
# 开始训练
model.train()
# 模型保存
model.save('mymodel.pkl')
```

##### 2）模型推理

```python
# 进行推理
result = model.inference()
print(result)
```

```python
# 输出最终的城市聚类文字结果
for index, row in df.iterrows():
    print('{0}属于第{1}个城市集群'.format(row['地区'],result[index])) # 输出每一行
```

可视化聚类结果的代码：

```python
# 可视化最终的城市集群结果
import matplotlib.pyplot as plt

# 画出不同颜色的城市集群点
plt.scatter(df.iloc[:, 1], df.iloc[:, 2], c=result, s=50, cmap='viridis')
# 画出聚类中心
centers = model.reverse_scale(model.model.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
# 标出聚类序号
for i in range(model.model.cluster_centers_.shape[0]):
    plt.text(centers[:, 0][i],y=centers[:, 1][i],s=i, 
             fontdict=dict(color='red',size=10),
             bbox=dict(facecolor='yellow',alpha=0.5),
            zorder=0)
```

