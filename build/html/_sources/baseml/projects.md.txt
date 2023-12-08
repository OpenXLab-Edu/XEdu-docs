# BaseML项目案例集

## 基于决策树的道路智能决策

本案例来源于上海科教版《人工智能初步》人教地图56-58页。

数据集来源：上海科教版《人工智能初步》人教地图56-58页。

项目地址：[https://www.openinnolab.org.cn/pjlab/project?id=64140719ba932064ea956a3e&sc=635638d69ed68060c638f979#public](https://www.openinnolab.org.cn/pjlab/project?id=64140719ba932064ea956a3e&sc=635638d69ed68060c638f979#public)

### 项目核心功能和实现效果展示：

借助[决策树](https://xedu.readthedocs.io/zh/master/baseml/introduction.html#id5)算法完成道路智能决策，可通过学习和实验了解决策树的工作原理，掌握决策树分类任务编程的流程。

![](https://www.openinnolab.org.cn/webdav/635638d69ed68060c638f979/638028c0777c254264da4dd7/current/assets/%E5%88%A9%E7%94%A8%E5%8E%86%E5%8F%B2%E6%95%B0%E6%8D%AE%E7%94%9F%E6%88%90%E5%86%B3%E7%AD%96%E6%A0%91.png)

### 数据说明：

第0列：序号；

第1列：道路施工状况：(1) 未施工, (2) 施工；

第2列：预计车流量 ；

第3列：分类结果（道路能否通行）：(1) 不可通行, (2) 可通行。

![](https://www.openinnolab.org.cn/webdav/635638d69ed68060c638f979/638028c0777c254264da4dd7/current/assets/screenshot-20221205-111611.png)

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
model.train(validate = False)
# 保存模型
model.save('my_CART_model.pkl')
```

##### 2）模型推理

```python
# 给定一组数据，推理查看效果
y=model.inference([[1,  10]]) 
# 输出结果
label=['不可通行', '可通行']
print(label[y[0]-1])
```



## 用多层感知机算法实现手写体数字分类

本案例来源于《人工智能初步》广东教育出版社版75-80页。

项目地址：[https://www.openinnolab.org.cn/pjlab/project?id=637db4e401df4535876a8690&sc=62f34141bf4f550f3e926e0e#public](https://www.openinnolab.org.cn/pjlab/project?id=637db4e401df4535876a8690&sc=62f34141bf4f550f3e926e0e#public)

### 项目核心功能：

阿拉伯数字的字形信息量很小,不同数字写法字形相差又不大，使得准确区分某些数字相当困难。本项目解决的核心问题是如何利用计算机自动识别人手写在纸张上的阿拉伯数字。使用的数据集MNIST数据集包含 0~9 共10种数字的手写图片，每种数字有7000张图片，采集自不同书写风格的真实手写图片，整个数据集一共70000张图片。70000张手写数字图片使用`train_test_split`方法划分为60000张训练集（Training Set）和10000张测试集（Test Set）。项目核心功能是使用BaseML库搭建[多层感知机](https://xedu.readthedocs.io/zh/master/baseml/introduction.html#mlp)实现手写数字识别。

### 实现步骤：

首先需对MNIST数据集进行图像数字化处理，使用BaseML自带的IMGLoader库。

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

##### 1）模型训练

```python
# 导入库，从BaseML导入分类模块
from BaseML import Classification as cls
# 实例化模型，模型名称选择MLP（Multilayer Perceptron）
model=cls(algorithm = 'MLP',n_hidden = (100,100))
# 元组中元素的数量代表隐藏层的层数，每个元素的值代表每个隐藏层中神经元数
# n_hidden = (100,100)表示2层神经元数量为100的隐藏层,
# 载入数据，从变量载入
model.load_dataset(X=X_train, y=y_train,type ='numpy')
# 模型训练
model.train()
# 保存模型
model.save('checkpoints/mymodel.pkl')
```

##### 2）模型推理

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

使用BaseML来实现[k近邻（knn）](https://xedu.readthedocs.io/zh/master/baseml/introduction.html#k)分类算法，为旅行者们推荐最适合他们的场馆。在项目实践中了解k近邻的工作原理，掌握使用BaseML进行k近邻分类的方法。

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
# 保存模型
model.save('mymodel.pkl')
```

使用BaseML特色功能进行评价指标可视化：

```
# 评价指标可视化
model.metricplot()
```

![](../images/baseml/knnvis.png)

根据可视化生成的图例可以清晰呈现哪些类别预测错误以及预测的结果。

如上图，正确答案是类别0，全部预测正确；

而正确答案是类别1时有一半预测结果为2，一半预测正确，另一半预测错误；

正确答案是类别2的则全部预测错误。

##### 2）模型推理 

```
# 给定一组数据，查看模型推理结果
test_data = [[0,1,0,1]]
test_y = model.inference(test_data)
print(test_y)
print(loc.inverse_transform(test_y))
```

修改k值进行训练：

```
# # 实例化模型，设置k=3
model1=cls(algorithm = 'KNN',n_neighbors =3)
model1.load_dataset(X = df, y = df, type ='pandas', x_column = [1,2,3,4],y_column=[5])
model1.train()
# 保存模型
model.save('mymodel2.pkl')
```



## 用线性回归预测蛋糕价格

本案例来源于人教地图版《人工智能初步》39-41页。

项目地址：[https://www.openinnolab.org.cn/pjlab/project?id=64141e08cb63f030543bffff&sc=635638d69ed68060c638f979#public](https://www.openinnolab.org.cn/pjlab/project?id=64141e08cb63f030543bffff&sc=635638d69ed68060c638f979#public)

项目地址2（增加可视化版本）：[https://www.openinnolab.org.cn/pjlab/project?id=6368a382bbcccd583a837a0e&sc=62f34141bf4f550f3e926e0e#public](https://www.openinnolab.org.cn/pjlab/project?id=6368a382bbcccd583a837a0e&sc=62f34141bf4f550f3e926e0e#public)

### 项目核心功能：

使用[线性回归](https://xedu.readthedocs.io/zh/master/baseml/introduction.html#id14)预测蛋糕价格，案例场景贴近生活，可通过学习和实验了解线性回归的工作原理，掌握使用BaseML中的线性回归进行预测的方法。

数据集来源：人教地图版《人工智能初步》39-41页。

### 实现步骤：

##### 1）模型训练

```python
# 导入需要的各类库，numpy和pandas用来读入数据和处理数据，BaseML是主要的算法库
import numpy as np
import pandas as pd
from BaseML import Regression as reg
# 实例化模型
model = reg(algorithm = 'LinearRegression')
# 指定数据集，需要指定类型
model.load_dataset("蛋糕尺寸与价格.csv", type='csv', x_column=[0],y_column = [1])
# 开始训练
model.train()
# 模型保存
model.save('mymodel.pkl')
```

##### 2）模型推理

```
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



## 用k均值实现园区集合地点选取

本案例来源于华东师范大学出版社《人工智能初步》53-55页。

项目地址：[https://www.openinnolab.org.cn/pjlab/project?id=638015a0777c254264da387f&sc=62f34141bf4f550f3e926e0e#public](https://www.openinnolab.org.cn/pjlab/project?id=638015a0777c254264da387f&sc=62f34141bf4f550f3e926e0e#public)

### 项目核心功能：

使用BaseML中的Cluster模块进行聚类，使用matplotlib库对聚类结果进行可视化。该项目可根据同学所在位置，解决聚集点设定问题。可通过学习和实验了解KMeans的工作原理，掌握使用BaseML进行[k均值（KMeans）](https://xedu.readthedocs.io/zh/master/baseml/introduction.html#id25)聚类的方法。

数据集来源：自定义数据集。

### 实现步骤：

首先完成数据读取。

```python
# 导入需要的各类库，numpy和pandas用来读入数据和处理数据，BaseML是主要的算法库
import numpy as np
import pandas as pd
from BaseML import Cluster as clt
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 生成自定义数据，并查看数据分布情况。随机生成1000个点，定义两个中心。
X,y=make_blobs(n_samples=1000,n_features=2,centers=[[1,5],[5,3]],cluster_std=[0.4,0.6],random_state=9)
plt.scatter(X[:,0],X[:,1],marker='o')
plt.show()
```

##### 1）模型训练

```
# 实例化模型
model = clt(algorithm = 'Kmeans', N_CLUSTERS=2)
# 指定数据集，需要指定类型
model.load_dataset(X = X, type='numpy', x_column=[0,1])
# 开始训练
model.train()
# 模型保存
model.save('mymodel.pkl')
```

##### 2）模型推理

1.无参数推理，输出聚类数据结果

```
# 进行推理
model.inference()
```

2.有参数推理，返回聚类结果，便于可视化

```
# 进行推理（）
result = model.inference(X,verbose = False)
```

可视化聚类结果的代码：

```
import matplotlib.pyplot as plt
# 聚类结果根据颜色区分
plt.scatter(X[:,0],X[:,1], c=result, s=50, cmap='viridis')
# 标出聚类序号，长方形序号的左下角为聚类中心所在位置
centers = model.model.cluster_centers_
for i in range(model.model.cluster_centers_.shape[0]):
    plt.text(centers[:, 0][i]+0.03,y=centers[:, 1][i]+0.03,s=i, 
             fontdict=dict(color='red',size=10),
             bbox=dict(facecolor='yellow',alpha=0.5))
```



## 用k均值实现车辆类别聚类分析

本案例来源于上海科技教育出版社《人工智能初步》88-89页。

项目地址：[https://www.openinnolab.org.cn/pjlab/project?id=638015e4777c254264da38ca&sc=62f34141bf4f550f3e926e0e#public](https://www.openinnolab.org.cn/pjlab/project?id=638015e4777c254264da38ca&sc=62f34141bf4f550f3e926e0e#public)

### 项目核心功能：

使用BaseML中的Cluster模块进行聚类，使用matplotlib库对聚类结果进行可视化。该项目可根据车辆的品质，解决车辆分类问题，便于用户进行决策。可通过学习和实验了解KMeans的工作原理，掌握使用BaseML进行[k均值（KMeans）](https://xedu.readthedocs.io/zh/master/baseml/introduction.html#id25)聚类的方法。

数据集来源：上海科技教育出版社《人工智能初步》88页。

### 实现步骤：

##### 1）模型训练

```
# 导入需要的各类库，numpy和pandas用来读入数据和处理数据，BaseML是主要的算法库
import numpy as np
import pandas as pd
from BaseML import Cluster as clt

# 读取数据
df = pd.read_csv("车辆聚类.csv")
# 实例化模型
model = clt(algorithm = 'Kmeans', N_CLUSTERS=2)
# 指定数据集，需要显式指定类型
model.load_dataset(X = df, type='pandas', x_column=[1,2])
# 开始训练
model.train()
# 模型保存
model.save('mymodel.pkl')
```

##### 2）模型推理

1.无参数推理，输出聚类数据结果

```
# 进行推理
model.inference()
```

2.有参数推理，返回聚类结果，便于可视化

```
# 进行推理
result = model.inference(df.loc[:,['大小','颜色']].values)
# 输出最终的车辆聚类文字结果
for index, row in df.iterrows():
    print('{0}号车辆属于第{1}个类别'.format(row['汽车编号'],result[index])) # 输出每一行
```

可视化聚类结果的代码：

```
import matplotlib.pyplot as plt
# 画出不同颜色的车辆点
plt.scatter(df.iloc[:, 1], df.iloc[:, 2], c=result, s=50, cmap='viridis')

# 标出聚类序号，长方形序号的左下角为聚类中心所在位置
centers = model.model.cluster_centers_
for i in range(model.model.cluster_centers_.shape[0]):
    plt.text(centers[:, 0][i]+0.03,y=centers[:, 1][i]+0.03,s=i, 
             fontdict=dict(color='red',size=10),
             bbox=dict(facecolor='yellow',alpha=0.5),
            zorder=-1)
```

