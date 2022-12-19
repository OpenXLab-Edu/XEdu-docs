# BaseML快速入门

## 简介

BaseML库提供了众多机器学习训练方法，可以快速训练和应用模型。

## 安装

`pip install baseml`或`pip install BaseML`

## 体验

可以在命令行输入BaseML查看安装的路径，在安装路径内，可以查看提供的更多demo案例。

此处以用“决策树方法配隐形眼镜”案例为示例，想看更多案例也可以查看附录。

## 训练

### 0.引入包

```
# 导入库，从BaseML导入分类模块
from BaseML import Classification
```

### 1.实例化模型

```
# 实例化模型，模型名称选择CART（Classification and Regression Trees）
model=Classification.cls('CART')
```

### 2.载入数据

```
# 载入数据集，并说明特征列和标签列
model.load_dataset('./lenses.csv', type ='csv', x_column = [1,2,3,4],y_column=[5])
```

`x_column`表示特征列，`y_column`表示标签列。

### 3.模型训练

```
# 模型训练
model.train()
```

## 推理

### 使用现有模型直接推理

对一组数据直接推理。

```
model=Classification.cls('CART')
model.load('mymodel.pkl')
y=model.inference([[1,  1,  1,  1]])
```

输出结果数据类型为`array`的一维数组。

### 输出推理结果

定义`label`存储标签名称，根据`label`和推理结果输出真实标签。

```
label=['不适合佩戴', '软材质', '硬材质']
print(label[y[0]-1])
```

### 模型的保存与加载

```
# 保存模型
model.save('mymodel.pkl')

# 加载模型
model.load('mymodel.pkl')
```

参数为模型保存的路径，`.pkl`文件格式可以理解为将python中的数组、列表等持久化地存储在硬盘上的一种方式。

## 快速体验

体验BaseML的最快速方式是通过OpenInnoLab平台。

OpenInnoLab平台为上海人工智能实验室推出的青少年AI学习平台，满足青少年的AI学习和创作需求，支持在线编程。在“AI项目工坊 - 人工智能工坊”中，查找”BaseML“，即可找到所有与BaseML相关的体验项目。

AI项目工坊：https://www.openinnolab.org.cn/pjLab/projects/channel（用Chorm浏览器打开效果最佳）



## 附录

### 案例1.基于决策树的道路智能决策

说明：案例来源于上海科教版《人工智能初步》人教地图56-58页。

目的：本案例借助决策树算法完成道路智能决策，通过学习和实验引导学生了解决策树的工作原理，掌握决策树分类任务编程的流程。

数据集来源：上海科教版《人工智能初步》人教地图56-58页。

##### 1）模型训练

```
# 导入库，从BaseML导入分类模块
from BaseML import Classification
# 实例化模型，模型名称选则CART（Classification and Regression Trees）
model=Classification.cls('CART')
# 载入数据集，并说明特征列和标签列
model.load_dataset('./道路是否可通行历史数据f.csv', type ='csv', x_column = [1,2],y_column=[3])
# 模型训练
model.train(validate = False)
# 保存模型
model.save('my_CART_model.pkl')
```

##### 2）模型推理

```
# 给定一组数据，推理查看效果
y=model.inference([[1,  10]]) 
# 输出结果
label=['不可通行', '可通行']
print(label[y[0]-1])
```



### 案例2.用多层感知机算法实现手写体数字分类

说明：案例来源于《人工智能初步》广东教育出版社版75-80页。

目的：本案例使用多层感知机算法完成手写体分类，通过学习和实验引导学生了解神经网络和多层感知机，掌握多层感知机分类任务编程的流程。

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

```
# 导入库，从BaseML导入分类模块
from BaseML import Classification
# 实例化模型，模型名称选择MLP（Multilayer Perceptron），n_hidden = (100,100)表示2层神经元数量为100的隐藏层
model=Classification.cls(algorithm = 'MLP',n_hidden = (100,100))
# 载入数据，从变量载入
model.load_dataset(X=X_train, y=y_train,type ='numpy')
# 模型训练
model.train()
# 保存模型
model.save('checkpoints/mymodel.pkl')
```

##### 2）模型推理

```
# 给定一张图片，推理查看效果
img = '/data/QX8UBM/mnist_sample/test_set/0/0.jpg' # 指定一张图片
img_cast = img_set.pre_process(img)
data = img_set.get_feature(img_cast,method = 'flatten')
print(data)
y = model.inference(data) #图片推理
print(y)
# 输出结果
label=['0', '1','2', '3', '4','5', '6', '7','8', '9']
print(label[y[0]])
```



### 案例3.用k近邻为参观者推荐场馆

说明：案例来源于华东师范大学出版社《人工智能初步》56-57页。

目的：本案例使用k近邻算法完成参观者信息的分类，通过学习和实验引导学生了解k近邻的工作原理，掌握使用BaseML进行k近邻分类的方法。

数据集来源：华东师范大学出版社《人工智能初步》 38页。

首先导入库并进行文本特征数字化。

```
# 导入需要的各类库，numpy和pandas用来读入数据和处理数据，BaseML是主要的算法库
import numpy as np
import pandas as pd
from BaseML import Classification

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

```
# 实例化模型，KNN默认值为k=5
model=Classification.cls('KNN')
# 载入数据集，并说明特征列和标签列
model.load_dataset(X = df, y = df, type ='pandas', x_column = [1,2,3,4],y_column=[5])
# 开始训练
model.train()
# 保存模型
model.save('mymodel.pkl')
```

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
model1=Classification.cls(algorithm = 'KNN',n_neighbors =3)
model1.load_dataset(X = df, y = df, type ='pandas', x_column = [1,2,3,4],y_column=[5])
model1.train()
# 保存模型
model.save('mymodel2.pkl')
```



### 案例4.用线性回归预测蛋糕价格

说明：案例来源于人教地图版《人工智能初步》39-41页。

目的：本案例使用线性回归预测蛋糕价格，该案场景贴近生活，通过学习和实验引导学生了解线性回归的工作原理，掌握使用BaseML中的线性回归进行预测的方法。

数据集来源：人教地图版《人工智能初步》39-41页。

##### 1）模型训练

```
# 导入需要的各类库，numpy和pandas用来读入数据和处理数据，BaseML是主要的算法库
import numpy as np
import pandas as pd
from BaseML import Regression
# 实例化模型
model = Regression.reg(algorithm = 'LinearRegression')
# 指定数据集，需要显式指定类型
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



### 案例5.用KMeans实现集合地点的选择

说明：案例来源于华东师范大学出版社《人工智能初步》53-55页。

目的：本案例使用KMeans实现分布地点的聚类，通过学习和实验引导学生了解KMeans的工作原理，掌握使用BaseML进行KMeans聚类的方法。

数据集来源：自动生成。

```
# 导入需要的各类库，numpy和pandas用来读入数据和处理数据，BaseML是主要的算法库
import numpy as np
import pandas as pd
from BaseML import Cluster
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
model = Cluster.clt(algorithm = 'Kmeans', N_CLUSTERS=2)
# 指定数据集，需要显式指定类型
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



### 案例6.车辆聚类

说明：案例来源于上海科技教育出版社《人工智能初步》88-89页。

目的：本案例使用KMeans实现汽车展上车辆信息的聚类，了解KMeans的工作原理，掌握使用BaseML进行KMeans聚类的方法。

数据集来源：上海科技教育出版社《人工智能初步》88页。

##### 1）模型训练

```
# 导入需要的各类库，numpy和pandas用来读入数据和处理数据，BaseML是主要的算法库
import numpy as np
import pandas as pd
from BaseML import Cluster

# 读取数据
df = pd.read_csv("车辆聚类.csv")
# 实例化模型
model = Cluster.clt(algorithm = 'Kmeans', N_CLUSTERS=2)
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

