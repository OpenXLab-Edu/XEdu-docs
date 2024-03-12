# 快速体验BaseML

## 简介

BaseML库提供了众多机器学习训练方法，可以快速训练和应用模型。

## 安装

`pip install baseml`或`pip install BaseML`

库文件源代码可以从[PyPi](https://pypi.org/project/BaseML/#files)下载，选择tar.gz格式下载，可用常见解压软件查看源码。

## 体验

可以在命令行输入BaseML查看安装的路径，在安装路径内，可以查看提供的更多demo案例。

下面我们以用“用KNN对鸢尾花Iris数据集进行分类”案例为示例，体验用BaseML做第一个机器学习项目！

认识鸢尾花数据集：

鸢尾属植物有三个品种，分别是山鸢尾(setosa)、变色鸢尾(versicolor)、维吉尼亚鸢尾(virginica)。这些种类之间差别不大，但是不同种类在花瓣和花萼的形状上有所区别。鸢尾花数据集（iris.csv）中包括150条不同鸢尾花的花萼长度、花萼宽度、花瓣长度、花瓣宽度数据。下面使用的是已经完成拆分的数据，iris_training.csv训练数据集，120条样本数据；iris_test.csv测试数据集，30条数据，可借助BaseDT库快速完成[数据集拆分](https://xedu.readthedocs.io/zh/master/basedt/introduction.html#id11)。

## 训练

### 0. 引入包

```
# 导入库，从BaseML导入分类模块
from BaseML import Classification as cls
```

### 1. 实例化模型

```
# 实例化模型
model = cls('KNN')
```

### 2. 载入数据

```
# 指定数据集
model.load_tab_data('datasets/iris_training.csv')
```

### 3. 模型训练

```
# 模型训练
model.train()
```

### 4. 模型评估

```
# 模型评估
model.valid('datasets/iris_test.csv',metrics='acc')

# 评价指标可视化
model.metricplot()
```

### 5. 模型保存

```
# 模型保存
model.save('checkpoints/baseml_model/knn_iris.pkl')
```

参数为模型保存的路径，`.pkl`文件格式可以理解为将python中的数组、列表等持久化地存储在硬盘上的一种方式。

## 推理与应用

### 使用现有模型直接推理

对一组数据直接推理。

```
model = cls('KNN')
model.load('checkpoints/baseml_model/knn_iris.pkl')
y=model.inference([[5.9, 3.0, 4.2, 1.5]])
```

输出结果数据类型为`array`的一维数组。

可以在此基础上完成一个建议系统，输入鸢尾花的花萼长度、花萼宽度、花瓣长度、花瓣宽度，输出该鸢尾花所属的类别。

```
from BaseML import Classification as cls
model = cls('KNN')
model.load('checkpoints/baseml_model/knn_iris.pkl')

sepal_length = eval(input('花萼长度为(cm): '))
sepal_width = eval(input('花萼宽度为(cm): '))
petal_length = eval(input('花瓣长度为(cm): '))
petal_width = eval(input('花瓣宽度为(cm): '))

# 构建测试数据
data = [[sepal_length,sepal_width,petal_length,petal_width]]
# 用上面训练好的模型来做推理
result = model.inference(data)
print("该鸢尾花属于第{0}类".format(result))
```



## 快速体验

体验BaseML的最快速方式是通过OpenInnoLab平台。

OpenInnoLab平台为上海人工智能实验室推出的青少年AI学习平台，满足青少年的AI学习和创作需求，支持在线编程。在“项目”中查看更多，查找“BaseML”即可找到所有BaseML相关的体验项目。

AI项目工坊：[https://www.openinnolab.org.cn/pjlab/projects/list?backpath=/pjlab/ai/projects](https://www.openinnolab.org.cn/pjlab/projects/list?backpath=/pjlab/ai/projects)

（用Chrome浏览器打开效果最佳）

更多案例详见[下文](https://xedu.readthedocs.io/zh/master/baseml/projects.html)。
