# 快速体验BaseML

## 简介

BaseML库提供了众多机器学习训练方法，可以快速训练和应用模型。

## 安装

`pip install baseml`或`pip install BaseML`

## 体验

可以在命令行输入BaseML查看安装的路径，在安装路径内，可以查看提供的更多demo案例。

下面我们以用“决策树方法配隐形眼镜”案例为示例，体验用BaseML做第一个机器学习项目！体验更多案例请看附录。

## 训练

### 0. 引入包

```
# 导入库，从BaseML导入分类模块
from BaseML import Classification as cls
```

### 1. 实例化模型

```
# 实例化模型，模型名称选择CART（Classification and Regression Trees）
model=cls('CART')
```

### 2. 载入数据

```
# 载入数据集，并说明特征列和标签列
model.load_dataset('./lenses.csv', type ='csv', x_column = [1,2,3,4],y_column=[5])
```

`x_column`表示特征列，`y_column`表示标签列。

### 3. 模型训练

```
# 模型训练
model.train()
```

## 推理

### 使用现有模型直接推理

对一组数据直接推理。

```
model=cls('CART')
model.load('mymodel.pkl')
y=model.inference([[1,  1,  1,  1]])
```

输出结果数据类型为`array`的一维数组。

### 输出推理结果

定义`label`存储标签名称，根据`label`和推理结果输出真实标签。

```python
label=['不适合佩戴', '软材质', '硬材质']
print(label[y[0]-1])# 这里-1是因为python中的数组下标从0开始，而推理结果从1开始，因此需要-1才能输出对应的标签
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

OpenInnoLab平台为上海人工智能实验室推出的青少年AI学习平台，满足青少年的AI学习和创作需求，支持在线编程。在“项目”中查看更多，查找“BaseML”即可找到所有BaseML相关的体验项目。

AI项目工坊：[https://www.openinnolab.org.cn/pjlab/projects/list?backpath=/pjlab/ai/projects](https://www.openinnolab.org.cn/pjlab/projects/list?backpath=/pjlab/ai/projects)

（用Chorme浏览器打开效果最佳）

更多案例详见[下文](https://xedu.readthedocs.io/zh/master/baseml/projects.html)。
