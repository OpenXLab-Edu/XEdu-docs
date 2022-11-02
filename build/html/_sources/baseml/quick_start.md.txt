# BaseML快速入门

## 简介

BaseML提供了众多机器学习训练方法，可以快速训练和应用模型。

## 安装

`pip install baseml`或`pip install BaseML`

## 体验

可以在命令行输入BaseNN查看安装的路径，在安装路径内，可以查看提供的更多demo案例。

此处以用决策树方法配隐形眼镜为示例。

## 训练

### 0.引入包

```
# 导入库，从BaseML导入分类模块，简称cls
from BaseML import Classification as cls
```

### 1.实例化模型

```
# 实例化模型，模型名称选择CART（Classification and Regression Trees）
model=cls('CART')
```

### 2.载入数据

从路径载入数据：

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
model=cls('CART')
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
