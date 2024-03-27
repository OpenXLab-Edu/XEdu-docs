# 用BaseNN训练搭建全连接神经网络（鸢尾花）

## 项目说明：

鸢尾花是一种常见的观赏类植物，在全球有着大量的变种。研究鸢尾花的分类问题，对解决植物分类识别问题有着重要的引导作用。本项目核心功能是完成使用经典的鸢尾花数据集完成鸢尾花分类，最后完成了一个简单的鸢尾花分类小应用，输入花萼长度、宽度、花瓣长度、宽度，可以输出预测结果。

数据集：UCI Machine Learning Repository: Iris Data Set（https://archive.ics.uci.edu/ml/datasets/Iris）

案例来源：《人工智能初步》人教地图p69

项目地址：[https://www.openinnolab.org.cn/pjlab/project?id=641bc2359c0eb14f22fdbbb1&sc=635638d69ed68060c638f979#public](https://www.openinnolab.org.cn/pjlab/project?id=641bc2359c0eb14f22fdbbb1&sc=635638d69ed68060c638f979#public)

## 项目步骤：

### 1.鸢尾花模型训练

#### 第0步 引入包（建议将库更新为最新版本再导入）

```python
# 导入BaseNN库
from BaseNN import nn
```

#### 第1步 声明模型

```python
model = nn('cls')
```

#### 第2步 载入数据

```python
train_path = 'data/iris_training.csv'
model.load_tab_data(train_path, batch_size=120)
```

#### 第3步 搭建模型

逐层添加，搭建起模型结构。注释标明了数据经过各层的维度变化。

```python
model.add(layer='linear',size=(4, 10),activation='relu') # [120, 10]
model.add(layer='linear',size=(10, 5), activation='relu') # [120, 5]
model.add(layer='linear', size=(5, 3), activation='softmax') # [120, 3]
```

以上使用`add()`方法添加层，参数`layer='linear'`表示添加的层是线性层，`size=(4,10)`表示该层输入维度为4，输出维度为10，`activation='relu'`表示使用ReLU激活函数。

#### 第4步 模型训练

模型训练可以采用以下代码：

```python
# 设置模型保存的路径
model.save_fold = 'checkpoints/iris_ckpt'
# 模型训练
model.train(lr=0.01, epochs=1000)
```

也可以使用继续训练：

```
checkpoint = 'checkpoints/basenn.pth'
model.train(lr=0.01, epochs=1000, checkpoint=checkpoint)
```

参数`lr`为学习率， `epochs`为训练轮数，`checkpoint`为现有模型路径，当使用`checkpoint`参数时，模型基于一个已有的模型继续训练，不使用`checkpoint`参数时，模型从零开始训练。

#### 第5步 模型测试

用测试数据查看模型效果。

```python
import numpy as np
# 用测试数据查看模型效果
model2 = nn('cls')
test_path = 'data/iris_test.csv'
test_x = np.loadtxt(test_path, dtype=float, delimiter=',',skiprows=1,usecols=range(0,4)) 
res = model2.inference(test_x, checkpoint="checkpoints/iris_ckpt/basenn.pth")
model2.print_result(res)

# 获取最后一列的真实值
test_y = np.loadtxt(test_path, dtype=float, delimiter=',',skiprows=1,usecols=4) 
# 定义一个计算分类正确率的函数
def cal_accuracy(y, pred_y):
    res = pred_y.argmax(axis=1)
    tp = np.array(y)==np.array(res)
    acc = np.sum(tp)/ y.shape[0]
    return acc

# 计算分类正确率
print("分类正确率为：",cal_accuracy(test_y, res))
```

用某组测试数据查看模型效果。

```python
# 用某组测试数据查看模型效果
data = [test_x[0]]
checkpoint = 'checkpoints/iris_ckpt/basenn.pth'
res = model.inference(data=data, checkpoint=checkpoint)
model.print_result(res) # 输出字典格式结果
```

参数`data`为待推理的测试数据数据，该参数必须传入值；

`checkpoint`为已有模型路径，即使用现有的模型进行推理。

上文介绍了借助BaseNN从模型训练到模型测试的简单方法，可前往[解锁BaseNN基本使用方法的教程](https://xedu.readthedocs.io/zh/master/basenn/introduction.html#id2)。

### 2.拓展：模型转换和后续应用

如果想要快速部署模型，可进行模型转换。BaseNN模型转换的代码如下：

```
from BaseNN import nn
model = nn('cls')
model.convert(checkpoint="checkpoints/iris_ckpt/basenn.pth",out_file="basenn_cd.onnx")
```

借助生成的示例代码，简单修改（如下所示），即可在本地或者硬件上运行（提前[安装XEduHub库](https://xedu.readthedocs.io/zh/master/xedu_hub/quick_start.html#id3)），甚至可以借助一些开源工具库做一个网页应用。

```
from XEdu.hub import Workflow as wf

# 模型声明
basenn = wf(task='basenn',checkpoint='basenn_cd.onnx')
# 待推理数据，此处仅以随机二维数组为例，以下为1个维度为4的特征
table = [[5.9, 3. , 4.2, 1.5]]
# 模型推理
res = basenn.inference(data=table)
# 标准化推理结果
result = basenn.format_output(lang="zh")
```

还可以借助一些开源工具库（如[PyWebIO](https://xedu.readthedocs.io/zh/master/how_to_use/scitech_tools/pywebio.html#webpywebio)）编写一个人工智能应用，如下代码可实现手动输入观察到的鸢尾花特征，输出花种判断。

```
from pywebio.input import *
from pywebio.output import *
from XEdu.hub import Workflow as wf

# 模型声明
basenn = wf(task='basenn',checkpoint='basenn_cd.onnx')
def pre():  
    a=input('请输入花萼长度：', type=FLOAT)
    b=input('请输入请花萼宽度：', type=FLOAT)
    c=input('请输入花瓣长度：', type=FLOAT)
    d=input('请输入花瓣宽度：', type=FLOAT)
    data = [a,b,c,d]
    result = basenn.inference(data=data)
    res = basenn.format_output(lang="zh")
    label=['山鸢尾','变色鸢尾','维吉尼亚鸢尾']
    put_text('预测结果是：', str(label[res[0]['预测值']]))
if __name__ == '__main__':
    pre()
```

运行效果如下：

![](../images/images/how_to_quick_start/pywebio.png)

更多模型转换和应用的教程详见[后文](https://xedu.readthedocs.io/zh/master/how_to_use/support_resources.html)。
