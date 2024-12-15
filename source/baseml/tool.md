# BaseML辅助工具

BaseML内置了一些辅助工具。

## 1.内置图像处理模型ImageLoader

`ImageLoader`是BaseML内置的图片处理模块，用于进行图像数字化处理，读取图片并提取其中的图像特征，如HOG特征和LBP特征，用以进行后续的机器学习任务。

使用此模块，可在BaseML载入数据前，对图片进行快速批量处理后再载入，并且能够完成单张图片的HOG特征提取（还可以更换为其他特征），示例代码如下。

```python
# 导入BaseML的图像处理模块
from BaseML import IMGLoader

# 定义一个提取单张图片HOG特征的函数
def read_hog_feature_single(file_path):
    # 创建ImageLoader实例并读取图片
    img_set = IMGLoader.ImageLoader(file_path,file_path,size = 128)
    # 对读取的图片进行预处理
    img = img_set.pre_process(file_path)
    # 提取图片的HOG特征
    feature = img_set.get_feature(img,method = 'hog')
    return feature

# 指定一张图片
img_path = 'test.jpg'
# 提取HOG特征
data = read_hog_feature_single(img_path)
# 打印HOG特征和其形状
print("HOG特征：",data)
print("图像形状：",data.shape)
```

IMGLoader.ImageLoader的参数说明：

- training_set_path (str): 图片训练集路径.
- testing_set_path (str): 图片测试集路径.
- label2id (dict, optional): 自定义的标签id字典. Defaults to {}.
- size (int, optional): 图像被resize的大小,尽量不要改size,否则使用lbp或者hog可能会出错,但是如果原始图像过小,可以调整size . Defaults to 128.


**HOG特征（方向梯度直方图）**

HOG（Histogram of Oriented Gradient）通过计算和统计图像局部区域的梯度方向直方图来构成特征。HOG特征能够很好地描述图像的局部形状信息，对光照和几何变换具有一定的不变性。


**LBP特征（局部二值模式）**

简介：LBP（Local Binary Patterns）是一种用来描述图像局部纹理特征的算子。它通过比较中心像素与其周围像素的灰度值，将比较结果转化为二进制数，从而得到图像的LBP特征。


如果没有指定参数，“get_feature”方法将返回展平状态的图像原始数据。


## 2.自带可视化工具

在做机器学习项目的过程中，可视化能帮助我们了解模型训练状态，评估模型效果，还能了解数据，辅助了解算法模型，改善模型。

BaseML中提供两种可视化方法：模型可视化及评价指标可视化。模型可视化可以通过测试数据及线条勾勒模型的大致形状，有助于解释和理解模型的内部结构。评价指标可视化显示了模型对于数据的拟合程度，描述了模型的性能，方便用户进行模型选择。使用可视化部分的前提是已经对模型进行初始化并且训练完成，否则可视化部分无法正常使用。

### 模型可视化

目前该模块只支持4类算法的可视化，分别为Classification中的KNN、SVM，Regression中的LinearRegression，Cluster中的Kmeans。调用方法为`model.plot()`。

### 评价指标可视化

目前该模块支持Classification、Regression中的所有算法及Cluster中的Kmeans算法，其他算法不支持。调用方法为`model.metricplot()`。

### 可视化调用限制

![](../images/baseml/limit.png)

### 1）快速体验训练过程可视化全流程！

以“KNN”模型的训练为例。


```Python
# 导入库，从BaseML导入分类模块
from BaseML import Classification as cls
# 实例化模型，模型名称选择KNN（K Nearest Neighbours）
model=cls('KNN')
# 载入数据集，并说明特征列和标签列
model.load_dataset('./lenses.csv', type ='csv', x_column = [1,2,3,4],y_column=[5])
# 模型训练
model.train()
# 模型可视化
model.plot()
# 评价指标可视化
model.metricplot()
```

### 快速体验推理过程可视化！

以“KNN”模型的推理为例。

```Python
# 导入库，从BaseML导入分类模块
from BaseML import Classification as cls
# 实例化模型，模型名称选择KNN（K Nearest Neighbours）
model=cls('KNN')
# 加载保存的模型参数
model.load('mymodel.pkl')
# 载入数据集，并说明特征列和标签列
model.load_dataset('./lenses.csv', type ='csv', x_column = [1,2,3,4],y_column=[5])
# 模型推理
model.inference()
# 模型可视化
model.plot()
# 评价指标可视化
model.metricplot()
```

实际上，训练过程可视化使用的数据与推理过程可视化使用的数据是相同的，均为数据集经过划分后的测试集（model.x_test）。

### 其他数据可视化



```Python
# 导入库，从BaseML导入分类模块
from BaseML import Classification as cls
# 实例化模型，模型名称选择KNN（K Nearest Neighbours）
model=cls('KNN')
# 加载保存的模型参数
model.load('mymodel.pkl')
# 模型推理
# test_data = [[0.2,0.4,3.2,5.6],
#             [2.3,1.8,0.4,2.3]]
model.inference(test_data)
# 模型可视化
# test_true_data = [[0],
#                  [1]]
model.plot(X=test_data, y_true=test_true_data)
# 评价指标可视化, 如果要使用其他数据进行测试，必须先加载之前的数据集
model.load_dataset('./lenses.csv', type ='csv', x_column = [1,2,3,4],y_column=[5])
model.metricplot(X=test_data, y_true=test_true_data)
```


