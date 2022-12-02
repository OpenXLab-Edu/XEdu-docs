# 版本更新记录

## 1.正式版

发布时间：2022年9月发布

开发计划：

1.实现以pip方式安装。

2.分MMEdu、BaseML、BaseNN三个功能模块。

3.工具持续迭代。

### 正式版更新记录

#### 1）MMEdu

##### V0.0.9 20221104

1. 检测模块训练函数支持device参数。
2. load_checkpoint()参数顺序更换。将checkpoint前置（第一个），device后置，可以只输入路径，而省略 "checkpoint="。
3. fast_infer错误反馈，补充错误情况，当fast_infer之前未使用load_checkpoint载入ckpt时会提示错误码305。
4. MMEdu.__ path __ 可正常返回环境中包所在地址。
5. 修复lenet 文件夹推理问题。

##### V0.0.1rc2 20221104

同V0.0.9，少依赖版本。

##### V0.0.8 20221102

1.加入错误反馈机制。

2.增加命令行字符画和简介。

3.提示目前支持的主干网络。

4.支持推理opencv、PIL读入的图片。

5.模型声明时允许读入配置文件，而不仅是模型名。

##### V0.0.1rc1 20221102

同V0.0.8，少依赖版本。

#### 2）BaseML

##### V0.0.1 20221110

1. load_dataset中设置了X和y的默认列，如果没有标明x_column和y_column，默认采用输入的所有列。但输入的是**txt**或**csv**格式的话，一定要标注出列号，否则报错。
2. inference()中加了参数verbose，默认值为True，表示会输出训练过程中的过程数据，False则不会。
3. train()中设置了参数validate（默认为True），表示会将输入的训练集划分为训练集和验证集，并输出验证集下的模型准确率。
4. 添加了图片读取处理模块ImageLoader，具体使用方式查看文件中的注释以及demo实现。
5. 对于加载数据集，添加了几个bool标记：shuffle, show, split, scale，分别表示是否打乱数据集、是否展示5条数据、是否划分数据集、是否对训练数据进行归一化。
6. 每个模型的初始化增加了参数字典方法，便于更高级的模型调参。

##### V0.0.2 20221110

1.  给每个类增加了docstring类型的注释，可以使用`cls.__doc__`查看拥有的算法以及类注释。
2. 更改了load_dataset函数的初始默认值，默认shuffle, 不展示前5条数据，不划分数据集，不进行数据归一化。
3. 添加了反归一化函数，可以将归一化后的数据转换为原数据，在`base.reverse_scale`函数中。

##### V0.0.3 20221115

1. 把 from BaseML import Classification  调用为Classification.cls  改成了 from BaseML import Classification as cls  调用为 cls(algorithm= ...)。

##### V0.0.4 20221121

1. 按照cls中的分类算法，给reg中的算法名进行了更改与添加，目前的回归算法有：['LinearRegression', 'CART', 'RandomForest',       'Polynomial', 'Lasso', 'Ridge', 'SVM', 'AdaBoost', 'MLP']。

## 2.测试版

### 1）0.5版

发布时间：2022.4

版本说明：整合图像分类（cls）、物体检测（det）两个核心模块，内置Pyzo、Jupyter，实现一键部署。

### 2）0.7版

发布时间：2022.6

版本说明：优化0.5版两个模块，新增自定义网络（BaseNN）模块。

### 3）XEdu 0.0.1版

发布时间：2022.6

版本说明：重构目录结构，建立MMEdu和BaseML两大模块。

### 4）MMEdu pip-0.0.1版

发布时间：2022.8

版本说明：发布内测版pip包。

