# 版本更新记录

## 1. 正式版

发布时间：2022年9月发布

### 开发计划：

1.实现以pip方式安装。

2.分MMEdu、BaseML、BaseNN三个功能模块。

3.工具持续迭代。

### 正式版更新记录

#### 1）MMEdu

##### V0.1.5 20230203

1. det修正infer和convert中类别数量的问题。
1. det模型训练时会自动保存best_map的权重。
1. det 规范化数据集文件夹名称。
1. 修复SSD_Lite类名传递不正确的问题。

##### V0.1.4 20230106

1. cls+det同时增加可选batch_size功能。
1. det补充SSD和yolov3。
1. det输出格式由xywh修正为x1y1x2y2。

##### V0.1.3 20221222

1. det增加模型转化功能。
2. cls+det更新模型转化功能，参数调整，会额外输出config文件。

##### V0.1.2 20221215

cls：

1. cls检查数据集中图片shape，指出损坏图片。检查图片出现损坏时，抛出错误码`The image file xxx is damaged`。

2. 数据集如缺少txt，自动生成

   case1：数据集缺少classes.txt, val.txt ，会自动生成并提示, eg，“生成val.txt”。

   case2：如缺少test_set，可正常训练，但不会生成test.txt 。（不影响正常功能）

   case3：如缺少val_set，可训练，但不能验证，即train函数中validate参数不能为True。（功能受损，看不到准确率，但还是可以训练出模型）。

   其他：

   允许数据集中出现其他类别的文件，eg,csv；

   数据集中test_set可以不按照类别存放。

3. 检查写权限，确定写到哪里

   innolab上数据集没有读写权限，则将txt生成至项目内，文件夹名为dataset_txt，内含classes.txt，val.txt。(若有读写权限则生成至数据集路径内）


​	4.加入模型转换convert()函数，pth转onnx。

det：

1. det增加支持PIL和np array 输入功能。图片形式可以通过PIL和np数组进行输入，PIL和数组列表也支持输入。
2. 参考cls，det增加相关错误码。

##### V0.1.1 20221118

​	支持读入pil，np格式数据。

##### V0.1.0 20221111

1. train和infer的`device=cuda`检查`torch.cuda.is_available()`，`device=cpu`当cuda可用时提示可以使用cuda加速。
2. 文件夹推理LeNet无误。
3. `fast_infer`支持LeNet。

##### V0.1.0rc2 20221111

​	同V0.0.9，少依赖版本。

##### V0.0.9 20221104

1. 检测模块训练函数支持device参数。
2. `load_checkpoint()`参数顺序更换。将checkpoint前置（第一个），`device`后置，可以只输入路径，而省略 "`checkpoint=`"。
3. fast_infer错误反馈，补充错误情况，当`fast_infer`之前未使用l`oad_checkpoint`载入`ckpt`时会提示错误码305。
4. `MMEdu.__ path __` 可正常返回环境中包所在地址。
5. 修复lenet 文件夹推理问题。

##### V0.0.1rc2 20221104

​	同V0.0.9，少依赖版本。

##### V0.0.8 20221102

1. 加入错误反馈机制。

2. 增加命令行字符画和简介。

3. 提示目前支持的主干网络。

4. 支持推理opencv、PIL读入的图片。

5. 模型声明时允许读入配置文件，而不仅是模型名。

##### V0.0.1rc1 20221102

​	同V0.0.8，少依赖版本。

#### 2）BaseML

##### V0.0.6 20230217

1. 与MMEdu的错误提示码风格进行了统一，并在此基础上进行了BaseML部分的补充。
2. 所有库类代码应用了PEP8代码规范，使得代码结构与语句更加美观。

##### V0.0.5 20230210

1. 完成模型可视化和评测指标可视化两个库。目前只有4种算法支持可视化，大部分模型支持评测指标可视化，少部分不支持。
2. 引入yellowbrick库，用于评测指标可视化。
3. 修改了load_dataset函数，cls和reg默认split=True, 即划分为训练和测试集， 聚类和降维默认不划分。
4. 加入了警告（蓝色字体）和报错（红色字体），但待与MMEdu的风格统一。

##### V0.0.4 20221121

​	按照cls中的分类算法，给reg中的算法名进行了更改与添加，目前的回归算法有：['LinearRegression', 'CART', 'RandomForest',       'Polynomial', 'Lasso', 'Ridge', 'SVM', 'AdaBoost', 'MLP']。

##### V0.0.3 20221115

​	把 `from BaseML import Classification`  调用为`Classification.cls`  改成了 `from BaseML import Classification as cls`  调用为 `cls(algorithm= ...)`。

##### V0.0.2 20221110

1.  给每个类增加了docstring类型的注释，可以使用`cls.__doc__`查看拥有的算法以及类注释。
2.  更改了`load_dataset`函数的初始默认值，默认shuffle, 不展示前5条数据，不划分数据集，不进行数据归一化。
3.  添加了反归一化函数，可以将归一化后的数据转换为原数据，在`base.reverse_scale`函数中。

##### V0.0.1 20221110

1. `load_dataset`中设置了X和y的默认列，如果没有标明`x_column`和`y_column`，默认采用输入的所有列。但输入的是**txt**或**csv**格式的话，一定要标注出列号，否则报错。
2. `inference()`中加了参数`verbose`，默认值为True，表示会输出训练过程中的过程数据，False则不会。

3. `train()`中设置了参数`validate`（默认为True），表示会将输入的训练集划分为训练集和验证集，并输出验证集下的模型准确率。

4. 添加了图片读取处理模块ImageLoader，具体使用方式查看文件中的注释以及demo实现。

5. 对于加载数据集，添加了几个bool标记：shuffle, show, split, scale，分别表示是否打乱数据集、是否展示5条数据、是否划分数据集、是否对训练数据进行归一化。

6. 每个模型的初始化增加了参数字典方法，便于更高级的模型调参。


#### 3）BaseNN

##### V0.0.5 20221215

1. 可视化特征，只有传统意义上的层才计数，relu，reshape，softmax不计数;且当输入为二维图像时，展示可视化的图像，输入为一维数据时，生成txt保存每层之后的输出。
2. 加入随机数种子，确保当指定种子后，反复训练可以得到完全一致的结果。
3. 可选损失函数，可选评价指标。

##### V0.0.4 20221202

​	参数控制可视化，一整张图or一系列图。

##### V0.0.3 20221116

​	增加提取特征，可视化特征功能。

#### 4）XEdu一键安装包

##### V1.1 20221220

支持模块：MMEdu0.1.2（支持cls和det），BaseNN0.0.5，BaseML0.0.3（支持cls、reg和clt）

内置编辑器：jupyter、pyzo、三个可视化工具（EasyTrain、EasyInference和EasyAPI）

## 2. 测试版

### XEdu 0.1.0版

发布时间：2022.11

版本说明：优化MMEdu、BaseML、BaseNN等模块，增加EasyAPI.bat、EasyInference.bat、EasyTrain.bat三个可视化工具，更新所有示例代码。

### MMEdu pip-0.0.1版

发布时间：2022.8

版本说明：发布内测版pip包。

### XEdu 0.0.1版

发布时间：2022.6

版本说明：重构目录结构，建立MMEdu和BaseML两大模块。

### 0.7版

发布时间：2022.6

版本说明：优化0.5版两个模块，新增自定义网络（BaseNN）模块。

### 0.5版

发布时间：2022.4

版本说明：整合图像分类（cls）、物体检测（det）两个核心模块，内置Pyzo、Jupyter，实现一键部署。

