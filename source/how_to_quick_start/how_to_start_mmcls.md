# 案例四：用MMEdu训练LeNet图像分类模型（手写体）

## 项目说明：

MMEdu是人工智能视觉算法集成的深度学习开发工具，目前图像分类模块MMClassifiation支持的SOTA模型有LeNet、MobileNet、ResNet18、ResNet50等，支持训练的数据集格式为ImageNet。更多关于MMClassifiation功能详见请前往[解锁MMEdu的图像分类模块](https://xedu.readthedocs.io/zh-cn/master/mmedu/mmclassification.html#mmclassification)。
本项目使用MMEdu的图像分类模块MMClassification，根据经典的手写体ImageNet格式数据集，训练LeNet模型实现手写体识别。

项目地址：[https://openinnolab.org.cn/pjlab/project?id=64a3c64ed6c5dc7310302853&sc=62f34141bf4f550f3e926e0e#public](https://openinnolab.org.cn/pjlab/project?id=64a3c64ed6c5dc7310302853&sc=62f34141bf4f550f3e926e0e#public)


数据集来源：mnist数据集，来源于National Institute of Standards and Technology，改编自MNIST。另外MMEdu图像分类模块要求的数据集格式为ImageNet格式，包含三个文件夹和三个文本文件，文件夹内，不同类别图片按照文件夹分门别类排好，通过trainning_set、val_set、test_set区分训练集、验证集和测试集。文本文件classes.txt说明类别名称与序号的对应关系，val.txt说明验证集图片路径与类别序号的对应关系，test.txt说明测试集图片路径与类别序号的对应关系。如何从零开始制作符合要求的数据集详见[后文](https://xedu.readthedocs.io/zh-cn/master/how_to_use/dl_library/howtomake_imagenet.html)。



## 项目步骤：

### 任务一：训练LeNet手写体识别模型

#### 第0步 导入基础库（建议将库更新为最新版本再导入）

```python
from MMEdu import MMClassification as cls
```

#### 第1步 实例化模型（选择LeNet）

```python
model = cls(backbone='LeNet') # 实例化模型为model
```

#### 第2步 配置基本信息

AI模型训练时需要配置的基本信息有三类，分别是：图片分类的类别数量（`model.num_classes`），模型保存的路径（`model.save_fold`）和数据集的路径（`model.load_dataset`）。

```python
model.num_classes = 10 # 手写体的类别是0-9，共十类数字
model.load_dataset(path='/data/MELLBZ/mnist') # 从指定数据集路径中加载数据
model.save_fold = 'checkpoints/cls_model/230226' # 模型保存路径，可自定义最后一个文件名
```

#### 第3步 开始训练模型

```python
model.train(epochs=10, lr=0.01, validate=True) 
```

注：如有GPU可启动GPU训练，在训练函数中加个参数`device='cuda'`，则训练代码变成如下这句。

```python
model.train(epochs=10, lr=0.01, validate=True, device='cuda')
```

训练过程中观察输出的每一轮acc的变化，判断模型在验证集上的准确率。

### 任务二：模型测试（用新的图片完成推理）

#### 第0步 导入基础库（建议将库更新为最新版本再导入）

```python
from MMEdu import MMClassification as cls
```

#### 第1步 实例化模型

```python
model = cls(backbone='LeNet')
```

#### 第2步 指定模型权重文件的所在路径

```python
checkpoint = 'checkpoints/cls_model/best_accuracy_top-5_epoch_4.pth' # 指定权重文件路径
```

第1步和第2步的模型需对应，首先模型权重需存在，同时还需该模型训练时实例化模型时选择的网络与推理时一致。

#### 第3步 指定图片

```python
img_path = 'picture/2.png' # 指定图片路径
```

#### 第4步 开始推理

```python
result = model.inference(image=img_path, show=True, checkpoint = checkpoint) # 模型推理
model.print_result(result) # 结果转换为中文输出
```

上文简单介绍了如何用MMEdu训练一个图像分类模型，更多关于MMEdu模型训练和推理的方法详见请前往[解锁MMEdu的图像分类模块](https://xedu.readthedocs.io/zh-cn/master/mmedu/mmclassification.html#mmclassification)[https://xedu.readthedocs.io/zh-cn/master/mmedu/mmclassification.html#mmclassification](https://xedu.readthedocs.io/zh-cn/master/mmedu/mmclassification.html#mmclassification)。 

### 拓展：模型转换和应用

当一个深度学习模型训练完成后，最终的任务是要结合其他编程工具，编写一个人工智能应用。一般来说，这些规模较小的模型都是会运行在一些边缘设备（指性能较弱的移动端和嵌入式设备）上。此时你可以使用MMEdu的模型转换工具将模型转换为ONNX格式，便于部署。

```python
from MMEdu import MMClassification as cls
model = cls(backbone='LeNet')
checkpoint = 'checkpoints/cls_model/best_accuracy_top-5_epoch_4.pth'
out_file="cls.onnx"
model.convert(checkpoint=checkpoint, out_file=out_file)
```

接下来无需借助MMEdu库（安装涉及较多依赖库），只需借助XEuHub库便可完成推理。

```python
from XEdu.hub import Workflow as wf
mmcls = wf(task='mmedu',checkpoint='cls.onnx')# 指定使用的onnx模型
result, result_img =  mmcls.inference(data='test.jpg',img_type='cv2')# 进行模型推理
format_result = mmcls.format_output(lang="zh")# 推理结果格式化输出
mmcls.show(result_img)# 展示推理结果图片
mmcls.save(result_img,'new.jpg')# 保存推理结果图片
```

编写一个人工智能应用并没有那么困难，比如可以借助[Gradio](https://xedu.readthedocs.io/zh-cn/master/how_to_use/scitech_tools/gradio.html#webgradio)这个开源的用于快速原型设计和部署机器学习模型的交互式界面的工具库就能快速搭建一个简易的模型展示应用，如下代码可实现在一个网页上传一张图片，返回推理结果。

```python
import gradio as gr
from XEdu.hub import Workflow as wf
mm = wf(task='mmedu',checkpoint='cls.onnx') 

def predict(img):
    res,img = mm.inference(data=img,img_type='cv2') # 模型推理
    result = mm.format_output(lang="zh") # 标准化推理结果
    text1 = '预测结果：'+result['预测结果']
    text2 = '标签：'+str(result['标签'])
    return text1,text2

image = gr.Image(type="filepath")
demo = gr.Interface(fn=predict, inputs=image, outputs=["text","text"])
demo.launch(share=True)
```

更多模型转换和应用的内容请看[模型转换和应用](https://xedu.readthedocs.io/zh-cn/master/how_to_use/support_resources/model_convert.html)[(https://xedu.readthedocs.io/zh-cn/master/how_to_use/support_resources/model_convert.html)](https://xedu.readthedocs.io/zh-cn/master/how_to_use/support_resources/model_convert.html)。

