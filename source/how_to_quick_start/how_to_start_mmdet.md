# 用MMEdu训练SSD_Lite目标识别模型（猫狗）

## 项目说明：

目标检测任务是图像分类任务的进阶任务，图像分类任务只有一个子任务：分类，而目标检测任务有两个任务：定位和分类，按照**图像中目标的数量**可分为单目标检测和多目标检测。如果一张图片里有两只猫，图像分类的模型可能可以识别出是猫。但是如果是这张图又有猫又有狗，那图像分类模型就肯定识别不出来了。本项目使用的是浦育平台公开的[猫狗目标检测数据集](https://openinnolab.org.cn/pjlab/dataset/6407fdcd9c0eb14f2297218d)，训练一个训练SSD_Lite目标识别模型。另外XEdu中MMEdu的目标检测模块支持的数据集类型是COCO，数据集需转换成COCO格式。如何从零开始制作符合要求的数据集详见[后文](https://xedu.readthedocs.io/zh/master/how_to_use/dl_library/howtomake_coco.html)。

项目地址：[https://openinnolab.org.cn/pjlab/project?id=64055f119c0eb14f22db647c&sc=62f34141bf4f550f3e926e0e#public](https://openinnolab.org.cn/pjlab/project?id=64055f119c0eb14f22db647c&sc=62f34141bf4f550f3e926e0e#public)

## 项目步骤：

### 1.模型训练

#### 第0步 导入基础库（建议将库更新为最新版本再导入）

```
from MMEdu import MMDetection as det
```

#### 第1步 实例化模型（选择SSD_Lite）

```
model = det(backbone='SSD_Lite')
```

#### 第2步 配置基本信息

AI模型训练的基本信息有三类，分别是：图片分类的类别数量（`model.num_classes`），模型保存的路径（`model.save_fold`）和数据集的路径（`model.load_dataset`）。

```
model.num_classes = 2 # 猫和狗共2类
model.load_dataset(path='/data/H47U12/cat_dog_det') 
model.save_fold = 'checkpoints/det_model/catdogs' 
```

#### 第3步 开始训练模型

```
model.train(epochs=10 ,lr=0.001,batch_size=4, validate=True)
```

训练过程中观察输出的每一轮bbox_mAP的变化，判断模型在验证集上的准确率。

### 2.基于预训练模型继续训练

全新开始训练一个模型，一般要花较长时间。因此我们强烈建议在预训练模型的基础上继续训练，哪怕你要分类的数据集和预训练的数据集并不一样。

```
model.num_classes = 2 # 猫和狗共2类
model.load_dataset(path='/data/H47U12/cat_dog_det') 
# 预训练模型权重路线
checkpoint = 'checkpoints/pretrain_ssdlite_mobilenetv2.pth'
model.save_fold = 'checkpoints/det_model/catdogs_pretrain' 
#启动cpu容器将device='cpu'，启动GPU容器将device='cuda'
model.train(epochs=10, lr=0.001, validate=True, batch_size = 4, device='cuda', checkpoint=checkpoint)
```

预训练模型下载地址：[https://p6bm2if73b.feishu.cn/drive/folder/fldcnxios44vrIOV9Je3wPLmExf](https://p6bm2if73b.feishu.cn/drive/folder/fldcnxios44vrIOV9Je3wPLmExf)

注：一般训练目标检测模型耗时较久，浦育平台可启动GPU服务器，建议去浦育平台完成模型训练，启动GPU服务器后便可以在训练参数中添加`device='cuda'`启动GPU训练。

### 3.模型测试（用新的图片完成推理）

#### 第0步 导入基础库（建议将库更新为最新版本再导入）

```
from MMEdu import MMClassification as cls
```

#### 第1步 实例化模型

```
model = cls(backbone='SSD_Lite')
```

#### 第2步 指定模型权重文件的所在路径

```
checkpoint = 'checkpoints/det_model/best_bbox_mAP_epoch_7.pth' # 指定权重文件路径
```

第1步和第2步的模型需对应，首先模型权重需存在，同时还需该模型训练时实例化模型时选择的网络与推理时一致。

#### 第3步 指定图片

```
img_path = 'picture/2.png' # 指定图片路径
```

#### 第4步 开始推理

```
result = model.inference(image=img, show=True, checkpoint = checkpoint,device='cuda') # 模型推理
model.print_result(result) # 结果转换为中文输出
```

上文简单介绍了如何用MMEdu训练一个目标检测模型，更多关于MMEdu模型训练和推理的方法详见请前往[揭秘MMEdu的目标检测模块](https://xedu.readthedocs.io/zh/master/mmedu/mmdetection.html#mmdetection)。

### 4.模型转换和应用

同样的，可以在模型应用前先完成模型转换，目标检测模型转换的代码风格和图像分类类似。

```
from MMEdu import MMDetection as det
model = det(backbone='SSD_Lite')
checkpoint = 'checkpoints/best_bbox_mAP_epoch_7.pth'
out_file='cats_dogs_det.onnx' # 指定输出的文件即转换后的文件
model.convert(checkpoint=checkpoint, backend="ONNX", out_file=out_file)
```

模型应用的基础代码：

```
from XEdu.hub import Workflow as wf
mmdet = wf(task='mmedu',checkpoint='cats_dogs_det.onnx')# 指定使用的onnx模型
result, result_img =  mmdet.inference(data='/data/H47U12/cat_dog_det/images/valid/001.jpg',img_type='cv2')# 进行模型推理
format_result = mmdet.format_output(lang="zh")# 推理结果格式化输出
mmdet.show(result_img)# 展示推理结果图片
mmdet.save(result_img,'new.jpg')# 保存推理结果图片
```

##### 6）部署到硬件

此时您可以挑选自己熟悉的硬件，去做自己训练并完成转换的模型部署啦，只需要下载转换的ONNX模型，在硬件上安装库即可。最简单的方式是借助摄像头，再使用OpenCV这个轻松完成图像和视频处理的工具库，实现猫狗实时检测。

```
from XEdu.hub import Workflow as wf
import cv2
cap = cv2.VideoCapture(0)
mmdet = wf(task='mmedu',checkpoint='cats_dogs_det.onnx')
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    result, result_img=  mmdet.inference(data=img,img_type='cv2')
    format_result = mmdet.format_output(lang="zh")
    cv2.imshow('video', result_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    
cap.release()
cv2.destroyAllWindows()
```

更多模型应用与部署的介绍详见[后文](https://xedu.readthedocs.io/zh/master/how_to_use/support_resources/model_convert.html#id9)。