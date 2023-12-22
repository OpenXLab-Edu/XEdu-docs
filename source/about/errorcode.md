# XEdu的错误码

## MMEdu的错误码
本文档定义了MMEdu的基本错误反馈。错误描述用英文（考虑到国际化）描述，同时输出错误代码。代码和目录编号一致，“1.1”的错误代码为“101”。

标准错误输出信息：Error Code: -编号,错误英文提示
示例：Error Code: -101, No such dataset file:XX/XXX/XXX/
### 1.文件路径错误
#### 1.1 数据集的路径错误
只能是存在的目录

英文提示设计：No such dataset directory:XX/XXX/XXX/
#### 1.2 权重文件的路径错误
只能是存在的文件

英文提示设计：No such checkpoint file:XX/XXX/XXX.pth
#### 1.3 要推理文件的路径错误
只能是存在的文件

英文提示设计：No such file:XX/XXX/XXX.jpg
### 2.文件类型错误
#### 2.1 数据集的类型错误
只能是目录，而且目录文件要符合要求。

如果是imagenet：需要检查文件夹名称+txt名称，如果不存在，给出下载地址。指定的图片类别数量必须和数据集一致。val.txt行数也与实际图片数量一致。

如果是coco，类似检查。

英文提示设计：Dateset file type error

case1：传入参数类型不是字符串
- Error Code - 201. Dataset file type error, which should be <class 'str'> instead of <class 'int'>
case 2：数据集路径存在，且为字符串，但其中文件缺失
- Error Code - 201. Dataset file type error. No such file: '../dataset/cls/hand_gray/classes.txt'
case 3：验证集图片数和val.txt中不一致
- Error Code - 201. Dataset file type error. The number of val set images does not match that in val.txt.
case 4: 数据集中图片损坏

图片类型为gif的，也属于损坏。
- Error Code -201. The image file ../../dataset/xx.jpg is damaged.
#### 2.2 权重文件的类型错误
只能是pth

英文提示设计：Checkpoint file type error
#### 2.3 要推理文件的类型错误
只能是图片文件，如jpg、png、bmp等受支持的文件格式

英文提示设计：File type error
### 3.参数值错误（等于号右边）
#### 3.1 device设置错误
只能是cpu和cuda

英文提示设计：No such argument
#### 3.2 主干网络名称错误
目前只支持‘LeNet’、‘MobileNet’、‘ResNet18’、‘ResNet50’

英文提示设计：No such argument
#### 3.3 validate设置错误
只能是True和False

英文提示设计：No such argument
#### 3.4 推理图片格式错误
变量类型必须是str（图片路径）或list【str】（多张图）或numpyarray（点阵图）。（bug目前可视化仅支持路径）
- Error Code: - 304. No such argument: (1, 'asd') which is <class 'tuple'>
#### 3.5 fast_infer之前，未正确使用load_checkpoint载入权重

### 4.预留给网络相关错误

#### 4.4 网络名称不存在（不存在的网络名称）
404
### 5. 参数名称错误（等于号左边）
#### 5.1 传入的参数名称错误
无此参数，请重新输入。

英文提示设计：No such parameter 
- Error Code: - 501. No such parameter: ig
