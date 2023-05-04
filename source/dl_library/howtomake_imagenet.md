## 从零开始制作一个ImageNet格式数据集

## ImageNet格式数据集简介

ImageNet不仅是一个数据集、一项比赛，也是一种典型的数据集格式。分类任务中最经典的数据集类型就是[ImageNet格式](https://xedu.readthedocs.io/zh/latest/mmedu/introduction.html#imagenet)。

XEdu中MMEdu的图像分类模块数据集类型是[ImageNet](https://xedu.readthedocs.io/zh/latest/mmedu/introduction.html#imagenet)，包含三个文件夹和三个文本文件，文件夹内，不同类别图片按照文件夹分门别类排好，通过trainning_set、val_set、test_set区分训练集、验证集和测试集。文本文件classes.txt说明类别名称与序号的对应关系，val.txt说明验证集图片路径与类别序号的对应关系，test.txt说明测试集图片路径与类别序号的对应关系。如需训练自己创建的数据集，数据集需转换成[ImageNet格式](https://xedu.readthedocs.io/zh/latest/mmedu/introduction.html#imagenet)。这里，为您提供几种自己制作[ImageNet格式](https://xedu.readthedocs.io/zh/latest/mmedu/introduction.html#coco)数据集的方法。

### 选择1：巧用BaseDT的make_dataset函数制作

#### 第一步：整理图片

首先新建一个images文件夹用于存放图片，然后开始采集图片，您可以用任何设备拍摄图像，也可以从视频中抽取帧图像，需要注意，这些图像可以被划分为多个类别。每个类别建立一个文件夹，文件夹名称为类别名称，将图片放在其中。

#### 第二步：制作类别说明文件

在images文件夹同级目录下新建一个文本文件classes.txt，将类别名称写入，要求符合[ImageNet格式](https://xedu.readthedocs.io/zh/latest/mmedu/introduction.html#imagenet)。

参考示例如下：

```
cat
dog
```

此时前两步整理的文件夹应是如下格式：

```
|---images
    |---class1
          |----xxx.jpg/png/....
    |---class2
          |----xxx.jpg/png/....
    |---class3
          |----xxx.jpg/png/....
    |---classN
          |----xxx.jpg/png/....
classes.txt
```

#### 第三步：生成数据集

使用BaseDT库完成数据集制作。如需了解更多BaseDT库数据集处理的功能，详见[BaseDT的数据集格式转换](https://xedu.readthedocs.io/zh/latest/basedt/introduction.html#id7)部分。

```
from BaseDT.dataset import DataSet
ds = DataSet(r"my_dataset_catdog2") # 指定为生成数据集的路径
# 默认比例为train_ratio = 0.7, test_ratio = 0.1, val_ratio = 0.2
ds.make_dataset(r"catdog2", src_format="IMAGENET",train_ratio = 0.8, test_ratio = 0.1, val_ratio = 0.1)# 指定原始数据集的路径，数据集格式选择IMAGENET
```

#### 第四步：检查数据集

结合数据集检查提示对数据集进行调整，必要时可重做前几步，最后完成整个数据集制作。在训练的时候，只要通过`model.load_dataset`指定数据集的路径就可以了。

注：网上下载的图像分类数据集也可使用上述方法完成数据集处理。

### 选择2：按照标准方式制作

#### 第一步：整理图片

您可以用任何设备拍摄图像，也可以从视频中抽取帧图像，需要注意，这些图像可以被划分为多个类别。每个类别建立一个文件夹，文件夹名称为类别名称，将图片放在其中。

接下来需要对图片进行尺寸、保存格式等的统一，简单情况下的参考代码如下：

```plain
from PIL import Image
from torchvision import transforms
import os

def makeDir(folder_path):
    if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(folder_path)

classes = os.listdir('./my_dataset/training_set')
read_dir = './my_dataset/training_set/' # 指定原始图片路径
new_dir = './my_dataset/newtraining_set'
for cnt in range(len(classes)):
    r_dir = read_dir + classes[cnt] + '/'
    files = os.listdir(r_dir)
    for index,file in enumerate(files):
        img_path = r_dir + file
        img = Image.open(img_path)   # 读取图片
        resize = transforms.Resize([224, 224])
        IMG = resize(img)
        w_dir = new_dir + classes[cnt] + '/'
        makeDir(w_dir)
        save_path = w_dir + str(index)+'.jpg'
        IMG = IMG.convert('RGB')
        IMG.save(save_path)
```

#### 第二步：划分训练集、验证集和测试集

根据整理的数据集大小，按照一定比例拆分训练集、验证集和测试集，可手动也可以使用如下代码将原始数据集按照“6:2:2”的比例拆分。

```plain
import os
import shutil
# 列出指定目录下的所有文件名，确定分类信息
classes = os.listdir('./my_photo')

# 定义创建目录的方法
def makeDir(folder_path):
    if not os.path.exists(folder_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(folder_path)

# 指定文件目录
read_dir = './my_photo/' # 指定原始图片路径
train_dir = './my_dataset/training_set/' # 指定训练集路径
test_dir = './my_dataset/test_set/'# 指定测试集路径
val_dir = './my_dataset/val_set/'# 指定验证集路径

for cnt in range(len(classes)):
    r_dir = read_dir + classes[cnt] + '/'  # 指定原始数据某个分类的文件目录
    files = os.listdir(r_dir)  # 列出某个分类的文件目录下的所有文件名
    # files = files[:4000]
    # 按照6:2:2拆分文件名
    offset1 = int(len(files) * 0.6)
    offset2 = int(len(files) * 0.8)
    training_data = files[:offset1]
    val_data = files[offset1:offset2]
    test_data = files[offset2:]

    # 根据拆分好的文件名新建文件目录放入图片
    for index,fileName in enumerate(training_data):
        w_dir = train_dir + classes[cnt] + '/'  # 指定训练集某个分类的文件目录
        makeDir(w_dir)
        # shutil.copy(r_dir + fileName,w_dir + classes[cnt] + str(index)+'.jpg')
        shutil.copy(r_dir + fileName, w_dir + str(index) + '.jpg')
    for index,fileName in enumerate(val_data):
        w_dir = val_dir + classes[cnt] + '/'  # 指定测试集某个分类的文件目录
        makeDir(w_dir)
        # shutil.copy(r_dir + fileName, w_dir + classes[cnt] + str(index) + '.jpg')
        shutil.copy(r_dir + fileName, w_dir + str(index) + '.jpg')
    for index,fileName in enumerate(test_data):
        w_dir = test_dir + classes[cnt] + '/'  # 指定验证集某个分类的文件目录
        makeDir(w_dir)
        # shutil.copy(r_dir + fileName, w_dir + classes[cnt] + str(index) + '.jpg')
        shutil.copy(r_dir + fileName, w_dir + str(index) + '.jpg')


```

#### 第三步：生成标签文件

划分完训练集、验证集和测试集，我们需要生成“classes.txt”，“val.txt”和“test.txt”。其中classes.txt包含数据集类别标签信息，每行包含一个类别名称，按照字母顺序排列。“val.txt”和“test.txt”这两个标签文件的要求是每一行都包含一个文件名和其相应的真实标签。

可以手动完成，这里也为您提供一段用Python代码完成标签文件的程序如下所示，程序中设计了“val.txt”和“test.txt”这两个标签文件每行会包含类别名称、文件名和真实标签。

```plain
# 在windows测试通过
import os
# 列出指定目录下的所有文件名，确定类别名称
classes = os.listdir('./my_dataset/training_set')
# 打开指定文件，并写入类别名称
with open('./my_dataset/classes.txt','w') as f:
    for line in classes:
        str_line = line +'\n'
        f.write(str_line) # 文件写入str_line，即类别名称

test_dir = './my_dataset/test_set/' # 指定测试集文件路径
# 打开指定文件，写入标签信息
with open('./my_dataset/test.txt','w') as f:
    for cnt in range(len(classes)):
        t_dir = test_dir + classes[cnt]  # 指定测试集某个分类的文件目录
        files = os.listdir(t_dir) # 列出当前类别的文件目录下的所有文件名
        # print(files)
        for line in files:
            str_line = classes[cnt] + '/' + line + ' '+str(cnt) +'\n' 
            f.write(str_line) 

val_dir = './my_dataset/val_set/'  # 指定文件路径
# 打开指定文件，写入标签信息
with open('./my_dataset/val.txt', 'w') as f:
    for cnt in range(len(classes)):
        t_dir = val_dir + classes[cnt]  # 指定验证集某个分类的文件目录
        files = os.listdir(t_dir)  # 列出当前类别的文件目录下的所有文件名
        # print(files)
        for line in files:
            str_line = classes[cnt] + '/' + line + ' ' + str(cnt) + '\n'
            f.write(str_line)  # 文件写入str_line，即标注信息
```

如果您使用的是Mac系统，可以使用下面的代码。

```plain
# 本文件可以放在数据集的根目录下运行
import os
# 如果不是在数据集根目录下，可以指定路径
set_path = './' 

templist = os.listdir(set_path +'training_set')
# 处理mac的特殊文件夹
classes = []
for line in templist:
    if line[0] !='.':
        classes.append(line)
    
with open(set_path +'classes.txt','w') as f:
    for line in classes: 
        str_line = line +'\n'
        f.write(str_line) # 文件分行写入，即类别名称

val_dir = set_path +'val_set/'  # 指定验证集文件路径
# 打开指定文件，写入标签信息
with open(set_path +'val.txt', 'w') as f:
    for cnt in range(len(classes)):
        t_dir = val_dir + classes[cnt]  # 指定验证集某个分类的文件目录
        files = os.listdir(t_dir)  # 列出当前类别的文件目录下的所有文件名
        # print(files)
        for line in files:
            str_line = classes[cnt] + '/' + line + ' ' + str(cnt) + '\n'
            f.write(str_line)  # 文件写入str_line，即标注信息

test_dir = set_path +'test_set/' # 指定测试集文件路径
# 打开指定文件，写入标签信息
with open(set_path +'test.txt','w') as f:
    for cnt in range(len(classes)):
        t_dir = test_dir + classes[cnt]  # 指定测试集某个分类的文件目录
        files = os.listdir(t_dir) # 列出当前类别的文件目录下的所有文件名
        # print(files)
        for line in files:
            str_line = classes[cnt] + '/' + line + ' '+str(cnt) +'\n'
            f.write(str_line)

```

#### 第四步：给数据集命名

最后，我们将这些文件放在一个文件夹中，命名为数据集的名称。这样，在训练的时候，只要通过`model.load_dataset`指定数据集的路径就可以了。

### 选择3：巧用XEdu自动补齐功能快速制作

如果您觉得整理规范格式数据集有点困难，其实您只收集了图片按照类别存放，然后完成训练集（trainning_set）、验证集（val_set）和测试集（test_set）等的拆分，最后整理在一个大的文件夹下作为您的数据集也可以符合要求。此时指定数据集路径后同样可以训练模型，因为XEdu拥有检测数据集的功能，如您的数据集缺失txt文件，会自动帮您生成“classes.txt”，“val.txt”等（如存在对应的数据文件夹）开始训练。这些txt文件会生成在您指定的数据集路径下，即帮您补齐数据集。

> 选择2和选择3制作的数据集自查：数据集制作完成后如想要检查数据集，可使用BaseDT的[数据集格式检查](https://xedu.readthedocs.io/zh/latest/basedt/introduction.html#id9)功能，结合数据集检查提示对数据集进行调整，最后完成整个数据集制作。
