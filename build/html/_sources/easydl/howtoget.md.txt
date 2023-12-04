# 如何获取EasyDL系列工具

## 方式一：使用XEdu一键安装包中内置EasyDL

EasyDL系列工具内置在XEdu一键安装包中（如下图所示）。

飞书网盘：[https://p6bm2if73b.feishu.cn/drive/folder/fldcn67XTwhg8qIFCl8edJBZZQb](https://p6bm2if73b.feishu.cn/drive/folder/fldcn67XTwhg8qIFCl8edJBZZQb)

浦源CDN加速：[https://download.openxlab.org.cn/models/yikshing/bash/weight/x16](https://download.openxlab.org.cn/models/yikshing/bash/weight/x16)

下载一键安装包后，进入EasyDL文件夹，选择对应的工具，双击即可打开。（提示：不要关闭命令提示符窗口）

![](../images/easydl/bat.png)

## 方式二：通过pip命令来安装新版easy系列工具

### 1. 安装库
你可以通过pip命令来安装新版easy系列工具。

`pip install easy-xedu` 

更新库文件：

`pip install --upgrade easy-xedu`

安装EasyDL库之后，就可以在终端使用命令`easytrain`或`easyconvert`，打开EasyTrain无代码模型训练工具或
EasyConvert无代码模型转换工具。

### 2. 启动工具

打开拟定的工作目录（自定的任意目录，建议空白文件夹），在当前目录下的路径栏，输入cmd+回车，就可以打开当前目录下的命令行，在命令行中输入`easytrain`或`easyconvert`，回车运行，即可启动EasyTrain工具或EasyConvert工具。

![](../images/easydl/howtoget1.png)

- `easytrain`工具在启动后会在命令行中返回工具地址，复制地址到浏览器的网址栏中，即可打开工具。

![](../images/easydl/howtoget2.png)

- `easyconvert`工具在启动后会即可弹出工具窗口。

![](../images/easydl/howtoget4.png)

在启动以上两个任意一个工具的过程中会自动检测这个工作目录下是否存在以下目录结构，不符合以下结构会新建相应的文件夹。

![](../images/easydl/howtoget3.png)

- checkpoints（模型）
  - basenn_model
  - mmedu_cls_model
  - mmedu_det_model
- datasets（数据集）
  - basenn
  - mmedu_cls
  - mmedu_det
- my_checkpoints（自己训练的模型）


EasyDL系列工具的代码全部以CC协议开源，欢迎再次修改。