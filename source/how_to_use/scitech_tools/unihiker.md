# 开源硬件行空板

## 1.简介

行空板是一款拥有自主知识产权的国产教学用开源硬件，采用微型计算机架构，集成LCD彩屏、WiFi蓝牙、多种常用传感器和丰富的拓展接口。同时，其自带Linux操作系统和python环境，还预装了常用的python库，让广大师生只需两步就能开始python教学。

快速使用教程：[https://www.unihiker.com.cn/wiki/get-started](https://www.unihiker.com.cn/wiki/get-started)

## 2.简单使用教程

### 1.选择合适的编程方式

行空板自身作为一个单板计算机可以直接运行Python代码，同时默认开启了ssh服务及samba文件共享服务，因此可以用任意的文本编辑器编写代码，然后将代码传输到行空板即可运行。

教程：[https://www.unihiker.com.cn/wiki/mindplus](https://www.unihiker.com.cn/wiki/mindplus)

### 2.行空板Python库安装

- 打开MInd+，**连接行空板**，切换到**代码**标签页，点击**库管理**，此时库管理页面左上角显示行空板logo，说明此处显示的是行空板的库管理。

- 如果你需要**卸载库**或者**更新库**则可以在**库列表**中进行操作。

- 如果**推荐库**中没有你需要的，则可以切换到**PIP模式**，在输入框中输入库名字安装，右上角可以切换不同的源，例如此处安装**dominate**则可以输入`dominate`或者完整指令 `pip install dominate`，或者指定版本安装`pip install dominate==2.5.1`,提示**"Successfully installed xxxx"**即表示安装成功。

  ![](../../images/basedeploy/install_1.png)

例如可以在行空板安装我们的深度学习工具库[XEduHub](https://xedu.readthedocs.io/zh-cn/master/xedu_hub.html)和模型部署库[BaseDeploy](https://xedu.readthedocs.io/zh-cn/master/basedeploy.html)，Mind+中甚至可以添加XEduHub积木套件和BaseDeploy积木套件，更多说明详见[在Mind+中使用XEduHub](https://xedu.readthedocs.io/zh-cn/master/xedu_hub/mindplus_xeduhub.html)和[Mind+中的BaseDeploy积木块](https://xedu.readthedocs.io/zh-cn/master/basedeploy/introduction.html#mind-basedeploy)。

![](../../images/basedeploy/install_lab.png)

## 3.部署模型到行空板

我们可以通过XEduHub库的方式部署。在XEduHub中，我们介绍了一个[实时摄像头的人体关键点识别案例](https://xedu.readthedocs.io/zh-cn/master/xedu_hub/projects.html#id2)，这里，我们首先需要安装对应的库文件：
```
pip install xedu-python==0.2.0 onnx==1.13.0 onnxruntime==1.13.1
```
安装完成后，可以通过python代码进行部署。代码可以直接迁移前面的案例，模型部分不需要做任何改动，可视化的部分，为了适配行空板的屏幕尺寸，我们参考[相关资料](https://mc.dfrobot.com.cn/thread-312867-1-1.html)做了适配调整，代码如下：

```python
from XEdu.hub import Workflow as wf
import cv2

body = wf(task='pose_body17')# 实例化pose模型
det = wf(task='det_body')# 实例化detect模型

#False:不旋转屏幕（竖屏显示，上下会有白边）
#True：旋转屏幕（横屏显示）
screen_rotation = True 

cap = cv2.VideoCapture(0)   #设置摄像头编号，如果只插了一个USB摄像头，基本上都是0
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  #设置摄像头图像宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240) #设置摄像头图像高度
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     #设置OpenCV内部的图像缓存，可以极大提高图像的实时性。

cv2.namedWindow('camera',cv2.WND_PROP_FULLSCREEN)    #窗口全屏
cv2.setWindowProperty('camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)   #窗口全屏

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    bboxs = det.inference(data=frame,thr=0.3)
    img = frame
    for i in bboxs:
        keypoints,img =body.inference(data=img,img_type='cv2',bbox=i)
    for [x1,y1,x2,y2] in bboxs: # 画检测框
        cv2.rectangle(img, (int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
    if screen_rotation: #是否要旋转屏幕
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) #旋转屏幕
    cv2.imshow('camera', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    
cap.release()
cv2.destroyAllWindows()
```

参考资料1-AI猜拳机器人：[https://mc.dfrobot.com.cn/thread-315543-1-1.html](https://mc.dfrobot.com.cn/thread-315543-1-1.html)


参考资料2-智能音箱：[https://xedu.readthedocs.io/zh-cn/master/how_to_use/support_resources/works/p4-smartspeaker.html](https://xedu.readthedocs.io/zh-cn/master/how_to_use/support_resources/works/p4-smartspeaker.html)


参考资料3-手搓图像识别硬件部署应用：[https://www.bilibili.com/video/BV1364y1T771?p=3](https://www.bilibili.com/video/BV1364y1T771?p=3)


更多AI用法：[https://www.unihiker.com.cn/wiki/ai_project](https://www.unihiker.com.cn/wiki/ai_project)

在浦育平台硬件工坊也可支持连接行空板，参考项目-行空板与XEdu：[https://openinnolab.org.cn/pjlab/project?id=65bc868615387949b281d622&backpath=/pjedu/userprofile?slideKey=project&type=OWNER#public](https://openinnolab.org.cn/pjlab/project?id=65bc868615387949b281d622&backpath=/pjedu/userprofile?slideKey=project&type=OWNER#public)