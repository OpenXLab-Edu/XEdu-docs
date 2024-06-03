# XEduHub项目案例集

借助XEduHub可以实现应用多元AI模型去解决复杂的问题。

## 目标检测+关键点检测

以下代码可以将一张图片中所有的手识别出来，并对每一只手提取关键点。这可以提高关键点检测的准确度和实现多目标检测的任务。

具体实现方式为：我们首先使用`det_hand`进行手目标检测，拿到所有的检测框`bbox`。随后对每个检测框中的手进行关键点提取。

```python
from XEdu.hub import Workflow as wf # 导入库
det  = wf(task='det_hand') # 实例化目标检测模型
hand = wf(task='pose_hand21') # 实例化关键点检测模型
img_path = 'demo/hand3.jpg' # 指定进行推理的图片
bboxs = det.inference(data=img_path) # 目标检测推理
display_img = img_path # 初始化用于显示的图像
for i in bboxs:
    keypoints,display_img = hand.inference(data=display_img,img_type='cv2',bbox=i) # 关键点检测推理
hand.show(display_img) # 结果可视化
```

## 实时人体关键点识别

以下代码可以实时检测摄像头中出现的多个人，并对每一个人体提取关键点。

具体实现方式为：我们首先将实时视频中每一帧的图像进行人体目标检测，拿到所有的检测框`bbox`及其坐标信息，绘制检测框。随后对每个检测框中的人体进行关键点提取。

```python
from XEdu.hub import Workflow as wf
import cv2
cap = cv2.VideoCapture(0)
body = wf(task='pose_body17')# 实例化pose模型
det = wf(task='det_body')# 实例化detect模型
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
    cv2.imshow('video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    
cap.release()
cv2.destroyAllWindows()
```

## 拓展：视频中的人体关键点识别

该项目可以识别视频中出现的人体的关键点。

具体实现方式与上面的代码类似，区别就是从摄像头的实时视频流变成了本地的视频流，对视频每一帧的操作不变。最后，我们还需要将处理好的每一帧的图片再合成为视频。

在这里我们将这个任务分成两步：Step1: 利用关键点识别处理视频的每一帧并保存到本地；Step2: 将本地的视频帧合成为视频。

Step1的代码：

```python
# STEP1: 利用关键点识别处理视频的每一帧并保存到本地
import cv2
from XEdu.hub import Workflow as wf
import os

video_path = "data/eason.mp4" # 指定视频路径
output_dir = 'output/' # 指定保存位置
body = wf(task='pose_body17')# 实例化pose模型
det = wf(task='det_body')# 实例化detect模型
cap = cv2.VideoCapture(video_path)
frame_count = 0 # 视频帧的数量

while True:
    ret, frame = cap.read()
    if not ret:
        print('Video read complete!')
        break
    frame_count += 1
    frame_file_name = f'{output_dir}frame_{frame_count:04d}.jpg' # 每一张帧图片的名称
    bboxs = det.inference(data=frame,thr=0.3)
    img = frame
    for i in bboxs:
        keypoints,img =body.inference(data=img,img_type='cv2',bbox=i)
    for [x1,y1,x2,y2] in bboxs: # 画检测框
        cv2.rectangle(img, (int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
    cv2.imshow('video', img)
    body.save(img,frame_file_name)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    

cap.release()
cv2.destroyAllWindows()
```

Step2的代码：

```python
import cv2
import os

output_video_path = 'output_video.mp4' # 指定合成后视频的名称
output_dir = 'output/' # 指定本地的帧图片的路径
# 获取推理结果文件列表
result_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.jpg')])
# 获取第一张图像的尺寸
first_frame = cv2.imread(result_files[0])
frame_height, frame_width, _ = first_frame.shape
# 设置视频编码器和输出对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码器
out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame_width, frame_height)) # 30 是帧率
print('开始合成视频...')
for image_path in result_files:
    img = cv2.imread(image_path)
    out.write(img)
out.release()
print('视频合成完毕，已保存到：', output_video_path)
```

## 人脸检测控制舵机方向

以下代码可以运行在Arduino开发板上，实现通过跟随人脸位置来控制舵机方向。具体实现方式为：通过人脸检测模型得到人脸检测框的坐标并计算x轴方向的中心点，根据中心点的位置判断是左转还是右转。通过pinpong库控制舵机的转向。

```python
import cv2
from pinpong.board import Board, Pin, Servo
import numpy as np
from XEdu.hub import Workflow as wf
import time


Board('uno').begin() # 指定Arduino开发板，自动识别COM口
det = wf(task='det_face') # 加载人脸检测模型
ser = Servo(Pin(Pin.D4)) # 初始化舵机，指定舵机接口为D4
cap = cv2.VideoCapture(0) # 打开摄像头

while cap.isOpened():
    ret, frame = cap.read()
    x = 300 # 初始化人脸中心点的x坐标
    if not ret:
        break
    result,img = det.inference(data=frame,img_type='cv2',thr=0.3) # 在CPU上进行推理
    if result is not None and len(result) > 0:
        x = int((result[0][2]+result[0][0])/2) # 计算人脸中心点的x坐标
        print(x)
    if x > 400: # 根据人脸中心点的x坐标控制舵机转动,大于400向左转
        time.sleep(0.05)
        ser.write_angle(0)
        print('left')
    elif x < 200: # 根据人脸中心点的x坐标控制舵机转动，小于200向右转
        time.sleep(0.05)
        ser.write_angle(180)
        print('right')
    cv2.imshow('video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # 按q键退出
        break    
cap.release()
cv2.destroyAllWindows()
```
## 识行小车

我们做了一辆能在驾驶过程中自动阅读交通指令并做出相应运动的智能小车，这辆智能小车具有第一人称视角视角系统，可以将数据发送到服务器进行统一计算处理，根据返回结果执行行进指令。

![](../images/xeduhub/car_ocr.gif)

功能描述：一辆能在驾驶过程中自动阅读交通指令并做出相应运动的小车。

技术实现方式：使用无线摄像头实时获取小车行驶的第一视角，并让一台上位机无线接收图像，再对接收的图像使用XEduHub进行文字识别的离线OCR技术，最后根据识别的文本下发运动指令。

该项目中的小车是JTANK履带车，该小车的无线摄像头和运动指令已经经过封装，由于该小车用的不是普通的ESP32的配置，详细配置请参考<a href="https://gitee.com/jt123_456/jtank">JTANK履带车简介</a>。如果你的小车是普通的ESP32，那关于该小车的配置方法可参考<a href="https://xedu.readthedocs.io/zh/master/how_to_use/scitech_tools/camera.html">科创神器ESP32-CAM小型摄像头模块</a>。识行小车使用的文字识别技术采用的是XEduHub现成模型，模型大小只要10M。通过发送HTTP GET请求来获取视频流和控制小车的运动。上位机的参考代码如下。

```python
# 导入库
import time
import requests
import cv2
import numpy as np
from XEdu.hub import Workflow as wf

#定义控制小车运动函数
def control_car(cmd):
    url = "http://192.168.4.1/state?cmd="#URL，需要根据实际设备或服务器的地址进行修改
    response = requests.get(url + cmd)
    time.sleep(0.3)
    requests.get(url='http://192.168.4.1/state?cmd=S') # 停止URL，需要根据实际设备或服务器的地址进行修改
    if response.status_code == 200:
        print("请求成功！")
        print(response.text)
    else:
        print("请求失败，状态码：", response.status_code)

# 定义相机流的URL，需要根据实际设备或服务器的地址进行修改
url = 'http://192.168.4.1:81/stream'
# 发送GET请求来获取视频流
response = requests.get(url, stream=True)
if response.status_code == 200:
    # 使用OpenCV创建一个视频窗口
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    # 实例化模型
    ocr = wf(task='ocr')
    # 设置抽取帧的间隔，例如每25帧抽取一帧
    frame_interval = 25
    frame_count = 0
    max_size=16384 # chunk大小实际情况进行调整 
    # 持续读取视频流，当我们连续获取图像信息就形成了视屏流
    for chunk in response.iter_content(chunk_size=max_size):
        #过滤其他信息，筛选出图像的数据信息
        if len(chunk) >100: 
            if frame_count % frame_interval == 0: # 累计到达相应的帧数对帧进行推理
                # 将数据转换成图像格式
                data = np.frombuffer(chunk, dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                # ocr模型推理
                texts= ocr.inference(data=img) # 进行推理
                # 在窗口中显示图像
                cv2.imshow('Video', img)
                cv2.waitKey(1)
                # 小车控制
                if len(texts)>0:
                    if texts[0][0]=='stop':
                        control_car('S') # 停止
                    elif texts[0][0]=='left':
                        control_car('L') # 左转
                    elif texts[0][0]=='right':
                       control_car('R') # 右转
                    elif texts[0][0]=='go':
                        control_car('F') # 前进
                    elif texts[0][0]=='back':
                        control_car('B') # 后退
            frame_count += 1 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
# 关闭窗口和释放资源
cv2.destroyAllWindows()
response.close()
```

此外我们还做了一辆能在驾驶过程中自动跟踪手掌并做出相应运动的小车。

![](../images/xeduhub/followhandcar.gif)

功能描述：一辆能自动跟随手掌，并跟手掌保持一定距离的小车。

技术实现方式：使用无线摄像头实时获取小车行驶的第一视角，并让一台上位机无线接收图像，再对接收的图像使用XEduHub进行目标检测技术，最后根据识别的手掌关键点距离和手与摄像机之间的距离(cm)的映射，以及手掌水平x的位置，来发送运动指令，实现前进、后退、停止、左转和右转的运动。

该项目中的小车是JTANK履带车，该小车的无线摄像头和运动指令已经经过封装，由于该小车用的不是普通的ESP32的配置，详细配置请参考<a href="https://gitee.com/jt123_456/jtank">JTANK履带车简介</a>。如果你的小车是普通的ESP32，那关于该小车的配置方法可参考<a href="https://xedu.readthedocs.io/zh/master/scitech_tools/camera.html#">科创神器ESP32-CAM小型摄像头模块</a>。识行小车使用的文字识别技术采用的是XEduHub现成模型，模型大小只要10M。通过发送HTTP GET请求来获取视频流和控制小车的运动。上位机的参考代码如下。

```python
import cv2
from XEdu.hub import Workflow as wf
import math
import numpy as np
import requests

# 找到手掌间的距离和实际的手与摄像机之间的距离的映射关系
# x 代表手掌间的距离(像素距离)，y 代表手和摄像机之间的距离(cm)
x = [300, 245, 200, 170, 145, 130, 112, 103, 93, 87, 80, 75, 70, 67, 62, 59, 57]
y = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
# 因此我们需要一个类似 y = AX^2 + BX + C 的方程来拟合
coff = np.polyfit(x, y, 2)  #构造二阶多项式方程，coff中存放的是二阶多项式的系数 A,B,C
A, B, C = coff# 拟合的二次多项式的系数保存在coff数组中，即掌间距离和手与相机间的距离的对应关系的系数

requests.get("http://192.168.4.1/state?cmd=T") #低速档位

#定义控制小车运动函数
def control_car(cmd):
    url = "http://192.168.4.1/state?cmd="#URL，需要根据实际设备或服务器的地址进行修改
    response = requests.get(url + cmd)
    if response.status_code == 200:
        print("请求成功！")
        print(response.text)
    else:
        print("请求失败，状态码：", response.status_code)


# 定义相机流的URL，需要根据实际设备或服务器的地址进行修改
url = 'http://192.168.4.1:81/stream'
# 发送GET请求来获取视频流
response = requests.get(url, stream=True)

if response.status_code == 200:
    # 使用OpenCV创建一个视频窗口
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    # 实例化模型
    detector = wf(task='pose_hand21')
    # 设置抽取帧的间隔，例如每15帧抽取一帧
    frame_interval = 15
    frame_count = 0
    max_size=16384 # chunk大小实际情况进行调整 
    # 持续读取视频流
    for chunk in response.iter_content(chunk_size=max_size):
        #过滤其他信息，筛选出图像的数据信息
        if len(chunk) >100: 
            if frame_count % frame_interval == 0: # 累计到达相应的帧数对帧进行推理
                # 将数据转换成图像格式
                data = np.frombuffer(chunk, dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                # 获取手部关键点信息，绘制关键点和连线后的图像img
                keypoints, img_with_keypoints = detector.inference(data=img, img_type='cv2') # 进行推理
                # 如果检测到手
                if len(keypoints):
                    # 获取食指根部'5'和小指根部'17'的坐标点
                    x1, y1 = keypoints[5] 
                    x2, y2 = keypoints[17]
                    # 勾股定理计算关键点'5'和'17'之间的距离，并变成整型
                    distance = int(math.sqrt((x2-x1)**2 + (y2-y1)**2))
                    # 得到像素距离转为实际cm距离的公式 y = Ax^2 + Bx + C
                    distanceCM = A*distance**2 + B*distance + C
                    # 勾股定理计算关键点'5'和'17'之间的距离，并变成整型
                    distance = int(math.sqrt((x2-x1)**2 + (y2-y1)**2))
                    mid=(x1+x2)/2
                else:
                    distanceCM=80
                    mid=150
                
                print(distanceCM,mid)
                
                # 小车控制
                if  90 > distanceCM > 70 and 270>mid>50:
                        control_car('S') # 停止
                        print('stop')
                elif distanceCM>=90 and 270>mid>50:
                        control_car('F') # 前进
                        print('go')
                elif distanceCM<=70 and 270>mid>50:
                        control_car('B') # 后退
                        print('back')
                elif 270<mid:
                        control_car('R') # 右转
                        print('right')
                elif 50>mid:
                        control_car('L') # 左转
                        print('left')
                # 在窗口中显示图像
                cv2.imshow('Video', img_with_keypoints)

            frame_count += 1 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
# 释放视频资源
cv2.destroyAllWindows()
response.close()
```