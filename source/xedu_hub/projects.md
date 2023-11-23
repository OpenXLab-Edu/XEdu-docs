# XEduHub项目案例集

借助XEduHub可以实现应用多元AI模型去解决复杂的问题。

## 实时人体关键点识别

以下代码可以实时检测摄像头中出现的多个人，并对每一个人体提取关键点。

具体实现方式为：我们首先将实时视频中每一帧的图像进行人体目标检测，拿到所有的检测框`bbox`及其坐标信息，绘制检测框。随后对每个检测框中的人体进行关键点提取。

```python
from XEdu.hub import Workflow as wf
import cv2
cap = cv2.VideoCapture(0)
body = wf(task='body17')# 实例化pose模型
det = wf(task='bodydetect')# 实例化detect模型
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
body = wf(task='body17')# 实例化pose模型
det = wf(task='bodydetect')# 实例化detect模型
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

## 多人脸关键点识别

以下代码可以将一张图片中所有的人脸识别出来，并对每一张脸提取关键点。这可以用于对一张图片中的所有人进行表情分类，推测情感等。

具体实现方式为：我们首先使用`facedetect`进行人脸检测，拿到所有的检测框`bbox`。随后对每个检测框中的人脸进行关键点提取。

```python
from XEdu.hub import Workflow as wf
face_det = wf(task='facedetect')
face_kp = wf(task='face')
bboxs,img = face_det.inference(data='face.jpg',img_type='cv2')
for i in bboxs:
    keypoints,img = face_kp.inference(data=img,img_type='cv2',bbox=i)
    face_kp.show(img)
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
det = wf(task='facedetect') # 加载人脸检测模型
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

以下代码可以运行在esp32cam核心板上，该芯片是一款高性能、低功耗的芯片，并且具有集成摄像头功能。使用文字识别的离线OCR技术，用XEduHub现成模型，模型大小只要10M。实现阅读交通信息来控制小车运动。具体实现方式为：使用 OCR 技术来识别图像中的文本，并根据识别的文本控制小车的运动。通过发送HTTP GET请求来获取视频流和控制小车的运动。

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
    # 持续读取视频流
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
                    if texts[0][0]=="stop":
                        control_car('S') # 停止
                    elif texts[0][0]=="left":
                        control_car('L') # 左转
                    elif texts[0][0]=="right":
                       control_car('R') # 右转
                    elif texts[0][0]=="go":
                        control_car('F') # 前进
                    elif texts[0][0]=="back":
                        control_car('B') # 后退
            frame_count += 1 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
# 关闭窗口和释放资源
cv2.destroyAllWindows()
response.close()
```