# MQTT库siot

## 1. 简介

MQTT是最常用的物联网协议之一。但是，MQTT的官方Python库明显不好用，前面要定义一个类，代码冗长，对初学者不够友好。siot是虚谷物联团队基于MQTT paho写的一个Python库，为了让初学者能够写出更加简洁、优雅的Python代码。 

需要强调的是，siot库同时支持MicroPython，语法完全一致。 

GitHub地址：[https://github.com/vvlink/SIoT/tree/master/Python_lib/siot](https://github.com/vvlink/SIoT/tree/master/Python_lib/siot)

SIoT文档地址：[https://siot.readthedocs.io/zh_CN/latest/](https://siot.readthedocs.io/zh_CN/latest/)

SIoT V2 新增了可视化面板，全新升级，性能提升，可以支持更快的速度，同时使用QOS区分了快速数据以及存入数据的数据以应对不同的使用场景，网页界面也进行了更新更美观。具体可以参见：[https://mindplus.dfrobot.com.cn/dashboard](https://mindplus.dfrobot.com.cn/dashboard)

本文涉及的部分代码见XEdu帮助文档配套项目集：[https://www.openinnolab.org.cn/pjlab/project?id=64f54348e71e656a521b0cb5&sc=645caab8a8efa334b3f0eb24#public](https://www.openinnolab.org.cn/pjlab/project?id=64f54348e71e656a521b0cb5&sc=645caab8a8efa334b3f0eb24#public)

## 2. 安装

可以使用使用pip命令安装siot库。

```
pip install siot
```

注：XEdu一键安装包中已经内置了siot库。

## 3. 代码范例

下面的代码以MQTT服务器软件SIoT为例。SIoT是一个一键部署的MQTT服务器，广泛应用于中小学的物联网教学中。

### 3.1 发送消息



```plain
import siot
import time

SERVER = "127.0.0.1"            #MQTT服务器IP
CLIENT_ID = ""                  #在SIoT上，CLIENT_ID可以留空
IOT_pubTopic  = 'xzr/001'       #“topic”为“项目名称/设备名称”
IOT_UserName ='siot'            #用户名
IOT_PassWord ='dfrobot'         #密码

siot.init(CLIENT_ID, SERVER, user=IOT_UserName, password=IOT_PassWord)
siot.connect()
siot.loop()

tick = 0
while True:
  siot.publish(IOT_pubTopic, "value %d"%tick)
  time.sleep(1)           #隔1秒发送一次
  tick = tick+1
```



### 3.2 订阅消息



```plain
import siot
import time

SERVER = "127.0.0.1"        #MQTT服务器IP
CLIENT_ID = ""              #在SIoT上，CLIENT_ID可以留空
IOT_pubTopic  = 'xzr/001'   #“topic”为“项目名称/设备名称”
IOT_UserName ='siot'        #用户名
IOT_PassWord ='dfrobot'     #密码

def sub_cb(client, userdata, msg):
  print("\nTopic:" + str(msg.topic) + " Message:" + str(msg.payload))
  # msg.payload中是消息的内容，类型是bytes，需要用解码。
  s=msg.payload.decode()
  print(s)

siot.init(CLIENT_ID, SERVER, user=IOT_UserName, password=IOT_PassWord)
siot.connect()
siot.subscribe(IOT_pubTopic, sub_cb)
siot.loop()
```



### 3.3 订阅多条消息



```plain
import siot
import time

SERVER = "127.0.0.1"         # MQTT服务器IP
CLIENT_ID = ""               # 在SIoT上，CLIENT_ID可以留空
IOT_pubTopic1  = 'xzr/001'   # “topic”为“项目名称/设备名称”
IOT_pubTopic2  = 'xzr/002'   # “topic”为“项目名称/设备名称”
IOT_UserName ='siot'         # 用户名
IOT_PassWord ='dfrobot'      # 密码

def sub_cb(client, userdata, msg):  # sub_cb函数仍然只有一个，需要在参数msg.topic中对消息加以区分
  print("\nTopic:" + str(msg.topic) + " Message:" + str(msg.payload))
  # msg.payload中是消息的内容，类型是bytes，需要用解码。
  s=msg.payload.decode()
  print(s)

siot.init(CLIENT_ID, SERVER, user=IOT_UserName, password=IOT_PassWord)
siot.connect()
siot.set_callback(sub_cb)         
siot.getsubscribe(IOT_pubTopic1)  # 订阅消息1
siot.getsubscribe(IOT_pubTopic2)  # 订阅消息2
siot.loop()
```



## 4. 借助siot部署智联网应用

当物联网遇上人工智能，就形成了智联网。当学生训练出一个AI模型，就可以通过物联网设备进行多模态交互。

1）远程感知

[mind+远程图传案例](https://mindplus.dfrobot.com.cn/dashboard#target_16)


2）远程控制

[mind+控制LED灯案例](https://mindplus.dfrobot.com.cn/dashboard#target_14)

3）可视化展示

[mind+可视化面板案例](https://mindplus.dfrobot.com.cn/dashboard#target_15)