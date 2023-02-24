# 开源硬件库pinpong

## 1. 简介

pinpong库是一个基于Firmata协议开发的Python硬件控制库。借助于pinpong库，直接用Python代码就能给各种常见的开源硬件编程，即使该硬件并不支持Python。

pinpong库的原理是给开源硬件烧录一个特定的固件，使其可以通过串口与电脑通讯，执行各种命令。pinpong库的名称由“Pin”和“Pong”组成，“Pin”指引脚，“pinpong”为“乒乓球”的谐音，指信号的往复。目前pinpong库支持Arduino、掌控板、micro:bit等开源硬件，同时支持虚谷号、树莓派和拿铁熊猫等。

github地址：https://github.com/DFRobot/pinpong-docs

## 2. 安装

可以使用使用pip命令安装pinpong库。

```
pip install pinpong
```

注：XEdu一键安装包中已经内置了pinpong库。

## 3. 代码示例

示例程序以“Arduino”为例，复制粘贴代码到python编辑器中，并修改Board初始化版型参数为你使用的板子型号即可。

### 3.1 数字输出

实验效果：控制arduino UNO板载LED灯一秒闪烁一次。

接线：使用windows或linux电脑连接一块arduino主控板。

```python
import time
from pinpong.board import Board,Pin

Board("uno").begin()               #初始化，选择板型(uno、microbit、RPi、handpy)和端口号，不输入端口号则进行自动识别
#Board("uno","COM36").begin()      #windows下指定端口初始化
#Board("uno","/dev/ttyACM0").begin() #linux下指定端口初始化
#Board("uno","/dev/cu.usbmodem14101").begin()   #mac下指定端口初始化

led = Pin(Pin.D13, Pin.OUT) #引脚初始化为电平输出

while True:
  #led.value(1) #输出高电平 方法1
  led.write_digital(1) #输出高电平 方法2
  print("1") #终端打印信息
  time.sleep(1) #等待1秒 保持状态
  #led.value(0) #输出低电平 方法1
  led.write_digital(0) #输出低电平 方法2
  print("0") #终端打印信息
  time.sleep(1) #等待1秒 保持状态
```

### 3.2 数字输入

实验效果：使用按钮控制arduino UNO板载亮灭。

接线：使用windows或linux电脑连接一块arduino主控板，主控板D8接一个按钮模块。

```python
import time
from pinpong.board import Board,Pin

Board("uno").begin() 

btn = Pin(Pin.D8, Pin.IN) #引脚初始化为电平输入
led = Pin(Pin.D13, Pin.OUT)

while True:
  #v = btn.value()  #读取引脚电平方法1
  v = btn.read_digital()  #读取引脚电平方法2
  print(v)  #终端打印读取的电平状态
  #led.value(v)  #将按钮状态设置给led灯引脚  输出电平方法1
  led.write_digital(v) #将按钮状态设置给led灯引脚  输出电平方法2
  time.sleep(0.1)
```

### 3.3 模拟输入

实验效果：打印UNO板A0口模拟值。

接线：使用windows或linux电脑连接一块arduino主控板，主控板A0接一个旋钮模块。

```python
import time
from pinpong.board import Board,Pin

Board("uno").begin()

#adc0 = ADC(Pin(Pin.A0)) #将Pin传入ADC中实现模拟输入  模拟输入方法1
adc0 = Pin(Pin.A0, Pin.ANALOG) #引脚初始化为电平输出 模拟输入方法2

while True:
  #v = adc0.read()  #读取A0口模拟信号数值 模拟输入方法1
  v = adc0.read_analog() #读取A0口模拟信号数值 模拟输入方法2
  print("A0=", v)
  time.sleep(0.5)
```

### 3.4 模拟输出

实验效果： PWM输出实验,控制LED灯亮度变化。

接线：使用windows或linux电脑连接一块arduino主板，LED灯接到D6引脚上。

```python
import time
from pinpong.board import Board,Pin

Board("uno").begin()

#pwm0 = PWM(Pin(board,Pin.D6)) #将引脚传入PWM初始化  模拟输出方法1
pwm0 = Pin(Pin.D6, Pin.PWM) #初始化引脚为PWM模式 模拟输出方法2

while True:
    for i in range(255):
        print(i)
        #pwm0.duty(i) #PWM输出 方法1
        pwm0.write_analog(i) #PWM输出 方法2
        time.sleep(0.05)
```

### 3.5 引脚中断

实验效果：引脚模拟中断功能测试。

接线：使用windows或linux电脑连接一块arduino主控板，主控板D8接一个按钮模块。

```python
import time
from pinpong.board import Board,Pin

Board("uno").begin()

btn = Pin(Pin.D8, Pin.IN)

def btn_rising_handler(pin):#中断事件回调函数
  print("\n--rising---")
  print("pin = ", pin)

def btn_falling_handler(pin):#中断事件回调函数
  print("\n--falling---")
  print("pin = ", pin)

def btn_both_handler(pin):#中断事件回调函数
  print("\n--both---")
  print("pin = ", pin)

btn.irq(trigger=Pin.IRQ_FALLING, handler=btn_falling_handler) #设置中断模式为下降沿触发
#btn.irq(trigger=Pin.IRQ_RISING, handler=btn_rising_handler) #设置中断模式为上升沿触发，及回调函数
#btn.irq(trigger=Pin.IRQ_RISING+Pin.IRQ_FALLING, handler=btn_both_handler) #设置中断模式为电平变化时触发

while True:
  time.sleep(1) #保持程序持续运行
```



更多代码请访问官方文档。

官方文档地址：https://pinpong.readthedocs.io/

## 4. 借助pinpong开发智能作品

开源硬件是创客的神器，而pinpong进一步降低了开源硬件的编程门槛。pinpong库的设计，是为了让开发者在开发过程中不用被繁杂的硬件型号束缚，而将重点转移到软件的实现。哪怕程序编写初期用Arduino开发，部署时改成了掌控板，只要修改一下硬件的参数就能正常运行，实现了“一次编写处处运行”。

当学生训练出一个AI模型，就可以通过各种硬件设备进行多模态交互。当学生训练出一个简单猫狗分类模型后，加上一个舵机，就能实现智能宠物“门禁”；加上一个马达，就能做出一个智能宠物驱逐器；加上一条快门线，就能做宠物自动拍照设备。有多少创意，就能实现多少与众不同的作品。