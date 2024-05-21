# **嵌入式人工智能实践项目——天气预测**

## 〇. 简介
众所周知，沆州天气变换莫测，天气预报似乎无能为力。为了成为杭州的天气之子，掌握实时天气动向，快快拿起手头的单片机，结合紫金港气象观测站的数据与你的智能，相信你也能成为 ~~浙大阳菜~~。


## 一. 实验目的

1. 设计并实现嵌入式人工智能天气预测系统，根据温湿度信息对当前晴雨天气进行判断，并通过改变LED灯的状态进行输出
2. 通过对硬件系统的实现，了解并体验STM32单片机编程的基本方法，加深对嵌入式编程的理解
3. 通过对模型的设计与训练，巩固之前学习的人工智能知识，并将其运用于实战中

## 二. 实验材料

+ stm32f103c8t6最小系统板，ST-Link下载器
+ USB-串口转换模块
+ DHT11温湿度传感器
+ LED灯珠
+ 面包板
+ 杜邦线若干

上述材料均可在淘宝买到，总金额50元左右，每组仅需保证至少有一套材料即可。如在购买过程中遇到困难，请钉钉联系。


!!! info
    感兴趣的同学可购置更多模块以实现一个更为完整的系统，推荐购买江科大套件（上述材料均含在套件中），后续可配合[相关课程](https://www.bilibili.com/video/BV1th411z7sn/)进一步学习。

## 三. 实验步骤

### 硬件部分

#### 0 环境配置

目前主流的单片机开发平台有Keil、IAR、STM32CubeIDE等，本实验出于便捷性考虑，采用VScode+PlatformIO的形式进行开发，配置方法如下：

1. 安装PlatformIO扩展：

    ![](https://s2.loli.net/2024/05/21/DBkuLndTjx2IPAY.png)

2. 下载[项目模板代码](https://pan.zju.edu.cn/share/ec47b5000a12d1ecc5c81d8305)
3. 打开示例项目文件夹，打开platformio.ini，等待必要组件下载完成

一个platformIO工程的项目结构如下：
```
├─.pio
│  └─build
├─.vscode
├─include
├─lib
├─src
│  ├─CORE
│  ├─FWLIB
│  │  ├─inc
│  │  └─src
│  ├─SYSTEM
│  └─main.c
├─test
├─.gitignore
└─platformio.ini
```
下对实验中涉及的文件/文件夹加以简单说明：

+ .pio/build：程序编译结果
+ .vscode：VScode工作区配置（对C/C++插件的配置，用于头文件寻找等）
+ include：stm32全局宏定义
+ src：用于存放用户代码
    + CORE：Cortex-M3芯片外设宏定义
    + FWLIB：stm32固件库，存放stm32片上外设驱动
    + SYSTEM：用于存放其它外设驱动
+ main.c：程序入口
+ platformio.ini：项目配置文件

故在main.c中书写处理的主要逻辑，==将其它驱动放置在SYSTEM目录下==。若想在项目中创建其它文件夹，请同步修改platformio.ini，将其添加到build_flags中：`-Idir_name`.

main.c的结构如下：
```c
#include "stm32f10x.h"

int main(void) {
  // setup
  while (1) {
    // loop
  }
}
```
单片机在进入程序后先进行初始化设定，设定完毕后进入事件循环。

!!! info
    由于platformIO的启动代码中并没有对全局时钟进行配置，进入main函数后需手动配置全局时钟，调用`RCC_Configuration()`函数对全局时钟进行配置。否则内置时钟不能正常工作，单片机将使用外部时钟，导致定时器走慢9倍。

    模板中已经在main.c中对该函数进行了调用，无需做额外修改。

按照以下接线图，使用ST-Link将单片机连接电脑

![](https://s2.loli.net/2024/05/21/konJQaUKychNL61.jpg)

使用底部功能按键完成代码编译与烧写

![](https://s2.loli.net/2024/05/21/nMxZdcbzf27NtmO.png)

烧写成功，完成配置。


#### 1 系统搭建

接线图如图所示

![](https://s2.loli.net/2024/05/21/TbCmoy2LafxtDkq.png)

实物图如下，仅供参考

![](https://s2.loli.net/2024/05/21/lW6Tv4ji8xm2fI7.png)

#### 2 成为点灯大师

下面正式开始单片机代码编写。复制模板以创建新项目，或直接在模板上进行修改。

第一个任务为点亮A0脚上的LED灯。由于LED长脚接电源正极，故短脚输出低电平时电路接通，LED灯点亮。

请尝试根据以上实验原理，点亮该LED灯。具体步骤为

1. 使能APB2外设GPIOA时钟
2. 通过GPIO_Init结构体初始化对应引脚
3. 向该引脚写入低电平

参考函数：
```c
// APB2外设时钟控制
void RCC_APB2PeriphClockCmd(uint32_t RCC_APB2Periph, FunctionalState NewState);

// 根据给定结构体初始化
void GPIO_Init(GPIO_TypeDef* GPIOx, GPIO_InitTypeDef* GPIO_InitStruct);

// 向指定端口写入值
void GPIO_WriteBit(GPIO_TypeDef* GPIOx, uint16_t GPIO_Pin, BitAction BitVal);
```

进一步的，通过调用给定的延时函数，可实现LED灯闪烁的效果，以下是一些使用例
```c
#include "Delay.h"

Delay_us(100); // 延时100us
Delay_ms(500); // 延时500ms
Delay_s(2); // 延时2s
```

参考代码：
```c
// 使能APB2外设时钟
RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);

// 构造GPIO_Init结构体初始化引脚
GPIO_InitTypeDef GPIO_InitStructure;
// 设置输出模式、引脚与速度
GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
GPIO_InitStructure.GPIO_Pin = GPIO_Pin_0;
GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
GPIO_Init(GPIOA, &GPIO_InitStructure);

// LED灯闪烁
while (1) {
  GPIO_WriteBit(GPIOA, GPIO_Pin_0, Bit_RESET);
  Delay_ms(500);
  GPIO_WriteBit(GPIOA, GPIO_Pin_0, Bit_SET);
  Delay_ms(500);  
}
```

#### 3 串口通信

!!! Important
    实验中涉及的[库代码下载](https://pan.zju.edu.cn/share/1da9195158778e748f4c6a563a)

由于单片机上运行的程序不像直接运行在PC上的程序，可以简单的通过`printf`将结果输出到stdout上可直接在控制台观察，我们需要通过串口将数据从单片机上发回PC，从而通过串口监视器观察输出结果。

先下载所需的库文件，并参照上述项目结构说明把它们放到指定的目录下。先对串口进行初始化，然后再调用相应的输出函数，具体请阅读Serial.h.

使用示例：
```c
// 初始化串口
Serial_Init();

// 发送字符串
Serial_SendString("Hello World!");
```

打开串口监视器，并烧写程序，观察实验现象。

!!! Warning
    如果你没有按照接线图要求对usb转串口模块进行接线，请自行修改Serial.c，使之与实际的接线保持一致。

为了让串口能够发送浮点数，需要修改编译指令。将下列编译指令加到platformio.ini文件的build_flags中：

```
-Wl,-u_printf_float
```
这样便可利用串口发送浮点数。

#### 4 获取温湿度信息

将DHT11模块驱动代码放到指定的位置。

调用库函数获取温湿度，示例如下：
```c
#include "dht11.h"

uint16_t _temperature, _humidity;
double temperature, humidity;

RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB, ENABLE); // 不要忘记使能DHT所在GPIO口的时钟 !!!
DHT11_Init();

DHT11_Read_Data(&_temperature, &_humidity);
temperature = (_temperature >> 8) + (_temperature & 0xff) * 0.1; // 单位℃
humidity = (_humidity >> 8) + (_humidity & 0xff) * 0.1; // 单位%
```

!!! Warning
    若没有按照接线图将DHT11的Data脚接到B13处，则需修改dht11.h与dht11.c保证接线与代码的一致性。

学有余力的同学可尝试阅读dht驱动代码，尝试理解dht通信时序。

### 软件部分

本部分请大家结合前面课程所学，自由发挥，设计模型并**在PC端训练**，并尝试将训练好的**模型推理**部署到单片机上，结合上述硬件与外设操作完成实验任务。

#### 1 数据处理

紫金港观测站观测数据[下载](https://pan.zju.edu.cn/share/7680a0dc051b27433ec4e7ba05)

!!! info
    鼓励尝试使用爬虫对网络数据进行爬取

主要利用到数据表中的列Ta_Avg(平均温度), Ta_Max(最大温度), Ta_Min(最低温度), RH_Avg(平均湿度), RH_Max(最大湿度), RH_Min(最低湿度), rain_Tot(降水量).

可以使用pandas库对数据表进行读取，下面给出一个示例
```python
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
# df.head() # 查看数据表前5行数据

temps = np.array(df["Ta_Avg"][2:], dtype=float)
humids = np.array(df["RH_Avg"][2:], dtype=float)
is_rain = np.array(df["rain_Tot"][2:], dtype=float) > 0
```

更多使用方法请参考[官方文档](https://pandas.pydata.org/).

#### 2 模型训练与部署

由于stm32f10x系列芯片存储、运算能力受限，模型训练应选择在PC端完成。可以利用pytorch在PC端完成数据的训练，得到模型参数，再在stm32上实现该模型，并填入模型参数，实现推理函数，并利用推理结果实现对LED灯的控制.

也可以尝试[TensorFlow Lite](https://www.tensorflow.org/lite/microcontrollers?hl=zh-cn)，具体内容请参考文档。

下面给出一个简单的示例，效果并不是很好，仅供参考，请勿照搬。

使用单层线性神经网络 + sigmoid激活函数，采用交叉熵损失函数，并使用随机梯度下降算法
```python
import torch
from torch import nn

model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
loss_fn = nn.BCELoss()
opt = torch.optim.SGD(model.parameters(), lr = 0.0001)
```

划分数据集与训练集
```python
from sklearn.model_selection import train_test_split

X = np.concatenate((temps[:,np.newaxis], humids[:,np.newaxis]), axis=1)
Y = is_rain[:np.newaxis]

train_x, test_x, train_y, test_y = train_test_split(X, Y)
train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
train_y = torch.from_numpy(train_y).type(torch.FloatTensor)
test_y = torch.from_numpy(test_y).type(torch.FloatTensor)
train_ds = TensorDataset(train_x, train_y)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
valid_ds = TensorDataset(test_x, test_y)
valid_dl = DataLoader(valid_ds, batch_size=128)
```

模型训练
```python
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb).reshape(-1), yb.reshape(-1))
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
 
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
 
        print(epoch, val_loss)

fit(100, model, loss_fn, opt, train_dl, valid_dl)
```

在单片机上实现该模型
```c
double sigmoid(double x) { return 1.0 / (1 + exp(-x)); }

uint8_t predict(double temperature, double humidity) {
  double A = ..., B = ..., C = ...;
  return sigmoid(A * temperature + B * humidity + C) > 0.5;
}
```

运用该模型进行推理，并反馈在LED灯上
```c
GPIO_WriteBit(GPIOA, GPIO_Pin_0, predict(temperature, humidity) ? Bit_RESET : Bit_SET);
```

## 四 评分标准

本项目涉及软硬件协同，具有一定的挑战性，推荐1-4人一组，请在5.25日23:59分前完成分组与组长登记。采取**小组展示**的形式，无需报告或验收，评分标准如下：

+ 系统完整性 80%：按要求实现完整系统
+ 展示与互评 20%：各组组长对其它小组的展示进行打分，从硬件系统的完整性、创新性与数据处理的合理性、正确性等方面进行评估
+ Bonus：
    1. 使用额外的模块构建系统可获得加分(20%)
    2. 解释dht驱动实现原理可获得加分(10%)
    3. 使用TensorFlow Lite部署模型(30%)
    4. 使用更加复杂的模型进行推理（如考虑数据时序、构造特征等）(25%)