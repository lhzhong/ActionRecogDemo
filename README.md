# Demo
## 效果

![image](https://github.com/lhzhong/ActionRecogDemo/blob/master/demo.gif)

## 运行环境及安装需求
我的程序是在Ubuntu16.04和装有Tian1080Ti的环境中运行。要运行该demo，需要安装
* python3
* Tensorflow 1.3+
* opencv
* ffmpeg

## 功能
该demo实现的是视频行为的简单识别，视频采集途径包括实施摄像头采集和从本低选取已有视频。

## 细节
1. 界面设计是基于pyqt5的
2. 测试时模型是调取已训练得到的模型，模型训练基于UCF101数据库（训练基于视频单帧的CNN训练，后续需要模型改进）
3. 目前测试是对单帧视频帧进行测试（后续需要进一步的算法完善，实现小段视频的识别识别率更加可靠）
4. 视频读取（摄像头、文件读取）和识别是放在两个线程处理的，视频读取放在主线程，而识别是通过pyqt中的Thread开的新线程，数据处理通过`pyqtSignal`函数进行线程间数据交互
