# 识别视频中的人脸.

仅做测试,测试环境为 Windows 10 + Python 3.6.X,未做充分的优化和容错。

基于 Python 项目 face_recognition , 该项目依赖:

```
pip install numpy
pip install scipy
pip install scikit-image
pip install dlib
pip install face_recognition
```

* dlib: 识别准确率一般偏上;
* OpenBR:全称是Open Source Biometric Recognition,侧重于生物识别,年龄,性别探测,面部比较,识别准确率一般偏上;
* OpenFace:基于深度学习对人脸进行识别,简单,准确率高,直接碾压 OpenBR等一众选手。


## 截图

![./screenshot/1.gif](./screenshot/1.gif)

## 测试与使用

```

git clone 

cd face_recognition-in-video-v1

python friv.py


```
friv=Face Recognition In Video.

在 OpenCV窗口上,按q键退出运行。





