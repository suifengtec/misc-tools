# -*- coding: utf-8 -*-
# @Author: suifengtec
# @Date:   2018-06-30 13:20:43
# @Last Modified by:    suifengtec
# @Last Modified time: 2018-09-30 22:44:41
#
#
#
# 识别视频中的人脸.
# 仅为测试,未作充分容错和优化。
#
#
# 使用示例(friv=Face Recognition In Video)
# python friv.py
# python friv.py Hawking.mp4
# python friv.py sp.mp4
#
# 在 OpenCV窗口上,按q键退出运行。
#
# ========扩展
#
# OpenBR:全称是Open Source Biometric Recognition,侧重于生物识别, 年龄,性别探测,面部比较,
# 面部识别准确率高于 COTS;
#
#
# OpenFace: 基于深度学习对人脸进行识别,简单,准确率高,直接碾压 OpenBR等一众选手,下面是相关介绍:
# https://zhuanlan.zhihu.com/p/24567586
#
#
import os
import sys
import face_recognition
import cv2
import numpy as np
# 用于和openCV配合使用,给视频中的人标记姓名
from PIL import Image, ImageDraw, ImageFont

# 下面的本地包引入,弃用了
# try:
#     from . import cv2cntxt    # "myapp" case
# except:
#     import cv2cntxt            # "__main__" case
# ================助手方法开始


def isDirExists(dirName):
    if isPathExists(dirName) == True:
        if isFileExists(dirName) == False:
            return True
    return False


def isPathExists(somePath):
    return os.path.exists(somePath)


def isFileExists(filePath):
    return os.path.isfile(filePath)
# ==============助手方法结束


def getRealPath(cwd, what):
    return os.path.join(cwd, what)


def getSubDirPath(cwd, what):
    subPath = os.path.join(cwd, what)
    if os.path.exists(subPath) == False:
        os.mkdir(subPath)
    return subPath


def getInputVideoName(args):
    name = ""
    if len(args) >= 1:
        name = args[0]
    else:
        name = "default.mp4"
    return name


def getInputVideoPath(cwd, fName):
    return os.path.join(cwd, "data-input", personName)


def getInputKnownFace(cwd, personName):
    return os.path.join(cwd, "data-input", "known-faces", personName)


def getOutputDir(dirPath):
    if isPathExists(dirPath) != True:
        os.mkdir(dirPath)
    return dirPath


def getOutPutImgPath(outputPath, pId, count):
    return os.path.join(getOutputDir(outputPath), pId+str(count)+".png")


def main(args):

    cwd = os.getcwd()

    # 准备输入数据
    # 输入视频,应故意使用不清晰的低分辨率视频

    videoInputName = getInputVideoName(args)

    dataInput = {

        "mode": "1",
        "inputPath": "data-input",
        "inputVideoName": videoInputName,
        "outputPath": "data-output",
        "useFont":  os.path.join(cwd, "fonts", "wqy-zenhei.ttc"),
        "knownPersons": [
                {
                    "id": "sheldon",
                    "label": "谢尔顿",
                    "featureImgs": [
                        getInputKnownFace(cwd, "sheldon.jpg")
                    ],
                    "count":0
                },
            {
                    "id": "hawking",
                    "label": "霍金",
                    "featureImgs": [
                        getInputKnownFace(cwd, "hawking.png")
                    ],
                    "count":0
                },
            {
                    "id": "penny",
                    "label": "佩妮",
                    "featureImgs": [
                        getInputKnownFace(cwd, "penny.jpg")
                    ],
                    "count":0
                }
        ]
    }

    # 尝试载入输入视频
    try:
        videoInput = cv2.VideoCapture(os.path.join(
            cwd, dataInput["inputPath"], dataInput["inputVideoName"]))
        if not videoInput.isOpened():
            raise NameError('输入视频的文件路径有问题?')
    except cv2.error as e:
        print("cv2.error:", e)
    except Exception as e:
        print("Exception:", e)
    else:
        print("已开始解析视频...")

    # 除了FPS外.输出视频的参数和输入视频的参数一致
    # https://docs.opencv.org/3.4.3/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
    frameLength = int(videoInput.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(videoInput.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(videoInput.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameFPS = int(videoInput.get(cv2.CAP_PROP_FPS))

    # 输出视频以
    # 30fps,尺寸故意应不大于输入视频
    #  (1280, 720)
    #  (640, 360)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoOutput = cv2.VideoWriter(os.path.join(
        cwd, dataInput["outputPath"], videoInputName + '_output.avi'),
        fourcc, frameFPS, (frameWidth, frameHeight))

    # 已知的人脸,未作充分优化和容错
    knownFaces = []

    for p in dataInput['knownPersons']:
        imgKnownFace = face_recognition.load_image_file(
            getInputKnownFace(cwd, p['featureImgs'][0]))
        encodingFace = face_recognition.face_encodings(imgKnownFace)[0]
        knownFaces.append(encodingFace)

    # 变量初始化
    face_locations = []
    face_encodings = []
    face_names = []
    faceLabel = ""
    # 已解析的帧的数量
    frameCount = 0
    # 开始处理输入视频的每一帧
    while True:
            # 单帧抓取
        ret, frame = videoInput.read()
        frameCount += 1
        # 输入视频结束后,结束循环
        if not ret:
            break

        # 查找当前视频帧中的所有人脸以及人脸编码
        # Python face_recognition 采用的是 dlib 库,这个库解析人脸上的68个点。
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # 查看当前帧中的人脸是否与已知的人脸相符
            match = face_recognition.compare_faces(
                knownFaces, face_encoding, tolerance=0.50)

            # 如果你有两个以上的脸，你可以让这个逻辑更漂亮
            # 但在演示中我保持了简单
            matchIndex = None
            if match[0]:
                matchIndex = dataInput['knownPersons'][0]['id']
            elif match[1]:
                matchIndex = dataInput['knownPersons'][1]['id']
            elif match[2]:
                matchIndex = dataInput['knownPersons'][2]['id']
            face_names.append(matchIndex)

        # 对结果进行标记
        for (top, right, bottom, left), matchIndex in \
                zip(face_locations, face_names):
            if not matchIndex:
                continue

            # 在视频帧中发现的人脸周围画一个方框
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
            # 从视频帧中裁切保存人脸到相应的路径
            outputImgFilePath = ""
            crop_img = frame[top:bottom, left:right]

            if(matchIndex == dataInput['knownPersons'][0]['id']):
                faceLabel = dataInput['knownPersons'][0]['label']
                outputImgFilePath = getOutPutImgPath(os.path.join(
                    cwd, dataInput['outputPath'], matchIndex),
                    dataInput['knownPersons'][0]['id'],
                    dataInput['knownPersons'][0]['count'])
                cv2.imwrite(outputImgFilePath, crop_img)
                dataInput['knownPersons'][0]['count'] = \
                    dataInput['knownPersons'][0]['count'] + 1
            elif(matchIndex == dataInput['knownPersons'][1]['id']):
                faceLabel = dataInput['knownPersons'][1]['label']
                outputImgFilePath = getOutPutImgPath(os.path.join(
                    cwd, dataInput['outputPath'], matchIndex),
                    dataInput['knownPersons'][1]['id'],
                    dataInput['knownPersons'][1]['count'])
                cv2.imwrite(outputImgFilePath, crop_img)
                dataInput['knownPersons'][1]['count'] = \
                    dataInput['knownPersons'][1]['count'] + 1
            elif(matchIndex == dataInput['knownPersons'][2]['id']):
                faceLabel = dataInput['knownPersons'][2]['label']
                outputImgFilePath = getOutPutImgPath(os.path.join(
                    cwd, dataInput['outputPath'], matchIndex),
                    dataInput['knownPersons'][2]['id'],
                    dataInput['knownPersons'][2]['count'])

                cv2.imwrite(outputImgFilePath, crop_img)

                dataInput['knownPersons'][2]['count'] = \
                    dataInput['knownPersons'][2]['count'] + 1

            # 在视频帧中发现的人脸下方,标记人物姓名
            cv2.rectangle(frame, (left, bottom - 18),
                          (right, bottom), (0, 0, 255), cv2.FILLED)
            # 为了输出中文,改用Pillow的方法
            # font = cv2.FONT_HERSHEY_DUPLEX
            # font = cv2.FONT_HERSHEY_COMPLEX

            # cv2.putText(frame, faceLabel, (left + 6, bottom - 6),
            #            font, 1.0, (255, 255, 255), 1)
            #
            fontpath = os.path.join(cwd, dataInput["useFont"])
            font = ImageFont.truetype(fontpath, 16)
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            draw.text((left + 4, bottom - 18), faceLabel,
                      font=font, fill=(0, 255, 0, 0))
            frame = np.array(img_pil)
        # 被帧写出到输出视频中
        videoOutput.write(frame)
        print("写出帧 {0} / {1}".format(frameCount, frameLength))
        cv2.imshow('coolwp: Video Face Recognition', frame)
        # Q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放输入视频,销毁openCV打开的窗口,结束程序
    videoInput.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv[1:])
