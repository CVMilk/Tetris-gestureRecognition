# -*- coding: utf-8 -*-
"""
Created on 2021-07-18 

@File : demo.py

@Author : Liulei (mrliu_9936@163.com)

@Purpose :  测试手势识别模型

"""
import torch
from trainNet import get_net
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import cv2
import numpy as np

bg = None

def run_avg(image, aWeight):
    """  求前 n 帧图像平均像素值，即获得背景图像"""
    global bg
    if bg is None:
        bg = image.copy().astype('float')
        return

    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    """ 检测到的图像与背景图像做差分，得到二值化手势图像 """

    global bg
    diff = cv2.absdiff(bg.astype('uint8'), image)

    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]
    closed = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    ( _, cnts , _) = cv2.findContours(closed.copy(),
                                 cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def run(net, test_augs, key, idx):
    """
        开启摄像头收集图片信息
    """
    camera = cv2.VideoCapture(idx)

    top, right, bottom, left = 90, 380, 314, 604

    aWeight = 0.5
    num_frames = 0
    thresholded = None


    while(True):

        (grabbed, frame) = camera.read()

        if grabbed:
            frame = cv2.flip(frame, 1)
            clone = frame.copy()
            roi = frame[top:bottom, right:left]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            if num_frames >= 5 and num_frames <= 10:
                run_avg(gray, aWeight)
            if num_frames > 10:
                hand = segment(gray)

                if hand is not None:
                    (thresholded, segmented) = hand

                    cv2.drawContours(
                        clone, [segmented + (right, top)], -1, (0, 0, 255))

                else:
                    thresholded = np.zeros(gray.shape)


                input_im = cv2.merge(
                    [thresholded, thresholded, thresholded])

                data = test_augs(input_im).unsqueeze(0).to(torch.float32)
                y = F.softmax(net(data), dim=1)
                result = key[y.argmax().item()]

                cv2.putText(input_im, result, (0, 20),
                            cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)
                layout = np.zeros(input_im.shape)

                for i in range(y.shape[1]):
                    score = y[0][i].item()
                    text = "{}: {:.2f}%".format(key[i], score * 100)

                    w = int(score * 300)
                    cv2.rectangle(layout, (7, (i * 35) + 5),
                                  (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(layout, text, (10, (i * 35) + 23),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 2)

                cv2.imshow('Thesholded', np.vstack([input_im, layout]))


            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

            num_frames += 1
            cv2.imshow("video", clone)


            keypress = cv2.waitKey(50)

            if keypress == ord('q'):
                camera.release()
                break
        else:
            camera.release()
            break



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    key = {0: 'left', 1: 'right', 2: 'pause', 3: 'down', 4: 'transform'}

    # 选择摄像头 0为内部摄像头， 1为外部摄像头
    idx = 1

    # 在 CPU 上运行 定义网络并加载参数
    net = get_net().to(device)
    if device == 'cpu':
        net.load_state_dict(torch.load('./gesture_model_k.pt', map_location="cpu"))
    else:
        net.load_state_dict(torch.load('./gesture_model_k.pt'))
    net.eval()

    test_augs = transforms.Compose([
        transforms.ToTensor()
    ])


    run(net, test_augs, key, idx)
    cv2.destroyAllWindows()



