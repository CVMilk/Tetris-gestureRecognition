# -*- coding: utf-8 -*-
"""
Created on 2021-07-17 

@File : collectdata.py

@Author : Liulei (mrliu_9936@163.com)

@Purpose : 利用 opencv 提取手势信息

"""

import cv2
import numpy as np
import os
from torchvision.datasets import ImageFolder


bg = None
idx = 0  # 选择摄像头

def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype('float')
        return

    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
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

def run():
    """
        开启摄像头收集图片信息
    """
    camera = cv2.VideoCapture(idx)

    #top, right, bottom, left = 90, 380, 285, 590
    top, right, bottom, left = 90, 380, 314, 604

    count = 210
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

            if num_frames < 20:
                run_avg(gray, aWeight)
            else:
                hand = segment(gray)

                if hand is not None:
                    (thresholded, segmented) = hand

                    cv2.drawContours(
                        clone, [segmented + (right, top)], -1, (0, 0, 255))


            cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

            num_frames += 1
            cv2.imshow("video", clone)

            if not thresholded is None:
                cv2.imshow('Thesholded', thresholded)

            #keypress = cv2.waitKey(1) & 0xFF
            keypress = cv2.waitKey(50)

            if keypress == ord('q'):
                break
            if keypress == ord('s'):
                #cv2.imwrite('data/{}/.jpg'.format(dtype, count), thresholded)

                cv2.imwrite('./image/binary/{}.png'.format(count),thresholded)
                cv2.imwrite('./image/primitive/{}.png'.format(count), roi)

                count += 1
                print(count, 'saved.')
        else:
            camera.release()
            break



if __name__ == '__main__':

    # 新建一个./image/primitive文件夹 用来存储摄像头采集的 原始图像
    #        ./image/binary 二值化图像

    if not os.path.exists('./image/primitive'):
        os.makedirs('./image/primitive')

    if not os.path.exists('./image/binary'):
        os.makedirs('./image/binary')


    run()
    cv2.destroyAllWindows()
