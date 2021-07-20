# -*- coding: utf-8 -*-
"""
Created on 2021-07-18 

@File : game.py

@Author : Liulei (mrliu_9936@163.com)

@Purpose :  主程序直接运行 俄罗斯方块小游戏

"""

import pygame
import random
import cv2
from trainNet import get_net
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np


"""
10 x 20 square grid
shapes: S, Z, I, O, J, L, T
represented in order by 0 - 6
"""

pygame.font.init()

# GLOBALS VARS
s_width = 800
s_height = 700
play_width = 300  # meaning 300 // 10 = 30 width per block
play_height = 600  # meaning 600 // 20 = 20 height per blo ck
block_size = 30
top_left_x = (s_width - play_width) // 2
top_left_y = s_height - play_height


bg = None  # 存背景图像
idx = 0    # 调用内部摄像头， 1调用外部摄像头

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
key = {0: 'left', 1: 'right', 2: 'pause', 3: 'down', 4: 'transform'}
# 定义网络并加载参数
net = get_net().to(device)
if not torch.cuda.is_available():
    net.load_state_dict(torch.load('./gesture_model_k.pt', map_location="cpu"))
else:
    net.load_state_dict(torch.load('./gesture_model_k.pt'))
net.eval()

# 转为tensor
test_augs = transforms.Compose([
    transforms.ToTensor()
])

# SHAPE FORMATS

S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255),
                (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]
# index 0 - 6 represent shape


class Piece(object):
    rows = 20  # y
    columns = 10  # x
    # 方块类
    def __init__(self, column, row, shape):
        self.x = column
        self.y = row
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0  # number from 0-3


def create_grid(locked_positions={}):
    # 绘制网格
    grid = [[(0, 0, 0) for x in range(10)] for x in range(20)]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j, i) in locked_positions:
                c = locked_positions[(j, i)]
                grid[i][j] = c
    return grid


def convert_shape_format(shape):
    # 生成方块
    positions = []
    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((shape.x + j, shape.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)

    return positions


def valid_space(shape, grid):
    accepted_positions = [[(j, i) for j in range(
        10) if grid[i][j] == (0, 0, 0)] for i in range(20)]
    accepted_positions = [j for sub in accepted_positions for j in sub]
    formatted = convert_shape_format(shape)

    for pos in formatted:
        if pos not in accepted_positions:
            if pos[1] > -1:
                return False

    return True


def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True
    return False


def get_shape():
    global shapes, shape_colors

    return Piece(5, 0, random.choice(shapes))


def draw_text_middle(text, size, color, surface):
    font = pygame.font.SysFont('comicsans', size, bold=True)
    label = font.render(text, 1, color)

    surface.blit(label, (top_left_x + play_width/2 - (label.get_width() / 2),
                         top_left_y + play_height/2 - label.get_height()/2))


def draw_grid(surface, row, col):
    sx = top_left_x
    sy = top_left_y
    for i in range(row):
        pygame.draw.line(surface, (128, 128, 128), (sx, sy + i*30),
                         (sx + play_width, sy + i * 30))  # horizontal lines
        for j in range(col):
            pygame.draw.line(surface, (128, 128, 128), (sx + j * 30, sy),
                             (sx + j * 30, sy + play_height))  # vertical lines


def clear_rows(grid, locked):
    # need to see if row is clear the shift every other row above down one

    inc = 0
    for i in range(len(grid)-1, -1, -1):
        row = grid[i]
        if (0, 0, 0) not in row:
            inc += 1
            # add positions to remove from locked
            ind = i
            for j in range(len(row)):
                try:
                    del locked[(j, i)]
                except:
                    continue
    if inc > 0:
        for key in sorted(list(locked), key=lambda x: x[1])[::-1]:
            x, y = key
            if y < ind:
                newKey = (x, y + inc)
                locked[newKey] = locked.pop(key)
        return 1
    else:
        return 0

def draw_next_shape(shape, surface):
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('Next Shape', 1, (255, 255, 255))

    sx = top_left_x + play_width + 50
    sy = top_left_y + play_height/2 - 100
    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, shape.color,
                                 (sx + j*30, sy + i*30, 30, 30), 0)

    surface.blit(label, (sx + 10, sy - 30))


class Button(object):
    # 按钮类
    def __init__(self, text, color, x=None, y=None, **kwargs):
        font = pygame.font.SysFont('comicsans', 30)
        self.surface = font.render(text, True, color)
        self.WIDTH = self.surface.get_width()
        self.HEIGHT = self.surface.get_height()
        self.x = x
        self.y = y

    def display(self):
        # 在主界面显示按钮
        win.blit(self.surface, (self.x, self.y))

    def check_click(self, position):
        # 检查按键上是否有鼠标
        x_match = position[0] > self.x and position[0] < self.x + self.WIDTH
        y_match = position[1] > self.y and position[1] < self.y + self.HEIGHT

        if x_match and y_match:
            return True
        else:
            return False

sx = top_left_x + play_width + 80
play_button = Button('Play', (255, 255, 255), sx, 450)
exit_button = Button('Exit', (255, 255, 255), sx, 500)
pause_button = Button('Pause', (255, 255, 255), sx, 550)


def draw_window(surface, score):
    surface.fill((0, 0, 0))
    # 绘制得分
    font = pygame.font.SysFont('comicsans', 60)
    label = font.render('Score:{:.2f}'.format(score), 1, (255, 255, 255))

    surface.blit(label, (top_left_x + play_width /
                         2 - (label.get_width() / 2), 30))

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            pygame.draw.rect(
                surface, grid[i][j], (top_left_x + j * 30, top_left_y + i * 30, 30, 30), 0)

    draw_grid(surface, 20, 10)
    pygame.draw.rect(surface, (255, 0, 0), (top_left_x,
                                            top_left_y, play_width, play_height), 5)

    # 显示三个按钮
    play_button.display()
    exit_button.display()
    pause_button.display()

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

def main():
    global grid

    locked_positions = {}  # (x,y):(255,0,0)
    grid = create_grid(locked_positions)

    change_piece = False
    run = True
    current_piece = get_shape()
    next_piece = get_shape()
    clock = pygame.time.Clock()
    fall_time = 0 # 计时
    level_time = 0
    fall_speed = 0.7 # 下落速度
    fps_clock = pygame.time.Clock()
    fps = 10
    score = 0 # 得分
    width = 600 # 视频图像宽度
    top, right, bottom, left = 90, 380, 314, 604 # ROI（即手势区域）
    camera = cv2.VideoCapture(idx) # 调用摄像头


    count = 0 # 计数器
    aWeight = 0.5
    thresholded = None

    while run:

        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        level_time += clock.get_rawtime()
        clock.tick()

        readable, frame = camera.read() # 读取视频帧
        if not readable:
            break

        frame = cv2.flip(frame, 1)  # 图像镜像处理（左右手问题）
        clone = frame.copy()
        roi = frame[top:bottom, right:left]  # 手势位置
        cv2.rectangle(frame, (left, top),
                      (right, bottom), (0, 255, 0), 2)  # 绘制绿框

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        direction = 'pause'

        if count >= 5 and count <= 10:
            run_avg(gray, aWeight)
        if count > 10:
            hand = segment(gray)

            if hand is not None:
                (thresholded, segmented) = hand
            else:
                thresholded = np.zeros(gray.shape)

            input_im = cv2.merge(
                [thresholded, thresholded, thresholded])

            data = test_augs(input_im).unsqueeze(0).to(torch.float32)
            y = F.softmax(net(data), dim=1)
            direction = key[y.argmax().item()]

            cv2.putText(input_im, direction, (0, 20),
                        cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)
            layout = np.zeros(input_im.shape)

            for i in range(y.shape[1]):
                scores = y[0][i].item()
                text = "{}: {:.2f}%".format(key[i], scores * 100)

                w = int(scores * 300)
                cv2.rectangle(layout, (7, (i * 35) + 5),
                              (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(layout, text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 2)

            cv2.imshow('Thesholded', np.vstack([input_im, layout]))
            print('-[INFO] Update movement to ', direction)

        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imshow("video", clone)
        count += 1


        if direction == 'left' and count % 4 == 0:
            # 左移
            current_piece.x -= 1
            if not valid_space(current_piece, grid):
                current_piece.x += 1
        elif direction == 'right' and count % 4 == 0:
            # 右移
            current_piece.x += 1
            if not valid_space(current_piece, grid):
                current_piece.x -= 1
        elif direction == 'transform' and count % 8 == 0 :
            # 变换形状
            current_piece.rotation = current_piece.rotation + \
                1 % len(current_piece.shape)
            if not valid_space(current_piece, grid):
                current_piece.rotation = current_piece.rotation - \
                    1 % len(current_piece.shape)
        elif direction == 'down' and count % 2 == 0:
            current_piece.y += 1
            if not (valid_space(current_piece, grid)) and current_piece.y > 0:
                current_piece.y -= 1
                change_piece = True


        if fall_time/1000 >= fall_speed:
            # 变速
            fall_time = 0
            current_piece.y += 1
            if not (valid_space(current_piece, grid)) and current_piece.y > 0:
                current_piece.y -= 1
                change_piece = True

        for event in pygame.event.get():
            # 获取键盘相应
            if event.type == pygame.QUIT:
                # 点×退出游戏
                run = False
                camera.release()
                pygame.display.quit()
                quit()

        shape_pos = convert_shape_format(current_piece)

        # add piece to the grid for drawing
        for i in range(len(shape_pos)):
            x, y = shape_pos[i]
            if y > -1:
                grid[y][x] = current_piece.color

        # IF PIECE HIT GROUND
        if change_piece:
            for pos in shape_pos:
                p = (pos[0], pos[1])
                locked_positions[p] = current_piece.color
            current_piece = next_piece
            next_piece = get_shape()
            change_piece = False

            # call four times to check for multiple clear rows
            if clear_rows(grid, locked_positions):
                score += 1

        draw_window(win, score)
        draw_next_shape(next_piece, win)
        pygame.display.update()

        if check_lost(locked_positions):
            run = False
        if pygame.mouse.get_pressed()[0]:
            # 获取鼠标左键点击相应
            if play_button.check_click(pygame.mouse.get_pos()):
                pass
            if exit_button.check_click(pygame.mouse.get_pos()):
                run = False
                pygame.display.quit()
                #quit()
            if pause_button.check_click(pygame.mouse.get_pos()):
                run = False

        #cv2.imshow('frame', frame)

        # cv2.waitKey(100)
        fps_clock.tick(fps) # 计时器

    draw_text_middle("You Lost", 40, (255, 255, 255), win)
    pygame.display.update()
    pygame.time.delay(2000)


def main_menu():
    run = True
    while run:
        win.fill((0, 0, 0))
        draw_text_middle('Press any key to begin.', 60, (255, 255, 255), win)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                main()
    pygame.quit()


win = pygame.display.set_mode((s_width, s_height))
pygame.display.set_caption('Tetris')
main_menu()  # start game
