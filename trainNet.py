# -*- coding: utf-8 -*-
"""
Created on 2021-07-18 

@File : trainNet.py

@Author : Liulei (mrliu_9936@163.com)

@Purpose : k折交叉验证 训练手势识别模型

"""
import torch
from torchvision.datasets import ImageFolder
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import models
from data_loader import MyDataset
import os
import glob
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
from torch.nn import init
import time
import torch.backends.cudnn as cudnn


def get_data(dataPath):
    """ 读取图像文件， 返回图像的路径 以及label """
    img_path = []
    key = {}
    sum = 0
    labelname = os.listdir(dataPath)

    for i, label in enumerate(labelname):
        # 找到对应标签下所有的图像文件
        image_dir = glob.glob(os.path.join(dataPath, label, "*.png"))
        sum += len(image_dir)
        print(label, '图片数量为:', len(image_dir))
        for image in image_dir:
            img_path.append((image, i))
        key[label] = i
    print("%d 个图像信息已经加载!!!" % (sum))

    np.random.shuffle(img_path)
    return img_path, key


# K折交叉代码
def get_k_fold_data(k, i, image_label):
    """ 返回第 i 折 交叉验证时所需要的训练和验证数据 """
    assert k > 1
    fold_size = len(image_label) // k

    train_data, valid_data = None, None

    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part = image_label[idx]
        if(i == j):
            valid_data = X_part
        elif train_data is None:
            train_data = X_part
        else:
            train_data += X_part
    return train_data, valid_data

# 初始化权重选择
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

# 选择resnet18作为分类模型
def get_net():
    net = models.resnet18(pretrained=False)
    net.fc = nn.Linear(512,5)
    init_weights(net, init_type='kaiming', gain=0.02)
    return net

# 模型评估
def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            n += y.shape[0]
    return acc_sum / n


# 模型训练
def train(i, net, train_loader, valid_loader, Epoch,
          loss_func, opt, lr_decay, device):

    net = net.to(device)
    print("training on ", device)
    start = time.time()
    test_acc_max_l = []
    train_acc_max_l = []
    train_l_min_l = []

    for epoch in range(Epoch):
        batch_count = 0
        train_l_sum, train_acc_sum, test_acc_sum, n = 0.0, 0.0, 0.0, 0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss_func(y_hat, y)
            opt.zero_grad()
            l.backward()
            opt.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        # 至此，一个epoches完成,进行评估
        test_acc_sum = evaluate_accuracy(valid_loader, net)
        train_l_min_l.append(train_l_sum / batch_count)
        train_acc_max_l.append(train_acc_sum / n)
        test_acc_max_l.append(test_acc_sum)

        if epoch % 10 == 0:
            print('fold %d epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
                % (i + 1, epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc_sum))

        lr_decay.step()

    k_loss_mean = sum(train_l_min_l) / len(train_l_min_l)
    k_train_acc_mean = sum(train_acc_max_l) / len(train_acc_max_l)
    k_test_acc_mean = sum(test_acc_max_l) / len(test_acc_max_l)

    print('==' * 10)
    print('fold %d, mean_loss %.4f, train_acc_mean %.3f, test_acc_mean %.3f, time:%.3f min'
          % (i + 1, k_loss_mean, k_train_acc_mean, k_test_acc_mean, (time.time()-start) / 60))
    print('==' * 10)



def k_fold(k, image_label, Epoch, loss_func, opt, lr_decay,
           device, batch_size, num_workers):

    for i in range(k):
        train_path, valid_path = get_k_fold_data(k, i, image_label)

        train_data = MyDataset(train_path, mode = 'train')
        valid_data = MyDataset(valid_path, mode='test')

        print('训练集大小为:', len(train_data))
        print('测试集大小为:', len(valid_data))

        train_loader = DataLoader(train_data,
                                 batch_size = batch_size,
                                 shuffle=True,
                                 num_workers = num_workers)
        valid_loader = DataLoader(valid_data,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)

        train(i, net, train_loader, valid_loader, Epoch,
            loss_func, opt, lr_decay, device)


if __name__ == '__main__':

    # 数据路径
    dataPath = './trainDateset/train'
    img_path, key = get_data(dataPath)

    # 定义超参数
    on_server = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    num_workers = 0 if on_server is False else 4

    # cudnn.benchmark 在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销，一般都会加。
    cudnn.benchmark = True

    # 训练轮数
    Epoch = 100
    #学习率
    leaing_rate = 1e-2
    # 批量大小
    batch_size = 64

    # 定义resnet18
    net = torch.nn.DataParallel(get_net()).cuda()  # 使用多GPU分布计算

    # 定义损失函数
    loss_func = nn.CrossEntropyLoss()

    # 定义优化器
    opt = torch.optim.Adam(net.parameters(), lr=leaing_rate, weight_decay=0.001)

    # 学习率衰减
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, [20, 40, 60, 80])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 开始5折交叉验证
    k = 5
    k_fold(k, img_path, Epoch, loss_func, opt, lr_decay,
           device, batch_size, num_workers)

    # 保存模型
    torch.save(net.module.state_dict(), "./gesture_model_k.pt")


