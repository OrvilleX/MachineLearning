#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:32:21 2019

@author: santiago
"""

import os
import time
import torch
import torchvision
from flyfish_net import Net
from torch.utils.data import Dataset, DataLoader

MARS = None

# 这里提供了两种数据集MARS和Market-1501
if (MARS):  # MARS
    root = "/home/santiago/dataset/MARS-v160809/"
    train_dir = os.path.join(root, "bbox_train")
    test_dir = os.path.join(root, "bbox_test")  # height256 width128
else:  # Market-1501
    root = "/Users/apple/PycharmProjects/MachineLearning/cnn/mot/market/pytorch/"
    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "query")  #

# torchvision.transforms.Compose(transforms)

# 数据增强部分 DataArgumentation
# Compose看做是一种容器
# 将多个transform组合起来使用,包括以下transforms
# "Compose", "ToTensor", "ToPILImage", "Normalize", "Resize",
# "Scale", "CenterCrop", "Pad", "Lambda", "RandomCrop",
# "RandomHorizontalFlip", "RandomVerticalFlip", "RandomResizedCrop",
# "RandomSizedCrop", "FiveCrop", "TenCrop","LinearTransformation",
# "ColorJitter", "RandomRotation", "Grayscale", "RandomGrayscale"
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop((128, 64), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),  # 对载入的图片按照随机概率进行水平翻转
    torchvision.transforms.ToTensor(),  ##转换成PyTorch能够计算和处理的Tensor数据类型的变量
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]这组数据是官网提供的数据,该数据来源的网址
# https://pytorch.org/docs/stable/torchvision/models.html

# 还可以采用如下表达方式
# transformList = []
# transformList.append(transforms.RandomHorizontalFlip())
# transformList.append(transforms.ToTensor())
# transformList.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# transform_train = transforms.Compose(transformList)

# 数据加载
trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(train_dir, transform=transform_train),
    batch_size=64, shuffle=True
)
testloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(test_dir, transform=transform_test),
    batch_size=64, shuffle=True
)

# torch.utils.data.DataSet

num_classes = len(trainloader.dataset.classes)
print(num_classes)  # mars 625 ,market1501 751
device = "mps"  # 根据使用GPU 还是CPU 操作进行变更

# net definition
start_epoch = 0  # 训练过程保存模型，训练过程中断后加载先前保存的模型，这里相当于记录训练次数
net = Net(num_classes=num_classes)

net.to(device)
total_epoch = 10000  # 总共训练次数

# 加载先前训练的模型
if (os.path.isfile("./checkpoint/ckpt.pytorch")):
    print('Loading model')
    checkpoint = torch.load("./checkpoint/ckpt.pytorch")
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('new model')

# loss and optimizer
# 网络定义好之后，还要定义模型的损失函数和对参数进行优化的优化函数
# 优化函数使用的SGD，损失函数使用的是交叉熵
criterion = torch.nn.CrossEntropyLoss()
learning_rate = 0.1  # lr
optimizer = torch.optim.SGD(net.parameters(), learning_rate, momentum=0.9, weight_decay=5e-4)
best_acc = 0.


# train function for each epoch
def train(epoch):
    print("\nEpoch : %d" % (epoch + 1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = 20
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        # forward
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # backward
        # 引入优化算法，要调用optimizer.zero_grad()完成对模型参数梯度的归0
        # 如果没有optimizer.zero_grad()，梯度会累加在一起，结果不收敛
        # loss.backward() 根据计算图自动计算每个节点的梯度值，并根据需要进行保留
        #
        # backward主要是模型的反向传播中的自动梯度计算，在网络定义中的forward是模型前向传播中的矩阵计算
        # optimizer.step()作用是使用计算得到的梯度值对各个节点参数进行梯度更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumurating
        training_loss += loss.item()  ##注意显存
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # 每个一段时间将损失的值打印出来看，确定模型误差越来越小
        # Variable会放在计算图中，然后进行前向传播，反向传播，自动求导
        # ，可以通过data取出Variable中的tensor数值
        # 如果要打印，还可以用loss.data[0]
        if (idx + 1) % interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100. * (idx + 1) / len(trainloader), end - start, training_loss / interval, correct, total,
                100. * correct / total
            ))
            training_loss = 0.
            start = time.time()

    return train_loss / len(trainloader), 1. - correct / total


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():  # 不需要bp的forward, 注意model.eval() 不等于 torch.no_grad()
        for idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)

        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
            100. * (idx + 1) / len(testloader), end - start, test_loss / len(testloader), correct, total,
            100. * correct / total
        ))

    # saving checkpoint
    acc = 100. * correct / total
    if acc >= best_acc:
        best_acc = acc
        print("Saving parameters to checkpoint/ckpt.pytorch")
        checkpoint = {
            'net_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './checkpoint/ckpt.pytorch')

    return test_loss / len(testloader), 1. - correct / total


# 学习率衰减方法：
# 线性衰减。例如：每过20个epochs学习率减半
# 指数衰减。例如：每过20个epochs将学习率乘以0.1
def learning_rate_decay():  # 学习率衰减（learning rate decay）
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))


def main():
    for epoch in range(start_epoch, start_epoch + total_epoch):
        train_loss, train_err = train(epoch)
        test_loss, test_err = test(epoch)
        if (epoch + 1) % 20 == 0:
            learning_rate_decay()


if __name__ == '__main__':
    main()
