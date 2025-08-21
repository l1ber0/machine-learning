"""
-*- coding: utf-8 -*-
@Author: l1ber0
@School: TYUST (Taiyuan University of Science and Technology)
@Time: 2025/8/21 15:00
@Project: l1ber0
@File: implementation-of-multi-layer-perceptron-from-scratch.py
@email:693096838@qq.com

Description:
    本文件由 PyCharm 自动创建，作者：l1ber0。
"""

import torch
from torch import nn
from d2l import torch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 256

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256


#设置感知机参数
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True,device=device) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True,device=device))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True,device=device) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True,device=device))

def relu(x):
    return torch.relu(x)

def net(X):
    X = X.reshape(-1, num_inputs)
    H = torch.relu(X @ W1 + b1)
    return H @ W2 + b2

loss = nn.CrossEntropyLoss()

#训练过程
#设置训练轮次
epochs = 10
#学习速率
lr = 0.1
#更新
optimizer = torch.optim.SGD(params, lr=lr)
d2l.train_ch6(net = net(train_iter),train_iter=train_iter,test_iter=test_iter,epochs=epochs,lr=lr,device=device)

