"""
-*- coding: utf-8 -*-
@Author: l1ber0
@School: TYUST (Taiyuan University of Science and Technology)
@Time: 2025/8/20 17:38
@Project: l1ber0
@File: multilayer-perceptron.py
@email:693096838@qq.com

Description:
    本文件由 PyCharm 自动创建，作者：l1ber0。
"""
import torch
from torch import nn
from d2l import torch as d2l

import torch
from torch import nn
from d2l import torch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 定义网络结构
net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

net = net.to(device)


# 初始化权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


net.apply(init_weights)

# 超参数设置
batchsize, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='mean')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batchsize)

# 可视化工具
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'test acc'])


# 训练函数（加入可视化）
def train(net, train_iter, test_iter, loss, num_epochs, lr, device):
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        # 训练损失总和与样本数
        metric = d2l.Accumulator(2)
        net.train()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            metric.add(l.item() * X.shape[0], X.shape[0])
        train_loss = metric[0] / metric[1]

        # 测试准确率
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)

        # 更新图表
        animator.add(epoch + 1, (train_loss, test_acc))
    print(f'训练结束：训练损失 {train_loss:.3f}, 测试准确率 {test_acc:.3f}')


# 开始训练
train(net, train_iter, test_iter, loss, num_epochs, lr, device)

