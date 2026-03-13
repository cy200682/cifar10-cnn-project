import torch
import torch.nn as nn
import torch.optim as optim


def train_model(net, trainloader, device):

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # CrossEntropyLoss 是 PyTorch 实现好的交叉熵损失函数
    # 内部已经包含 softmax

    # 定义优化器
    optimizer = optim.SGD(
        net.parameters(),
        lr=0.001,
        momentum=0.9
    )
    # SGD随机梯度下降
    # momentum动量参数，减少震荡

    for epoch in range(2):  # 整个数据集训练两遍

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # i = batch编号
            # data = 数据

            inputs, labels = data
            # 图片，标签

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            # 清空梯度，不然会累计

            outputs = net(inputs)
            # 前向传播

            loss = criterion(outputs, labels)
            # 计算损失

            loss.backward()
            # 反向传播

            optimizer.step()
            # 更新权重
            # W = W - lr × gradient

            running_loss += loss.item()
            # loss是tensor，用item变成数字

            if i % 2000 == 1999:  # 每2000次打印一次
                print(epoch + 1, i + 1, running_loss / 2000)
                running_loss = 0.0

    print("Finished Training")

    # 保存模型
    torch.save(net.state_dict(), "cnn_cifar10.pth")
    print("Model saved!")