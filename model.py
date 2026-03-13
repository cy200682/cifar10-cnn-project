import torch.nn as nn
import torch.nn.functional as F


# 创建一个神经网络类
class Net(nn.Module):  # nn.Module是所有神经网络的父类
    def __init__(self):
        super(Net, self).__init__()

        # 第一层卷积
        self.conv1 = nn.Conv2d(3, 6, 5)

        '''
        nn.Conv2d(in_channels, out_channels, kernel_size)

        输入通道 = 3
        输出通道 = 6
        卷积核大小 = 5×5
        '''

        # 最开始是3×32×32
        # 输出大小 = (输入大小 − 卷积核大小) + 1
        # 第一层卷积返回6×28×28

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 缩小图片，2×2取最大值
        # 返回6×14×14

        # 第二层卷积
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 返回16×10×10

        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        # 第一次卷积 + ReLU + 池化
        x = self.pool(F.relu(self.conv1(x)))

        # 第二次卷积 + ReLU + 池化
        x = self.pool(F.relu(self.conv2(x)))

        # 把三维特征图变成一维向量
        x = x.view(-1, 16 * 5 * 5)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 10个类别的得分

        return x