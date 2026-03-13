import torch#张量积算
import torchvision#视觉库 提供常见数据集，图像处理和已实现模型
import torchvision.transforms as transforms#图像预处理工具
import matplotlib.pyplot as plt#显示图片，画图
import numpy as np

# 数据预处理
transform = transforms.Compose([#compose:多个预处理步骤组合，目前只有一个
    transforms.ToTensor()#ToTensor:把PIL Image变成Tensor像素从 0~255变成 0~1
    #PIL图片文件，Tensor，数字矩阵，与numpy区别，能自动求导反向传播，神经网络训练需要
])

# 下载并加载CIFAR10数据集
trainset = torchvision.datasets.CIFAR10(
    root='./data',#存放位置，项目目录data
    train=True,#true训练集，#false测试集
    download=True,#如果没有下载，有就加载
    transform=transform#启动预处理
)

# 数据加载器
trainloader = torch.utils.data.DataLoader(#按批次读取数据，Dataloader作用，每次返回一个batch，不是列表是可迭代对象
    trainset,
    batch_size=4,#分成小批量batch
    shuffle=True#每个epoch打乱数据
)
testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,#测试集
    download=True,
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False#不需要打乱
)
#定义cnn
import torch.nn as nn#神经网络模块
import torch.nn.functional as F#函数形式的神经网络操作 如relu，ReLU(x) = max(0,x)

class Net(nn.Module):#创建一个神经网络类，nn.module是所有神经网络的父类
    def __init__(self):
        super(Net, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(3, 6, 5)
        '''nn.Conv2d(in_channels, out_channels, kernel_size)所以这里是：输入通道 = 3，输出通道 = 6，卷积核大小 = 5×5
        输入3 × 32 × 32
        输出6个卷积核：
        卷积核大小5×5'''
        #最开始是3*32*32，输出大小 = (输入大小 − 卷积核大小) + 1，第一层卷积返回6*28*28
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)#缩小图片，2*2取最大值，返回6*14*14
        # 第二层卷积
        self.conv2 = nn.Conv2d(6, 16, 5)#返回16*10*10
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))#第一次卷积，第一次池化
        x = self.pool(F.relu(self.conv2(x)))#第二次卷积，第二次池化
        x = x.view(-1, 16 * 5 * 5)#把三维特征图变成一维向量 -1是自动计算batch大小
        x = F.relu(self.fc1(x))#全连接再relu
        x = F.relu(self.fc2(x))
        x = self.fc3(x)#10个类别的得分
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#模型放到gpu有gpu用gpu否则cpu

net = Net().to(device)#神经网络参数传到gpu
#定义损失函数
import torch.optim as optim#导入优化算法模块
criterion = nn.CrossEntropyLoss()#nn.CrossEntropyLoss() 是 PyTorch 里实现好的“交叉熵损失函数类”，内有softmax
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)#定义优化器，SGD随机梯度下降 np.parameters是所有训练参数包括卷积核权重，全连接层权重
#momentum动量参数，减小震荡
for epoch in range(2):#
#整个数据集训练两遍
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
#i batch编号，data 数据
        inputs, labels = data
#图片，标签
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
#清空梯度，不然会累计
        outputs = net(inputs)
#前向传播
        loss = criterion(outputs, labels)
#计算损失
        loss.backward()
#计算梯度，反向传播
        optimizer.step()
#更新权重W = W - lr × gradient
        running_loss += loss.item()
#loss是tensor，用item变成数字
        if i % 2000 == 1999:#每两千次打印一次
            print(epoch + 1, i + 1, running_loss / 2000)
            running_loss = 0.0

print("Finished Training")
correct = 0#预测图片数量
total = 0

with torch.no_grad():#不计算梯度

    for data in testloader:

        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)

        _, predicted = torch.max(outputs, 1)#找到最大概率类别

        total += labels.size(0)#+batch

        correct += (predicted == labels).sum().item()

print("Accuracy:", 100 * correct / total, "%")

import matplotlib.pyplot as plt
import numpy as np

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(testloader)
images, labels = next(dataiter)

images = images.to(device)

outputs = net(images)

_, predicted = torch.max(outputs, 1)

images = images.cpu()

# 显示图片
img = torchvision.utils.make_grid(images)
npimg = img.numpy()

plt.imshow(np.transpose(npimg, (1, 2, 0)))
plt.show()

print("真实标签: ", [classes[labels[j]] for j in range(4)])
print("预测结果: ", [classes[predicted[j]] for j in range(4)])