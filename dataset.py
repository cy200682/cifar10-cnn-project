import torch
import torchvision
import torchvision.transforms as transforms

# 数据预处理
transform = transforms.Compose([  # compose:多个预处理步骤组合，目前只有一个
    transforms.ToTensor()  # ToTensor:把PIL Image变成Tensor像素从 0~255变成 0~1
    # PIL图片文件，Tensor，数字矩阵，与numpy区别，能自动求导反向传播，神经网络训练需要
])


def get_dataloader():
    # 下载并加载CIFAR10训练集
    trainset = torchvision.datasets.CIFAR10(
        root='./data',  # 存放位置
        train=True,  # True训练集
        download=True,
        transform=transform
    )

    # 数据加载器
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=4,  # 分成小批量batch
        shuffle=True  # 每个epoch打乱数据
    )

    # 测试集
    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=4,
        shuffle=False
    )

    return trainloader, testloader