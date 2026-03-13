import torch

from dataset import get_dataloader
from model import Net
from train import train_model
from test import test_model
from visualize import visualize_predictions


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型放到gpu，有gpu用gpu，否则cpu

    trainloader, testloader = get_dataloader()

    net = Net().to(device)
    # 神经网络参数传到gpu

    # 训练
    train_model(net, trainloader, device)

    # 测试
    test_model(net, testloader, device)

    # 可视化预测
    visualize_predictions(net, testloader, device)


if __name__ == "__main__":
    main()