import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_predictions(net, testloader, device):

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