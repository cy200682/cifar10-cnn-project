import torch


def test_model(net, testloader, device):

    correct = 0
    total = 0

    with torch.no_grad():  # 不计算梯度

        for data in testloader:

            images, labels = data

            images, labels = images.to(device), labels.to(device)

            outputs = net(images)

            _, predicted = torch.max(outputs, 1)
            # 找到最大概率类别

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    print("Accuracy:", 100 * correct / total, "%")