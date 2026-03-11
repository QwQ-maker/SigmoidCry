import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time


# ==========================================
# 1. 自定义多项式激活函数 (对应第3章的推导)
# ==========================================
class Poly3Activation(nn.Module):
    def __init__(self):
        super(Poly3Activation, self).__init__()

    def forward(self, x):
        # P3(x) = 0.5 + 0.21232527x - 0.00497681x^3
        return 0.5 + 0.21232527 * x - 0.00497681 * torch.pow(x, 3)


class Poly5Activation(nn.Module):
    def __init__(self):
        super(Poly5Activation, self).__init__()

    def forward(self, x):
        # P5(x) = 0.5 + 0.23844647x - 0.01134516x^3 + 0.00023539x^5
        return 0.5 + 0.23844647 * x - 0.01134516 * torch.pow(x, 3) + 0.00023539 * torch.pow(x, 5)


# ==========================================
# 2. 面向 FHE 的轻量级 CNN 架构设计
# ==========================================
class LightweightFHECNN(nn.Module):
    def __init__(self, activation_type='sigmoid'):
        super(LightweightFHECNN, self).__init__()

        # 卷积层1: 输入1通道(灰度图), 输出4通道, 5x5卷积核, 步长2, 无填充
        # 输出特征图尺寸: (28 - 5) / 2 + 1 = 12 -> [4, 12, 12]
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, stride=2, padding=0)

        # 激活层选择
        if activation_type == 'sigmoid':
            self.act1 = nn.Sigmoid()
            self.act2 = nn.Sigmoid()
        elif activation_type == 'poly3':
            self.act1 = Poly3Activation()
            self.act2 = Poly3Activation()
        elif activation_type == 'poly5':
            self.act1 = Poly5Activation()
            self.act2 = Poly5Activation()
        else:
            raise ValueError("Unsupported activation type")

        # 平均池化层: 2x2核, 步长2 (替换Max Pooling以适配同态)
        # 输出特征图尺寸: 12 / 2 = 6 -> [4, 6, 6]
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        # 全连接层1: 展平后维度 4 * 6 * 6 = 144
        self.fc1 = nn.Linear(4 * 6 * 6, 64)

        # 全连接层2 (输出层): 10分类
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.fc2(x)
        return x


# ==========================================
# 3. 训练与评估流程
# ==========================================
def train_and_evaluate(activation_type, epochs=5):
    print(f"\n[{activation_type.upper()}] 开始训练与评估...")

    # 强制使用 CPU，避免 AMD 显卡环境配置问题拖慢进度
    device = torch.device("cpu")

    # 数据加载 (Fashion-MNIST)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False)

    # 初始化模型、损失函数与优化器
    model = LightweightFHECNN(activation_type=activation_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # 训练循环
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    train_time = time.time() - start_time

    # 测试评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"[{activation_type.upper()}] 训练耗时: {train_time:.2f}秒 | 测试集准确率: {accuracy:.2f}%")

    # 保存模型权重供后续同态推理使用
    torch.save(model.state_dict(), f'model_{activation_type}.pth')
    return accuracy


if __name__ == '__main__':
    # 对比三种激活函数的表现
    acc_sigmoid = train_and_evaluate('sigmoid', epochs=5)
    acc_poly3 = train_and_evaluate('poly3', epochs=5)
    acc_poly5 = train_and_evaluate('poly5', epochs=5)

    print("\n=== 最终准确率对比汇总 ===")
    print(f"标准 Sigmoid 基准准确率: {acc_sigmoid:.2f}%")
    print(f"3阶多项式 (Poly-3) 准确率: {acc_poly3:.2f}%")
    print(f"5阶多项式 (Poly-5) 准确率: {acc_poly5:.2f}%")