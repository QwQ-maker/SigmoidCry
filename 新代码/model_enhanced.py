# -*- coding: utf-8 -*-
"""
论文第四章实验补充
- 轻量级 CNN 在 Fashion-MNIST 上的三种激活函数对比训练
- 记录每个 epoch 的 loss 和 acc，绘制训练曲线
- 输出混淆矩阵
- 统计激活层实际输入值分布
- 保存训练好的模型权重供第五章密文推理使用
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ========== 中文字体设置 ==========
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 11

# ========== 输出目录 ==========
OUT_DIR = './outputs_ch4'
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_DIR = './models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Fashion-MNIST 类别中文名
FASHION_LABELS = ['T恤', '裤子', '套头衫', '连衣裙', '外套',
                  '凉鞋', '衬衫', '运动鞋', '包', '短靴']


# ==========================================
# 1. 自定义多项式激活函数
# ==========================================
class Poly3Activation(nn.Module):
    def forward(self, x):
        return 0.5 + 0.21232527 * x - 0.00497681 * torch.pow(x, 3)


class Poly5Activation(nn.Module):
    def forward(self, x):
        return (0.5
                + 0.23844647 * x
                - 0.01134516 * torch.pow(x, 3)
                + 0.00023539 * torch.pow(x, 5))


# ==========================================
# 2. 轻量级 CNN（同你原结构）
#    增加了 hook 以便记录激活层输入分布
# ==========================================
class LightweightFHECNN(nn.Module):
    def __init__(self, activation_type='sigmoid'):
        super().__init__()
        self.activation_type = activation_type
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, stride=2, padding=0)

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
            raise ValueError(activation_type)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(4 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 10)

        # 用于收集激活函数输入的缓冲区
        self._collect = False
        self._act1_inputs = []
        self._act2_inputs = []

    def forward(self, x):
        x = self.conv1(x)
        if self._collect:
            self._act1_inputs.append(x.detach().cpu().numpy().ravel())
        x = self.act1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        if self._collect:
            self._act2_inputs.append(x.detach().cpu().numpy().ravel())
        x = self.act2(x)
        x = self.fc2(x)
        return x


# ==========================================
# 3. 训练与评估
# ==========================================
def train_and_evaluate(activation_type, epochs=5, batch_size=64, lr=0.005):
    print(f"\n{'=' * 60}")
    print(f"开始训练：{activation_type.upper()}")
    print('=' * 60)

    device = torch.device('cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False)

    model = LightweightFHECNN(activation_type=activation_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {'epoch': [], 'train_loss': [], 'test_acc': []}

    start = time.time()
    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        running_loss = 0.0
        n_batches = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        avg_loss = running_loss / n_batches

        # ---- evaluate ----
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        acc = 100.0 * correct / total

        history['epoch'].append(epoch)
        history['train_loss'].append(avg_loss)
        history['test_acc'].append(acc)
        print(f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  test_acc={acc:.2f}%")

    train_time = time.time() - start
    print(f"训练耗时 {train_time:.1f} 秒")

    # ---- 收集预测结果用于混淆矩阵 ----
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_true, all_pred)

    # ---- 收集激活层输入分布 ----
    model._collect = True
    model._act1_inputs = []
    model._act2_inputs = []
    with torch.no_grad():
        count = 0
        for inputs, _ in testloader:
            model(inputs)
            count += inputs.size(0)
            if count >= 2000:
                break
    model._collect = False
    act1_vals = np.concatenate(model._act1_inputs)
    act2_vals = np.concatenate(model._act2_inputs)

    # 保存模型（注意：保存到 MODEL_DIR，方便 tens_full.py 加载）
    save_path = os.path.join(MODEL_DIR, f'model_{activation_type}.pth')
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至 {save_path}")

    return history, cm, act1_vals, act2_vals, history['test_acc'][-1]


# ==========================================
# 4. 绘图与数据保存
# ==========================================
def plot_training_curves(histories):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = {'sigmoid': 'k', 'poly3': 'r', 'poly5': 'b'}
    styles = {'sigmoid': '-', 'poly3': '--', 'poly5': '-.'}
    names = {'sigmoid': '标准 Sigmoid', 'poly3': '3 阶多项式 $p_3$',
             'poly5': '5 阶多项式 $p_5$'}

    # 左：loss
    ax = axes[0]
    for k, h in histories.items():
        ax.plot(h['epoch'], h['train_loss'],
                color=colors[k], linestyle=styles[k],
                marker='o', linewidth=2, label=names[k])
    ax.set_xlabel('训练轮次 (Epoch)', fontsize=12)
    ax.set_ylabel('训练损失 (Cross-Entropy Loss)', fontsize=12)
    ax.set_title('(a) 训练损失曲线', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.7)

    # 右：accuracy
    ax = axes[1]
    for k, h in histories.items():
        ax.plot(h['epoch'], h['test_acc'],
                color=colors[k], linestyle=styles[k],
                marker='s', linewidth=2, label=names[k])
    ax.set_xlabel('训练轮次 (Epoch)', fontsize=12)
    ax.set_ylabel('测试集准确率 (%)', fontsize=12)
    ax.set_title('(b) 测试集准确率曲线', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.7)

    plt.suptitle('图 4-1  三种激活函数下轻量级 CNN 的训练过程对比',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig4_1_training_curves.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[已保存] {os.path.join(OUT_DIR, 'fig4_1_training_curves.png')}")


def plot_confusion_matrices(cms):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    names = {'sigmoid': '标准 Sigmoid', 'poly3': '3 阶多项式',
             'poly5': '5 阶多项式'}
    for ax, (key, cm) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=FASHION_LABELS,
                    yticklabels=FASHION_LABELS,
                    ax=ax, cbar=False, annot_kws={'size': 9})
        ax.set_xlabel('预测类别', fontsize=11)
        ax.set_ylabel('真实类别', fontsize=11)
        ax.set_title(f'{names[key]}', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.suptitle('图 4-2  三种激活函数下模型的混淆矩阵对比（Fashion-MNIST 测试集）',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig4_2_confusion_matrix.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[已保存] {os.path.join(OUT_DIR, 'fig4_2_confusion_matrix.png')}")


def plot_activation_distributions(act_data):
    """绘制激活层输入值分布直方图（验证 [-5,5] 区间合理性）"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    names = {'sigmoid': '标准 Sigmoid', 'poly3': '3 阶多项式',
             'poly5': '5 阶多项式'}

    for col, (key, (a1, a2)) in enumerate(act_data.items()):
        # 第一个激活层（卷积后）
        ax = axes[0, col]
        ax.hist(a1, bins=80, color='steelblue', alpha=0.75, edgecolor='black')
        ax.axvline(-5, color='red', linestyle='--', linewidth=1.5, label='[-5, 5] 区间')
        ax.axvline(5, color='red', linestyle='--', linewidth=1.5)
        within = np.mean((a1 >= -5) & (a1 <= 5)) * 100
        ax.set_title(f'{names[key]}\n第 1 激活层输入  {within:.2f}% 在 [-5,5]',
                     fontsize=11)
        ax.set_xlabel('激活函数输入值', fontsize=10)
        ax.set_ylabel('频次', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.5)

        # 第二个激活层（FC1 后）
        ax = axes[1, col]
        ax.hist(a2, bins=80, color='darkorange', alpha=0.75, edgecolor='black')
        ax.axvline(-5, color='red', linestyle='--', linewidth=1.5)
        ax.axvline(5, color='red', linestyle='--', linewidth=1.5)
        within = np.mean((a2 >= -5) & (a2 <= 5)) * 100
        ax.set_title(f'{names[key]}\n第 2 激活层输入  {within:.2f}% 在 [-5,5]',
                     fontsize=11)
        ax.set_xlabel('激活函数输入值', fontsize=10)
        ax.set_ylabel('频次', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle('图 4-3  各模型激活层实际输入值分布（用于验证 [-5,5] 区间选择的合理性）',
                 fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig4_3_activation_input_dist.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[已保存] {os.path.join(OUT_DIR, 'fig4_3_activation_input_dist.png')}")


def main():
    EPOCHS = 5

    histories = {}
    cms = {}
    act_data = {}
    final_acc = {}

    for act in ['sigmoid', 'poly3', 'poly5']:
        h, cm, a1, a2, acc = train_and_evaluate(act, epochs=EPOCHS)
        histories[act] = h
        cms[act] = cm
        act_data[act] = (a1, a2)
        final_acc[act] = acc

    # 训练历史 CSV
    rows = []
    for act, h in histories.items():
        for i in range(len(h['epoch'])):
            rows.append({
                '激活函数': act,
                'epoch': h['epoch'][i],
                'train_loss': h['train_loss'][i],
                'test_acc': h['test_acc'][i],
            })
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'training_history.csv'),
                              index=False, encoding='utf-8-sig')

    # 准确率汇总 CSV
    pd.DataFrame([
        {'激活函数': '标准 Sigmoid', '测试准确率(%)': final_acc['sigmoid']},
        {'激活函数': '3 阶多项式 Poly-3', '测试准确率(%)': final_acc['poly3']},
        {'激活函数': '5 阶多项式 Poly-5', '测试准确率(%)': final_acc['poly5']},
    ]).to_csv(os.path.join(OUT_DIR, 'final_accuracy.csv'),
              index=False, encoding='utf-8-sig')

    # 画图
    plot_training_curves(histories)
    plot_confusion_matrices(cms)
    plot_activation_distributions(act_data)

    print("\n=== 最终准确率 ===")
    for k, v in final_acc.items():
        print(f"  {k}: {v:.2f}%")
    print("\n[完成] 第四章所有图表、数据及模型权重已生成。")
    print(f"  - 图表：{OUT_DIR}")
    print(f"  - 模型：{MODEL_DIR}")


if __name__ == '__main__':
    main()
