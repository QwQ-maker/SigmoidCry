# -*- coding: utf-8 -*-
"""
论文第五章核心实验（性能优化版）

实现完整 CNN 密文推理：Conv2d -> Poly Activation -> AvgPool -> FC1 -> Poly Activation -> FC2

性能优化说明：
- 卷积层用 im2col 思想：每个输出位置构造稀疏明文掩码，通过 enc.dot(mask) 得到标量密文
- 卷积输出的 4×144 = 576 个标量密文执行多项式激活
- 平均池化后展平为 144 维特征
- 为适配 FC 层的批量密文运算，对 144 维中间特征执行一次重打包
  （这一步可视为客户端在中间介入的工程优化，云端从未获得密钥）
- FC1、激活2、FC2 在打包密文向量上批量执行，大幅提速
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import tenseal as ts

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 11

OUT_DIR = './outputs_ch5'
os.makedirs(OUT_DIR, exist_ok=True)
MODEL_DIR = './models'

FASHION_LABELS = ['T恤', '裤子', '套头衫', '连衣裙', '外套',
                  '凉鞋', '衬衫', '运动鞋', '包', '短靴']

N_SAMPLES = 200
N_LOGITS_VIS = 5


# =====================================================================
# 模型
# =====================================================================
class Poly3Activation(nn.Module):
    def forward(self, x):
        return 0.5 + 0.21232527 * x - 0.00497681 * torch.pow(x, 3)


class Poly5Activation(nn.Module):
    def forward(self, x):
        return (0.5 + 0.23844647 * x
                - 0.01134516 * torch.pow(x, 3)
                + 0.00023539 * torch.pow(x, 5))


class LightweightFHECNN(nn.Module):
    def __init__(self, activation_type):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5, stride=2, padding=0)
        if activation_type == 'poly3':
            self.act1 = Poly3Activation(); self.act2 = Poly3Activation()
        elif activation_type == 'poly5':
            self.act1 = Poly5Activation(); self.act2 = Poly5Activation()
        else:
            raise ValueError(activation_type)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(4 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x); x = self.act1(x); x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x); x = self.act2(x); x = self.fc2(x)
        return x


def make_ckks_context(poly_modulus_degree=16384,
                      coeff_mod_bit_sizes=None, scale_bits=26):
    if coeff_mod_bit_sizes is None:
        coeff_mod_bit_sizes = [40, 26, 26, 26, 26, 26, 40]
    ctx = ts.context(ts.SCHEME_TYPE.CKKS,
                     poly_modulus_degree=poly_modulus_degree,
                     coeff_mod_bit_sizes=coeff_mod_bit_sizes)
    ctx.global_scale = 2 ** scale_bits
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx


def im2col_indices(in_h, in_w, kh, kw, stride):
    out_h = (in_h - kh) // stride + 1
    out_w = (in_w - kw) // stride + 1
    cols = []
    for i in range(out_h):
        for j in range(out_w):
            top, left = i * stride, j * stride
            idxs = []
            for di in range(kh):
                for dj in range(kw):
                    idxs.append((top + di) * in_w + (left + dj))
            cols.append(idxs)
    return np.array(cols).T, out_h, out_w


def conv2d_encrypted(enc_x, conv_weight, conv_bias,
                     in_h, in_w, im2col_idx, out_h, out_w):
    """对每个 (kernel, 输出位置) 构造稀疏 mask，做 enc.dot(mask)"""
    num_kernels = conv_weight.shape[0]
    plain_in_size = in_h * in_w
    num_windows = out_h * out_w
    output_channels = []
    for k in range(num_kernels):
        kernel = conv_weight[k, 0].reshape(-1)
        bias_k = float(conv_bias[k])
        ch_out = []
        for w_idx in range(num_windows):
            mask = np.zeros(plain_in_size, dtype=np.float64)
            for pos, val in zip(im2col_idx[:, w_idx], kernel):
                mask[pos] = float(val)
            enc_scalar = enc_x.dot(mask.tolist()) + bias_k
            ch_out.append(enc_scalar)
        output_channels.append(ch_out)
    return output_channels


def apply_poly_scalar_list(scalar_list, coeffs):
    return [s.polyval(coeffs) for s in scalar_list]


def avgpool_2x2_scalars(ch, h, w):
    out_h, out_w = h // 2, w // 2
    result = []
    for i in range(out_h):
        for j in range(out_w):
            i0, j0 = 2 * i, 2 * j
            tl = i0 * w + j0
            tr = i0 * w + (j0 + 1)
            bl = (i0 + 1) * w + j0
            br = (i0 + 1) * w + (j0 + 1)
            result.append((ch[tl] + ch[tr] + ch[bl] + ch[br]) * 0.25)
    return result


def repack_to_vector(ctx, scalar_list):
    """客户端中间介入：解密 scalar 列表后重打包为向量密文"""
    decrypted = [float(s.decrypt()[0]) for s in scalar_list]
    return ts.ckks_vector(ctx, decrypted)


def encrypted_forward(ctx, image_28x28, model_weights, poly_coeffs,
                      im2col_idx, out_h, out_w):
    timings = {'encrypt': 0, 'conv1': 0, 'act1': 0, 'pool': 0,
               'repack': 0, 'fc1': 0, 'act2': 0, 'fc2': 0, 'decrypt': 0}

    t = time.time()
    enc_x = ts.ckks_vector(ctx, image_28x28.reshape(-1).astype(np.float64).tolist())
    timings['encrypt'] = time.time() - t

    t = time.time()
    conv_ch = conv2d_encrypted(enc_x, model_weights['conv1.weight'],
                               model_weights['conv1.bias'],
                               28, 28, im2col_idx, out_h, out_w)
    timings['conv1'] = time.time() - t

    t = time.time()
    activated = [apply_poly_scalar_list(c, poly_coeffs) for c in conv_ch]
    timings['act1'] = time.time() - t

    t = time.time()
    pooled = [avgpool_2x2_scalars(c, out_h, out_w) for c in activated]
    flat_scalars = []
    for c in pooled:
        flat_scalars.extend(c)
    timings['pool'] = time.time() - t

    t = time.time()
    enc_feat = repack_to_vector(ctx, flat_scalars)
    timings['repack'] = time.time() - t

    t = time.time()
    enc_h1 = (enc_feat.mm(model_weights['fc1.weight'].T.tolist())
              + model_weights['fc1.bias'].tolist())
    timings['fc1'] = time.time() - t

    t = time.time()
    enc_h1_act = enc_h1.polyval(poly_coeffs)
    timings['act2'] = time.time() - t

    t = time.time()
    enc_out = (enc_h1_act.mm(model_weights['fc2.weight'].T.tolist())
               + model_weights['fc2.bias'].tolist())
    timings['fc2'] = time.time() - t

    t = time.time()
    logits = np.array(enc_out.decrypt())
    timings['decrypt'] = time.time() - t

    return logits, timings


def load_test_data(n_samples):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    testset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform)
    images, labels = [], []
    for i in range(n_samples):
        img, lbl = testset[i]
        images.append(img.squeeze(0).numpy())
        labels.append(lbl)
    return images, labels


def load_model_weights(activation_type):
    model = LightweightFHECNN(activation_type)
    pth = os.path.join(MODEL_DIR, f'model_{activation_type}.pth')
    if not os.path.exists(pth):
        raise FileNotFoundError(f"找不到 {pth}，请先运行 model_enhanced.py")
    state = torch.load(pth, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    weights = {k: state[k].numpy() for k in state.keys()}
    return model, weights


def plain_inference(model, images):
    model.eval()
    logits_all = []
    with torch.no_grad():
        for img in images:
            t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            logits_all.append(model(t).numpy().ravel())
    logits_all = np.array(logits_all)
    return logits_all, np.argmax(logits_all, axis=1)


def run_experiment(activation_type, ctx, images, labels,
                   im2col_idx, out_h, out_w):
    print(f"\n{'=' * 70}\n开始 {activation_type.upper()} 完整密文 CNN 推理\n{'=' * 70}")
    model, weights = load_model_weights(activation_type)
    coeffs = ([0.5, 0.21232527, 0.0, -0.00497681] if activation_type == 'poly3'
              else [0.5, 0.23844647, 0.0, -0.01134516, 0.0, 0.00023539])

    plain_logits, plain_preds = plain_inference(model, images)

    cipher_logits, cipher_preds, all_timings = [], [], []
    t_overall = time.time()
    for i, img in enumerate(images):
        t0 = time.time()
        logits, timings = encrypted_forward(
            ctx, img, weights, coeffs, im2col_idx, out_h, out_w)
        timings['total'] = time.time() - t0
        cipher_logits.append(logits)
        cipher_preds.append(int(np.argmax(logits)))
        all_timings.append(timings)

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_overall
            eta = elapsed / (i + 1) * (len(images) - i - 1)
            print(f"  {i+1:3d}/{len(images)}  本张{timings['total']:.2f}s  "
                  f"已耗{elapsed:.0f}s  剩{eta:.0f}s  "
                  f"真={FASHION_LABELS[labels[i]]} "
                  f"明={FASHION_LABELS[plain_preds[i]]} "
                  f"密={FASHION_LABELS[cipher_preds[i]]}")

    cipher_logits = np.array(cipher_logits)
    cipher_preds = np.array(cipher_preds)
    labels_arr = np.array(labels)

    plain_acc = np.mean(plain_preds == labels_arr) * 100
    cipher_acc = np.mean(cipher_preds == labels_arr) * 100
    consistency = np.mean(plain_preds == cipher_preds) * 100
    max_err = float(np.max(np.abs(plain_logits - cipher_logits)))
    mean_err = float(np.mean(np.abs(plain_logits - cipher_logits)))

    print(f"\n[{activation_type.upper()} 结果汇总]")
    print(f"  明文准确率:   {plain_acc:.2f}%")
    print(f"  密文准确率:   {cipher_acc:.2f}%")
    print(f"  分类一致率:   {consistency:.2f}%")
    print(f"  Logits 最大误差: {max_err:.4e}")
    print(f"  Logits 平均误差: {mean_err:.4e}")

    keys = ['encrypt', 'conv1', 'act1', 'pool', 'repack',
            'fc1', 'act2', 'fc2', 'decrypt', 'total']
    avg = {k: float(np.mean([t[k] for t in all_timings])) for k in keys}
    print("  阶段平均耗时（秒）:")
    for k in keys:
        print(f"    {k:<10} {avg[k]:.4f}")

    return {
        'activation': activation_type,
        'plain_logits': plain_logits, 'cipher_logits': cipher_logits,
        'plain_preds': plain_preds, 'cipher_preds': cipher_preds,
        'labels': labels_arr,
        'plain_acc': plain_acc, 'cipher_acc': cipher_acc,
        'consistency': consistency,
        'max_logit_err': max_err, 'mean_logit_err': mean_err,
        'avg_timings': avg, 'all_timings': all_timings,
    }


def plot_logits_comparison(r3, r5, n_vis=N_LOGITS_VIS):
    fig, axes = plt.subplots(2, n_vis, figsize=(4 * n_vis, 8))
    x = np.arange(10); width = 0.35
    for col in range(n_vis):
        idx = col
        for row, r in enumerate([r3, r5]):
            ax = axes[row, col]
            ax.bar(x - width / 2, r['plain_logits'][idx], width,
                   label='明文', color='steelblue', edgecolor='black')
            ax.bar(x + width / 2, r['cipher_logits'][idx], width,
                   label='密文', color='salmon', edgecolor='black')
            name = 'Poly-3' if row == 0 else 'Poly-5'
            ax.set_title(f'{name}  样本#{idx}\n真实={FASHION_LABELS[r["labels"][idx]]}',
                         fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(FASHION_LABELS, rotation=60, fontsize=8)
            ax.set_ylabel('Logits 数值', fontsize=9)
            if col == 0:
                ax.legend(fontsize=9)
            ax.grid(True, linestyle=':', alpha=0.5)
    plt.suptitle('图 5-1  明文推理与密文推理输出 Logits 数值对比',
                 fontsize=13, y=1.00)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'fig5_1_logits_comparison.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[已保存] {p}")


def plot_time_breakdown(r3, r5):
    stages = ['encrypt', 'conv1', 'act1', 'pool', 'repack',
              'fc1', 'act2', 'fc2', 'decrypt']
    cn = ['输入加密', '卷积层', '激活层1', '平均池化', '重打包',
          '全连接1', '激活层2', '全连接2', '结果解密']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes[0]
    t3 = [r3['avg_timings'][s] for s in stages]
    t5 = [r5['avg_timings'][s] for s in stages]
    bottoms = [0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(stages)))
    for i, s in enumerate(stages):
        vals = [t3[i], t5[i]]
        ax.bar(['Poly-3', 'Poly-5'], vals, bottom=bottoms,
               label=cn[i], color=colors[i], edgecolor='black', linewidth=0.5)
        for j, v in enumerate(vals):
            if v > max(sum(t3), sum(t5)) * 0.03:
                ax.text(j, bottoms[j] + v / 2, f'{v:.2f}s',
                        ha='center', va='center', fontsize=9)
        bottoms = [bottoms[0] + vals[0], bottoms[1] + vals[1]]
    ax.set_ylabel('单张图像平均耗时（秒）', fontsize=12)
    ax.set_title('(a) 各阶段耗时分解', fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=9)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)

    ax = axes[1]
    total = sum(t3)
    sizes = [v / total * 100 for v in t3]
    ax.pie(sizes, labels=cn, autopct='%1.1f%%', colors=colors,
           startangle=90, textprops={'fontsize': 10})
    ax.set_title(f'(b) Poly-3 各阶段耗时占比  总耗时 {total:.2f}s', fontsize=12)

    plt.suptitle('图 5-2  CKKS 密文 CNN 推理各阶段耗时分解', fontsize=13, y=1.02)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'fig5_2_time_breakdown.png')
    plt.savefig(p, dpi=200, bbox_inches='tight'); plt.close()
    print(f"[已保存] {p}")


def plot_time_comparison(r3, r5):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    methods = ['Poly-3 密文推理', 'Poly-5 密文推理']
    totals = [r3['avg_timings']['total'], r5['avg_timings']['total']]
    colors = ['#4C72B0', '#C44E52']
    bars = ax.bar(methods, totals, color=colors, edgecolor='black',
                  linewidth=1.2, width=0.55)
    for b, v in zip(bars, totals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(totals) * 0.02,
                f'{v:.3f} s', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    ax.set_ylabel('单张图像平均推理耗时（秒）', fontsize=12)
    ax.set_title('图 5-3  Poly-3 与 Poly-5 完整密文 CNN 推理耗时对比',
                 fontsize=13)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)
    ratio = totals[1] / totals[0]
    ax.text(0.5, max(totals) * 0.5, f'Poly-5 / Poly-3 = {ratio:.2f}×',
            transform=ax.transData, ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#fff2cc', edgecolor='black'))
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'fig5_3_time_comparison.png')
    plt.savefig(p, dpi=200); plt.close()
    print(f"[已保存] {p}")


def save_summary(r3, r5):
    df = pd.DataFrame([
        {'激活函数': 'Poly-3',
         '明文准确率(%)': f"{r3['plain_acc']:.2f}",
         '密文准确率(%)': f"{r3['cipher_acc']:.2f}",
         '明密文一致率(%)': f"{r3['consistency']:.2f}",
         'Logits最大误差': f"{r3['max_logit_err']:.4e}",
         'Logits平均误差': f"{r3['mean_logit_err']:.4e}",
         '单张总耗时(s)': f"{r3['avg_timings']['total']:.4f}"},
        {'激活函数': 'Poly-5',
         '明文准确率(%)': f"{r5['plain_acc']:.2f}",
         '密文准确率(%)': f"{r5['cipher_acc']:.2f}",
         '明密文一致率(%)': f"{r5['consistency']:.2f}",
         'Logits最大误差': f"{r5['max_logit_err']:.4e}",
         'Logits平均误差': f"{r5['mean_logit_err']:.4e}",
         '单张总耗时(s)': f"{r5['avg_timings']['total']:.4f}"},
    ])
    df.to_csv(os.path.join(OUT_DIR, 'cipher_inference_summary.csv'),
              index=False, encoding='utf-8-sig')

    stages = ['encrypt', 'conv1', 'act1', 'pool', 'repack',
              'fc1', 'act2', 'fc2', 'decrypt', 'total']
    df_t = pd.DataFrame({
        '阶段': stages,
        'Poly-3 平均耗时(s)': [r3['avg_timings'][s] for s in stages],
        'Poly-5 平均耗时(s)': [r5['avg_timings'][s] for s in stages],
    })
    df_t.to_csv(os.path.join(OUT_DIR, 'time_breakdown.csv'),
                index=False, encoding='utf-8-sig')
    print(f"[已保存] {os.path.join(OUT_DIR, 'cipher_inference_summary.csv')}")
    print(f"[已保存] {os.path.join(OUT_DIR, 'time_breakdown.csv')}")


def main():
    print(f"加载 {N_SAMPLES} 张 Fashion-MNIST 测试图像 ...")
    images, labels = load_test_data(N_SAMPLES)

    im2col_idx, out_h, out_w = im2col_indices(28, 28, 5, 5, 2)
    print(f"卷积输出尺寸: {out_h}x{out_w} = {out_h*out_w} 窗口/通道")

    print("初始化 CKKS 上下文 (Poly-3) ...")
    ctx3 = make_ckks_context(
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[40, 26, 26, 26, 26, 26, 40], scale_bits=26)
    r3 = run_experiment('poly3', ctx3, images, labels,
                        im2col_idx, out_h, out_w)

    print("\n初始化 CKKS 上下文 (Poly-5, 链更长) ...")
    ctx5 = make_ckks_context(
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[40, 26, 26, 26, 26, 26, 26, 26, 40], scale_bits=26)
    r5 = run_experiment('poly5', ctx5, images, labels,
                        im2col_idx, out_h, out_w)

    plot_logits_comparison(r3, r5)
    plot_time_breakdown(r3, r5)
    plot_time_comparison(r3, r5)
    save_summary(r3, r5)

    print("\n[完成] 第五章所有图表与数据已生成。")
    print(f"  输出目录: {OUT_DIR}")


if __name__ == '__main__':
    main()
