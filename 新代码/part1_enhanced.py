# -*- coding: utf-8 -*-
"""
论文第三章实验补充
- Sigmoid 的切比雪夫多项式逼近（3 阶 / 5 阶）
- 偶数项截断前后对比
- 逼近曲线图、误差曲线图、乘法深度示意图
"""

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os

# ========== 中文字体设置（Windows）==========
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 11

# ========== 输出目录 ==========
OUT_DIR = './outputs_ch3'
os.makedirs(OUT_DIR, exist_ok=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_chebyshev(degree, L):
    """
    在 [-L, L] 区间上用切比雪夫插值拟合 Sigmoid，并转换为标准幂基系数。
    返回：原始系数（含浮点误差产生的偶数次项）、截断后系数（偶数次项强制置零）
    """
    domain = [-L, L]
    cheb = Chebyshev.interpolate(sigmoid, degree, domain)
    standard = cheb.convert(kind=Polynomial)
    coeffs_raw = standard.coef.copy()

    # 偶数项截断（保留 a0 即 x^0 项作为常数 0.5，其它偶数次项置零）
    coeffs_trunc = coeffs_raw.copy()
    for i in range(1, len(coeffs_trunc)):
        if i % 2 == 0:
            coeffs_trunc[i] = 0.0
    return coeffs_raw, coeffs_trunc


def eval_poly(coeffs, x):
    """按低到高次序的系数计算多项式值"""
    y = np.zeros_like(x)
    for i, c in enumerate(coeffs):
        y = y + c * (x ** i)
    return y


def error_metrics(coeffs, L, n_points=2000):
    """计算最大绝对误差和均方误差"""
    x = np.linspace(-L, L, n_points)
    y_true = sigmoid(x)
    y_pred = eval_poly(coeffs, x)
    max_err = float(np.max(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    return max_err, mse, x, y_true, y_pred


def main():
    L = 5.0
    print("=" * 70)
    print(f"切比雪夫多项式逼近 Sigmoid 函数  逼近区间：[-{L}, {L}]")
    print("=" * 70)

    # 求解 3 阶和 5 阶系数
    c3_raw, c3_trunc = fit_chebyshev(3, L)
    c5_raw, c5_trunc = fit_chebyshev(5, L)

    # 误差评估（截断后）
    max3, mse3, x_eval, y_true, y3 = error_metrics(c3_trunc, L)
    max5, mse5, _, _, y5 = error_metrics(c5_trunc, L)

    # 误差评估（截断前，用于对比偶数项截断对精度的影响）
    max3_r, mse3_r, _, _, _ = error_metrics(c3_raw, L)
    max5_r, mse5_r, _, _, _ = error_metrics(c5_raw, L)

    # ---------- 打印结果 ----------
    print("\n[3 阶多项式系数]")
    print(f"{'幂次':<6}{'原始系数':<22}{'截断后系数':<22}")
    for i in range(len(c3_raw)):
        print(f"x^{i:<4}{c3_raw[i]:<22.8e}{c3_trunc[i]:<22.8e}")

    print("\n[5 阶多项式系数]")
    print(f"{'幂次':<6}{'原始系数':<22}{'截断后系数':<22}")
    for i in range(len(c5_raw)):
        print(f"x^{i:<4}{c5_raw[i]:<22.8e}{c5_trunc[i]:<22.8e}")

    print("\n[误差对比]")
    print(f"{'方案':<20}{'最大绝对误差':<18}{'均方误差':<18}")
    print(f"{'3阶 截断前':<20}{max3_r:<18.6e}{mse3_r:<18.6e}")
    print(f"{'3阶 截断后':<20}{max3:<18.6e}{mse3:<18.6e}")
    print(f"{'5阶 截断前':<20}{max5_r:<18.6e}{mse5_r:<18.6e}")
    print(f"{'5阶 截断后':<20}{max5:<18.6e}{mse5:<18.6e}")

    # ---------- 保存 CSV ----------
    df_coef = pd.DataFrame({
        '幂次': list(range(6)),
        '3阶_原始': list(c3_raw) + [0.0, 0.0],
        '3阶_截断后': list(c3_trunc) + [0.0, 0.0],
        '5阶_原始': list(c5_raw),
        '5阶_截断后': list(c5_trunc),
    })
    df_coef.to_csv(os.path.join(OUT_DIR, 'polynomial_coefficients.csv'),
                   index=False, encoding='utf-8-sig')

    df_err = pd.DataFrame({
        '方案': ['3阶 截断前', '3阶 截断后', '5阶 截断前', '5阶 截断后'],
        '最大绝对误差': [max3_r, max3, max5_r, max5],
        '均方误差': [mse3_r, mse3, mse5_r, mse5],
    })
    df_err.to_csv(os.path.join(OUT_DIR, 'error_metrics.csv'),
                  index=False, encoding='utf-8-sig')

    # ====================================================================
    # 图 3-1：Sigmoid vs 3 阶 vs 5 阶 主对比曲线
    # ====================================================================
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x_eval, y_true, 'k-', linewidth=2.5, label='原始 Sigmoid 函数')
    ax.plot(x_eval, y3, 'r--', linewidth=2.0, label=f'3 阶多项式 $p_3(x)$')
    ax.plot(x_eval, y5, 'b-.', linewidth=2.0, label=f'5 阶多项式 $p_5(x)$')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('输入 $x$', fontsize=12)
    ax.set_ylabel('函数值', fontsize=12)
    ax.set_title(f'图 3-1  Sigmoid 函数及其切比雪夫低阶多项式逼近对比（区间 [-{int(L)}, {int(L)}]）',
                 fontsize=13)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig3_1_approximation.png'), dpi=200)
    plt.close()
    print(f"\n[已保存] {os.path.join(OUT_DIR, 'fig3_1_approximation.png')}")

    # ====================================================================
    # 图 3-2：逼近误差曲线
    # ====================================================================
    err3 = y_true - y3
    err5 = y_true - y5

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x_eval, err3, 'r-', linewidth=2.0, label=f'3 阶误差  最大={max3:.4f}')
    ax.plot(x_eval, err5, 'b-', linewidth=2.0, label=f'5 阶误差  最大={max5:.4f}')
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.fill_between(x_eval, err3, 0, alpha=0.15, color='red')
    ax.fill_between(x_eval, err5, 0, alpha=0.15, color='blue')
    ax.set_xlabel('输入 $x$', fontsize=12)
    ax.set_ylabel('逼近误差 $e(x) = \\sigma(x) - p_n(x)$', fontsize=12)
    ax.set_title(f'图 3-2  低阶多项式对 Sigmoid 函数的逐点逼近误差曲线', fontsize=13)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig3_2_error_curve.png'), dpi=200)
    plt.close()
    print(f"[已保存] {os.path.join(OUT_DIR, 'fig3_2_error_curve.png')}")

    # ====================================================================
    # 图 3-3：乘法深度计算路径示意
    # ====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左：3 阶
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('(a) 3 阶多项式  $p_3(x) = 0.5 + a_1 x + a_3 x^3$\n乘法深度 = 2',
                 fontsize=12)

    def box(ax, xy, w, h, text, color='#cce5ff'):
        from matplotlib.patches import FancyBboxPatch
        p = FancyBboxPatch((xy[0], xy[1]), w, h,
                           boxstyle="round,pad=0.05",
                           linewidth=1.2, edgecolor='black', facecolor=color)
        ax.add_patch(p)
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
                ha='center', va='center', fontsize=11)

    box(ax, (0.5, 2.5), 1.6, 1.0, '$x$', '#fff2cc')
    box(ax, (3.5, 4.0), 1.6, 1.0, '$x^2 = x \\cdot x$\n深度=1', '#cce5ff')
    box(ax, (3.5, 1.0), 1.6, 1.0, '保留 $x$', '#fff2cc')
    box(ax, (6.5, 2.5), 1.6, 1.0, '$x^3 = x^2 \\cdot x$\n深度=2', '#cce5ff')

    ax.annotate('', xy=(3.5, 4.5), xytext=(2.1, 3.2),
                arrowprops=dict(arrowstyle='->', lw=1.3))
    ax.annotate('', xy=(3.5, 1.5), xytext=(2.1, 2.8),
                arrowprops=dict(arrowstyle='->', lw=1.3))
    ax.annotate('', xy=(6.5, 3.0), xytext=(5.1, 4.5),
                arrowprops=dict(arrowstyle='->', lw=1.3))
    ax.annotate('', xy=(6.5, 3.0), xytext=(5.1, 1.5),
                arrowprops=dict(arrowstyle='->', lw=1.3))

    # 右：5 阶
    ax = axes[1]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_title('(b) 5 阶多项式  $p_5(x) = 0.5 + a_1 x + a_3 x^3 + a_5 x^5$\n乘法深度 = 3',
                 fontsize=12)

    box(ax, (0.3, 2.5), 1.6, 1.0, '$x$', '#fff2cc')
    box(ax, (3.0, 4.0), 1.6, 1.0, '$x^2$\n深度=1', '#cce5ff')
    box(ax, (5.7, 2.5), 1.6, 1.0, '$x^3 = x^2 \\cdot x$\n深度=2', '#cce5ff')
    box(ax, (8.5, 2.5), 1.6, 1.0, '$x^5 = x^3 \\cdot x^2$\n深度=3', '#f8cbad')

    ax.annotate('', xy=(3.0, 4.5), xytext=(1.9, 3.2),
                arrowprops=dict(arrowstyle='->', lw=1.3))
    ax.annotate('', xy=(5.7, 3.0), xytext=(4.6, 4.5),
                arrowprops=dict(arrowstyle='->', lw=1.3))
    ax.annotate('', xy=(5.7, 3.0), xytext=(1.9, 2.8),
                arrowprops=dict(arrowstyle='->', lw=1.3))
    ax.annotate('', xy=(8.5, 3.0), xytext=(7.3, 3.0),
                arrowprops=dict(arrowstyle='->', lw=1.3))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig3_3_mult_depth.png'), dpi=200)
    plt.close()
    print(f"[已保存] {os.path.join(OUT_DIR, 'fig3_3_mult_depth.png')}")

    print("\n[完成] 第三章所有图表与数据已生成至 ./outputs_ch3/")


if __name__ == '__main__':
    main()
