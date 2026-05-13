# -*- coding: utf-8 -*-
"""
加分项实验
1) 不同逼近区间 [-3,3] / [-5,5] / [-7,7] 对比
2) 不同 CKKS 参数 (poly_modulus_degree = 8192/16384/32768) 对比
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev, Polynomial

import tenseal as ts

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 11

OUT_DIR = './outputs_extras'
os.makedirs(OUT_DIR, exist_ok=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def fit_truncate(degree, L):
    """切比雪夫拟合 + 偶数项截断"""
    cheb = Chebyshev.interpolate(sigmoid, degree, [-L, L])
    coef = cheb.convert(kind=Polynomial).coef.copy()
    for i in range(1, len(coef)):
        if i % 2 == 0:
            coef[i] = 0.0
    return coef


def eval_poly(coef, x):
    y = np.zeros_like(x)
    for i, c in enumerate(coef):
        y = y + c * (x ** i)
    return y


# =====================================================================
# 实验 1：不同逼近区间对比
# =====================================================================
def experiment_domain_compare():
    print("\n" + "=" * 70)
    print("实验 1：不同逼近区间对 Sigmoid 多项式拟合误差的影响")
    print("=" * 70)

    domains = [3.0, 5.0, 7.0]
    degrees = [3, 5]

    # 数据采集
    results = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # 为了让所有图共用同一画图区间，统一用更大的范围去画 sigmoid
    x_global = np.linspace(-8, 8, 2000)
    y_global = sigmoid(x_global)

    for row, deg in enumerate(degrees):
        for col, L in enumerate(domains):
            coef = fit_truncate(deg, L)

            # 区间内误差
            x_eval = np.linspace(-L, L, 1000)
            y_true = sigmoid(x_eval)
            y_pred = eval_poly(coef, x_eval)
            max_err = float(np.max(np.abs(y_true - y_pred)))
            mse = float(np.mean((y_true - y_pred) ** 2))

            # 全局曲线（拟合多项式在更宽范围上的表现，看是否发散）
            y_pred_global = eval_poly(coef, x_global)

            results.append({
                '多项式阶数': deg,
                '逼近区间': f'[-{L:.0f}, {L:.0f}]',
                '最大绝对误差': max_err,
                '均方误差': mse,
            })

            ax = axes[row, col]
            ax.plot(x_global, y_global, 'k-', linewidth=2,
                    label='Sigmoid 原函数')
            ax.plot(x_global, y_pred_global, 'r--', linewidth=1.8,
                    label=f'{deg} 阶多项式')
            ax.axvspan(-L, L, alpha=0.15, color='green',
                       label=f'拟合区间 [-{L:.0f},{L:.0f}]')
            ax.set_xlim(-8, 8)
            ax.set_ylim(-0.5, 1.5)
            ax.set_xlabel('$x$')
            ax.set_ylabel('函数值')
            ax.set_title(f'{deg} 阶多项式，拟合区间 [-{L:.0f}, {L:.0f}]\n'
                         f'区间内最大误差={max_err:.4f}', fontsize=10)
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, linestyle=':', alpha=0.5)

    plt.suptitle('图 加-1  不同逼近区间下 Sigmoid 多项式拟合效果对比',
                 fontsize=13, y=1.00)
    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'fig_extra_1_domain_compare.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[已保存] {fig_path}")

    # CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, 'domain_compare.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[已保存] {csv_path}")
    print(df.to_string(index=False))


# =====================================================================
# 实验 2：不同 CKKS 参数对比
# =====================================================================
def benchmark_ckks_params(poly_degree, coeff_bits, scale_bits, repeat=5):
    """对一个简化的 FC 网络做 CKKS 推理，测耗时与精度"""
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_degree,
        coeff_mod_bit_sizes=coeff_bits
    )
    ctx.global_scale = 2 ** scale_bits
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()

    np.random.seed(42)
    in_dim, hid_dim, out_dim = 144, 64, 10
    x = np.random.randn(in_dim) * 0.3
    w1 = np.random.randn(hid_dim, in_dim) * 0.1
    b1 = np.random.randn(hid_dim) * 0.1
    w2 = np.random.randn(out_dim, hid_dim) * 0.1
    b2 = np.random.randn(out_dim) * 0.1

    coeffs = [0.5, 0.21232527, 0.0, -0.00497681]  # Poly-3

    # 明文参考
    h_pt = w1.dot(x) + b1
    h_pt_act = (coeffs[0] + coeffs[1] * h_pt
                + coeffs[3] * h_pt ** 3)
    out_pt = w2.dot(h_pt_act) + b2

    times = []
    err = None
    for _ in range(repeat):
        t = time.time()
        enc_x = ts.ckks_vector(ctx, x.tolist())
        enc_h = enc_x.mm(w1.T.tolist()) + b1.tolist()
        enc_h_act = enc_h.polyval(coeffs)
        enc_out = enc_h_act.mm(w2.T.tolist()) + b2.tolist()
        dec_out = np.array(enc_out.decrypt())
        times.append(time.time() - t)
        err = float(np.max(np.abs(dec_out - out_pt)))

    return float(np.mean(times)), err


def experiment_ckks_params():
    print("\n" + "=" * 70)
    print("实验 2：不同 CKKS 参数 (poly_modulus_degree) 对密文推理性能的影响")
    print("=" * 70)

    configs = [
        # (poly_degree, coeff_bits, scale_bits, 说明)
        (8192,  [40, 26, 26, 26, 26, 40], 26, 'N=8192'),
        (16384, [40, 26, 26, 26, 26, 40], 26, 'N=16384'),
        (32768, [40, 26, 26, 26, 26, 40], 26, 'N=32768'),
    ]

    results = []
    for poly_degree, bits, sb, name in configs:
        print(f"\n>>> 测试 {name}  coeff_bits={bits}  scale=2^{sb}")
        try:
            t_avg, err = benchmark_ckks_params(poly_degree, bits, sb, repeat=3)
            print(f"    平均耗时 {t_avg:.4f}s，最大误差 {err:.4e}")
            results.append({
                'poly_modulus_degree': poly_degree,
                '配置名称': name,
                '平均耗时(s)': t_avg,
                '最大误差': err,
                '状态': '成功',
            })
        except Exception as e:
            print(f"    失败：{e}")
            results.append({
                'poly_modulus_degree': poly_degree,
                '配置名称': name,
                '平均耗时(s)': np.nan,
                '最大误差': np.nan,
                '状态': f'失败：{e}',
            })

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, 'ckks_params.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 绘图（双轴：耗时 + 误差）
    success = [r for r in results if r['状态'] == '成功']
    if len(success) >= 2:
        fig, ax1 = plt.subplots(figsize=(9, 5.5))
        names = [r['配置名称'] for r in success]
        ts_arr = [r['平均耗时(s)'] for r in success]
        errs = [r['最大误差'] for r in success]

        color1 = '#4C72B0'
        bars = ax1.bar(names, ts_arr, color=color1,
                       edgecolor='black', linewidth=1.0,
                       label='平均耗时', width=0.45)
        for b, v in zip(bars, ts_arr):
            ax1.text(b.get_x() + b.get_width() / 2, v + max(ts_arr) * 0.02,
                     f'{v:.3f}s', ha='center', va='bottom', fontsize=10)
        ax1.set_ylabel('平均推理耗时（秒）', color=color1, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xlabel('CKKS 多项式模数度数 (poly_modulus_degree)', fontsize=12)

        ax2 = ax1.twinx()
        color2 = '#C44E52'
        ax2.plot(names, errs, color=color2, marker='o', linewidth=2.0,
                 markersize=9, label='最大误差')
        ax2.set_ylabel('明密文最大绝对误差', color=color2, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_yscale('log')

        plt.title('图 加-2  不同 CKKS 参数下密文推理的耗时与精度对比',
                  fontsize=13)
        ax1.grid(True, axis='y', linestyle=':', alpha=0.5)
        fig.tight_layout()
        fig_path = os.path.join(OUT_DIR, 'fig_extra_2_ckks_params.png')
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"[已保存] {fig_path}")

    print(f"[已保存] {csv_path}")
    print(df.to_string(index=False))


def main():
    experiment_domain_compare()
    experiment_ckks_params()
    print("\n[完成] 加分项实验所有图表与数据已生成至 ./outputs_extras/")


if __name__ == '__main__':
    main()
