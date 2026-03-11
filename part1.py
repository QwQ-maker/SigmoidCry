import numpy as np
from numpy.polynomial import Chebyshev
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calculate_polynomial_approximation(degree, domain_limit):
    """
    计算给定阶数和定义域的 Sigmoid 切比雪夫逼近多项式系数
    """
    # 设定逼近区间 [-L, L]
    domain = [-domain_limit, domain_limit]

    # 1. 使用 numpy 内置的 Chebyshev 类在指定区间内拟合 Sigmoid 函数
    cheb_poly = Chebyshev.interpolate(sigmoid, degree, domain)

    # 2. 将切比雪夫基底转换为标准多项式基底 (a0 + a1*x + a2*x^2 + ...)
    standard_poly = cheb_poly.convert(kind=np.polynomial.Polynomial)
    coeffs = standard_poly.coef

    # 3. 计算误差 (L-infinity norm & MSE)
    x_eval = np.linspace(-domain_limit, domain_limit, 1000)
    y_true = sigmoid(x_eval)
    y_pred = standard_poly(x_eval)

    max_error = np.max(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)

    print(f"=== {degree} 阶多项式逼近结果 (区间 [-{domain_limit}, {domain_limit}]) ===")
    print("标准多项式系数 (由低次到高次):")
    for i, c in enumerate(coeffs):
        print(f"  x^{i}: {c:.8e}")
    print(f"最大绝对误差 (L_inf): {max_error:.6f}")
    print(f"均方误差 (MSE): {mse:.6f}\n")

    return standard_poly, x_eval, y_true, y_pred


# 设置实验参数
# 注意：同态加密中为了控制乘法深度，通常选择偶数函数的对称性，这里我们测试 3 阶和 5 阶
# 由于 Sigmoid 的对称性 (sigma(x) - 0.5 是奇函数)，偶数次项系数通常接近于 0
L = 5.0  # 假设神经网络层输出范围主要在 [-5, 5] 内

poly_3, x_eval, y_true, y_pred_3 = calculate_polynomial_approximation(3, L)
poly_5, _, _, y_pred_5 = calculate_polynomial_approximation(5, L)

# 绘制验证图 (供论文使用)
plt.figure(figsize=(10, 5))
plt.plot(x_eval, y_true, 'k-', label='True Sigmoid', linewidth=2)
plt.plot(x_eval, y_pred_3, 'r--', label='Degree-3 Chebyshev')
plt.plot(x_eval, y_pred_5, 'b-.', label='Degree-5 Chebyshev')
plt.title(f'Polynomial Approximation of Sigmoid on [-{L}, {L}]')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()