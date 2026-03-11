import tenseal as ts
import numpy as np
import time


def setup_fhe_context():
    """
    初始化 CKKS 同态计算上下文
    """
    print("[1] 正在初始化 CKKS 同态上下文...")

    # 终极参数组合：
    # 素数链总位宽: 40 + 26 + 26 + 26 + 40 = 158 bits (满足 <218 bits 安全限制)
    # 中间留有 3 个 26 bits 素数，恰好满足 Poly-5 的 3 层深度消耗！
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[40, 26, 26, 26, 40]
    )

    # 全局缩放因子必须严格小于素数的位宽，这里设为 2^26
    context.global_scale = 2 ** 26

    # 【最关键的一步】开启引擎的自动内存与噪声管理机制
    context.auto_relin = True  # 自动重线性化（解决密文体积膨胀）
    context.auto_rescale = True  # 自动重缩放（解决 Scale 溢出）
    context.auto_mod_switch = True  # 自动模数切换（对齐明密文计算层级）

    context.generate_galois_keys()
    context.generate_relin_keys()
    return context


def benchmark_inference():
    # 模拟网络结构维度 (对应论文 4.1 节设计)
    # 展平后输入维度 144，FC1 映射到 64，FC2 映射到 10
    input_dim = 144
    hidden_dim = 64
    output_dim = 10

    # 随机生成模拟特征与明文权重 (时间开销与数值无关)
    np.random.seed(42)
    sample_feature = np.random.randn(input_dim)
    fc1_weight = np.random.randn(hidden_dim, input_dim)
    fc1_bias = np.random.randn(hidden_dim)
    fc2_weight = np.random.randn(output_dim, hidden_dim)
    fc2_bias = np.random.randn(output_dim)

    # 定义截断多项式系数 (按 0阶, 1阶, 2阶... 排列)
    poly3_coeffs = [0.5, 0.21232527, 0.0, -0.00497681]
    poly5_coeffs = [0.5, 0.23844647, 0.0, -0.01134516, 0.0, 0.00023539]

    # --- 1. 明文基准测速 (CPU) ---
    start_time = time.time()
    for _ in range(100):  # 循环 100 次取平均以提高精度
        pt_out = fc1_weight.dot(sample_feature) + fc1_bias
        # 模拟 Poly-3 明文激活
        pt_act = poly3_coeffs[0] + poly3_coeffs[1] * pt_out + poly3_coeffs[3] * (pt_out ** 3)
        pt_final = fc2_weight.dot(pt_act) + fc2_bias
    pt_time = (time.time() - start_time) / 100
    print(f"\n[测试完成] 明文基准单次推理平均耗时: {pt_time:.6f} 秒")

    # --- 2. 建立同态环境与数据加密 ---
    ctx = setup_fhe_context()
    print("[2] 正在加密特征向量...")
    enc_x = ts.ckks_vector(ctx, sample_feature)

    # --- 3. 密文 Poly-3 推理测速 ---
    print("[3] 开始执行 FHE Poly-3 密文推理...")
    start_time = time.time()

    # FC1 明密文乘加
    enc_out1 = enc_x.mm(fc1_weight.T) + fc1_bias
    # Poly-3 密文激活 (TenSEAL 底层自动处理乘法深度与重缩放)
    enc_act1 = enc_out1.polyval(poly3_coeffs)
    # FC2 明密文乘加
    enc_final_3 = enc_act1.mm(fc2_weight.T) + fc2_bias

    fhe_poly3_time = time.time() - start_time
    print(f"[测试完成] 密文 Poly-3 推理耗时: {fhe_poly3_time:.4f} 秒")
    print(f"  -> 相对明文耗时膨胀倍数: {fhe_poly3_time / pt_time:.0f} 倍")

    # --- 4. 密文 Poly-5 推理测速 ---
    print("\n[4] 开始执行 FHE Poly-5 密文推理...")
    # 重新加密以重置噪声和深度
    enc_x_fresh = ts.ckks_vector(ctx, sample_feature)
    start_time = time.time()

    # FC1 明密文乘加
    enc_out2 = enc_x_fresh.mm(fc1_weight.T) + fc1_bias
    # Poly-5 密文激活
    enc_act2 = enc_out2.polyval(poly5_coeffs)
    # FC2 明密文乘加
    enc_final_5 = enc_act2.mm(fc2_weight.T) + fc2_bias

    fhe_poly5_time = time.time() - start_time
    print(f"[测试完成] 密文 Poly-5 推理耗时: {fhe_poly5_time:.4f} 秒")
    print(f"  -> 相对明文耗时膨胀倍数: {fhe_poly5_time / pt_time:.0f} 倍")


if __name__ == "__main__":
    benchmark_inference()