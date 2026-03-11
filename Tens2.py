import tenseal as ts
import numpy as np
import time


def setup_fhe_context(poly_degree=16384, coeff_bits=None, scale_bits=26):
    """
    初始化 CKKS 同态计算上下文
    """
    if coeff_bits is None:
        coeff_bits = [40, 26, 26, 26, 26, 40]

    print(f"[1] 正在初始化 CKKS 上下文: poly_modulus_degree={poly_degree}, "
          f"coeff_mod_bit_sizes={coeff_bits}, global_scale=2^{scale_bits}")

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_degree,
        coeff_mod_bit_sizes=coeff_bits
    )

    context.global_scale = 2 ** scale_bits

    # 自动管理机制
    context.auto_relin = True
    context.auto_rescale = True
    context.auto_mod_switch = True

    context.generate_galois_keys()
    context.generate_relin_keys()
    return context


def plain_forward(x, fc1_weight, fc1_bias, fc2_weight, fc2_bias, poly_coeffs):
    """
    明文前向传播
    """
    out1 = fc1_weight.dot(x) + fc1_bias

    # 计算多项式激活
    act = np.zeros_like(out1) + poly_coeffs[0]
    for i in range(1, len(poly_coeffs)):
        if poly_coeffs[i] != 0:
            act += poly_coeffs[i] * (out1 ** i)

    out2 = fc2_weight.dot(act) + fc2_bias
    return out2


def fhe_forward(ctx, x, fc1_weight, fc1_bias, fc2_weight, fc2_bias, poly_coeffs):
    """
    密文前向传播
    """
    enc_x = ts.ckks_vector(ctx, x)

    # FC1：密文向量 × 明文矩阵
    enc_out1 = enc_x.mm(fc1_weight.T) + fc1_bias

    # 多项式激活
    enc_act1 = enc_out1.polyval(poly_coeffs)

    # FC2：密文向量 × 明文矩阵
    enc_out2 = enc_act1.mm(fc2_weight.T) + fc2_bias

    return enc_out2


def benchmark_inference():
    # -----------------------------
    # 1. 模拟网络结构
    # -----------------------------
    input_dim = 144
    hidden_dim = 64
    output_dim = 10

    np.random.seed(42)

    sample_feature = np.random.randn(input_dim)
    fc1_weight = np.random.randn(hidden_dim, input_dim)
    fc1_bias = np.random.randn(hidden_dim)
    fc2_weight = np.random.randn(output_dim, hidden_dim)
    fc2_bias = np.random.randn(output_dim)

    # 多项式系数
    poly3_coeffs = [0.5, 0.21232527, 0.0, -0.00497681]
    poly5_coeffs = [0.5, 0.23844647, 0.0, -0.01134516, 0.0, 0.00023539]

    # -----------------------------
    # 2. 明文基准测速
    # -----------------------------
    plain_repeat = 200

    start_time = time.time()
    for _ in range(plain_repeat):
        pt_final_3 = plain_forward(
            sample_feature, fc1_weight, fc1_bias, fc2_weight, fc2_bias, poly3_coeffs
        )
    pt_time_3 = (time.time() - start_time) / plain_repeat

    start_time = time.time()
    for _ in range(plain_repeat):
        pt_final_5 = plain_forward(
            sample_feature, fc1_weight, fc1_bias, fc2_weight, fc2_bias, poly5_coeffs
        )
    pt_time_5 = (time.time() - start_time) / plain_repeat

    print(f"\n[测试完成] 明文 Poly-3 单次推理平均耗时: {pt_time_3:.6f} 秒")
    print(f"[测试完成] 明文 Poly-5 单次推理平均耗时: {pt_time_5:.6f} 秒")

    # -----------------------------
    # 3. Poly-3 密文推理
    #    深度需求约为：FC1(1) + x^3(2) + FC2(1) = 4层
    # -----------------------------
    print("\n[3] 开始执行 FHE Poly-3 密文推理...")

    ctx3 = setup_fhe_context(
        poly_degree=16384,
        coeff_bits=[40, 26, 26, 26, 26, 40],   # 4个中间层，适配 Poly-3
        scale_bits=26
    )

    fhe_repeat = 5
    fhe_times_3 = []

    enc_final_3 = None
    for _ in range(fhe_repeat):
        start_time = time.time()
        enc_final_3 = fhe_forward(
            ctx3, sample_feature, fc1_weight, fc1_bias, fc2_weight, fc2_bias, poly3_coeffs
        )
        fhe_times_3.append(time.time() - start_time)

    fhe_poly3_time = np.mean(fhe_times_3)
    dec_final_3 = np.array(enc_final_3.decrypt())
    err3 = np.max(np.abs(dec_final_3 - pt_final_3))

    print(f"[测试完成] 密文 Poly-3 单次推理平均耗时: {fhe_poly3_time:.4f} 秒")
    print(f"  -> 相对明文耗时膨胀倍数: {fhe_poly3_time / pt_time_3:.0f} 倍")
    print(f"  -> 与明文输出最大绝对误差: {err3:.6e}")

    # -----------------------------
    # 4. Poly-5 密文推理
    #    深度需求约为：FC1(1) + x^5(3) + FC2(1) = 5层
    # -----------------------------
    print("\n[4] 开始执行 FHE Poly-5 密文推理...")

    ctx5 = setup_fhe_context(
        poly_degree=16384,
        coeff_bits=[40, 26, 26, 26, 26, 26, 40],  # 5个中间层，适配 Poly-5
        scale_bits=26
    )

    fhe_times_5 = []
    enc_final_5 = None
    for _ in range(fhe_repeat):
        start_time = time.time()
        enc_final_5 = fhe_forward(
            ctx5, sample_feature, fc1_weight, fc1_bias, fc2_weight, fc2_bias, poly5_coeffs
        )
        fhe_times_5.append(time.time() - start_time)

    fhe_poly5_time = np.mean(fhe_times_5)
    dec_final_5 = np.array(enc_final_5.decrypt())
    err5 = np.max(np.abs(dec_final_5 - pt_final_5))

    print(f"[测试完成] 密文 Poly-5 单次推理平均耗时: {fhe_poly5_time:.4f} 秒")
    print(f"  -> 相对明文耗时膨胀倍数: {fhe_poly5_time / pt_time_5:.0f} 倍")
    print(f"  -> 与明文输出最大绝对误差: {err5:.6e}")


if __name__ == "__main__":
    benchmark_inference()