# -*- coding: utf-8 -*-
"""无 torch 版逻辑测试"""
import numpy as np


def im2col_indices(in_h, in_w, kh, kw, stride):
    out_h = (in_h - kh) // stride + 1
    out_w = (in_w - kw) // stride + 1
    cols = []
    for i in range(out_h):
        for j in range(out_w):
            top = i * stride
            left = j * stride
            idxs = []
            for di in range(kh):
                for dj in range(kw):
                    idxs.append((top + di) * in_w + (left + dj))
            cols.append(idxs)
    return np.array(cols).T, out_h, out_w


def conv2d_mask_method(flat_input, conv_weight, conv_bias,
                       in_h, in_w, im2col_idx, out_h, out_w, num_kernels):
    plain_in_size = in_h * in_w
    num_windows = out_h * out_w
    output_channels = []
    for k in range(num_kernels):
        kernel = conv_weight[k, 0].reshape(-1)
        bias = float(conv_bias[k])
        ch = np.zeros(num_windows)
        for w_idx in range(num_windows):
            mask = np.zeros(plain_in_size)
            positions = im2col_idx[:, w_idx]
            for p, v in zip(positions, kernel):
                mask[p] = v
            ch[w_idx] = flat_input.dot(mask) + bias
        output_channels.append(ch)
    return output_channels


def conv2d_naive(image, w, b, stride=2):
    nk, _, kh, kw = w.shape
    ih, iw = image.shape
    oh = (ih - kh) // stride + 1
    ow = (iw - kw) // stride + 1
    out = np.zeros((nk, oh, ow))
    for k in range(nk):
        for i in range(oh):
            for j in range(ow):
                p = image[i*stride:i*stride+kh, j*stride:j*stride+kw]
                out[k, i, j] = np.sum(p * w[k, 0]) + b[k]
    return out


def avgpool_flat(c, h, w):
    oh, ow = h // 2, w // 2
    r = []
    for i in range(oh):
        for j in range(ow):
            i0, j0 = 2 * i, 2 * j
            r.append(0.25 * (c[i0*w+j0] + c[i0*w+j0+1]
                             + c[(i0+1)*w+j0] + c[(i0+1)*w+j0+1]))
    return np.array(r)


def avgpool_naive(f):
    h, w = f.shape
    oh, ow = h // 2, w // 2
    o = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            o[i, j] = f[2*i:2*i+2, 2*j:2*j+2].mean()
    return o


# ===== Test 1: Conv =====
print("[1] 卷积")
np.random.seed(0)
img = np.random.randn(28, 28)
w = np.random.randn(4, 1, 5, 5)
b = np.random.randn(4)

ref = conv2d_naive(img, w, b, 2)
idx, oh, ow = im2col_indices(28, 28, 5, 5, 2)
ours = conv2d_mask_method(img.reshape(-1), w, b, 28, 28, idx, oh, ow, 4)

md = 0
for k in range(4):
    d = float(np.max(np.abs(ours[k].reshape(oh, ow) - ref[k])))
    md = max(md, d)
print(f"   max diff = {md:.4e}")
assert md < 1e-10

# ===== Test 2: Pool =====
print("[2] 池化")
np.random.seed(1)
f = np.random.randn(12, 12)
r = avgpool_naive(f)
o = avgpool_flat(f.reshape(-1), 12, 12).reshape(6, 6)
d = float(np.max(np.abs(o - r)))
print(f"   max diff = {d:.4e}")
assert d < 1e-10

# ===== Test 3: Full =====
print("[3] 完整前向 Conv->Poly3->Pool->FC1->Poly3->FC2")
np.random.seed(42)
cw = np.random.randn(4, 1, 5, 5) * 0.3
cb = np.random.randn(4) * 0.3
f1w = np.random.randn(64, 144) * 0.1
f1b = np.random.randn(64) * 0.1
f2w = np.random.randn(10, 64) * 0.1
f2b = np.random.randn(10) * 0.1
img = np.random.randn(28, 28) * 0.5


def poly3(x):
    return 0.5 + 0.21232527 * x - 0.00497681 * x**3


# ref
c_ref = conv2d_naive(img, cw, cb, 2)
a1_ref = poly3(c_ref)
p_ref = np.stack([avgpool_naive(a1_ref[k]) for k in range(4)])
flat_ref = p_ref.reshape(-1)
h1_ref = f1w.dot(flat_ref) + f1b
a2_ref = poly3(h1_ref)
o_ref = f2w.dot(a2_ref) + f2b

# ours
idx, oh, ow = im2col_indices(28, 28, 5, 5, 2)
chs = conv2d_mask_method(img.reshape(-1), cw, cb, 28, 28, idx, oh, ow, 4)
a1 = [poly3(c) for c in chs]
p = [avgpool_flat(c, oh, ow) for c in a1]
flat = np.concatenate(p)
h1 = f1w.dot(flat) + f1b
a2 = poly3(h1)
o = f2w.dot(a2) + f2b

d = float(np.max(np.abs(o - o_ref)))
print(f"   max diff = {d:.4e}")
assert d < 1e-8, d

print("\n所有测试通过 ✓ tens_full.py 的算法逻辑正确")
