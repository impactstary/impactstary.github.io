---
title: 用 Triton 手写 FlashAttention：从原理到实现
date: 2024-03-12
tags: [Triton, FlashAttention, GPU优化, 大模型]
description: FlashAttention 是大模型推理的核心技术之一。这篇文章从标准 Attention 的内存瓶颈出发，讲清 FlashAttention 的核心思想——分块计算 + Online Softmax，最后用 Triton 从零实现一个完整的 FlashAttention kernel。
---

# 用 Triton 手写 FlashAttention：从原理到实现

如果说有哪项技术对大模型时代影响最大，FlashAttention 绝对排得上号。

在 FlashAttention 出现之前，Attention 操作的显存占用是 O(N²) 的——序列长一点，显存就炸了。FlashAttention 把这个复杂度降到了 O(N)，同时速度还更快，简直是「既要又要还要」的典范。

这篇文章，我们从原理讲到实现，最后用 Triton 写出一个完整可用的 FlashAttention kernel。

## 标准 Attention 的内存瓶颈

先回顾一下标准的 Scaled Dot-Product Attention：

```
Attention(Q, K, V) = softmax(QK^T / √d) * V
```

其中 Q、K、V 的形状都是 `[N, d]`（N 是序列长度，d 是 head dim）。

### 标准实现的三步法

```python
import torch

def attention_naive(Q, K, V):
    # 第一步：计算 S = Q @ K^T  → 形状 [N, N]
    S = Q @ K.T / (d ** 0.5)
    # 第二步：计算 P = softmax(S) → 形状 [N, N]
    P = torch.softmax(S, dim=-1)
    # 第三步：计算 O = P @ V      → 形状 [N, d]
    O = P @ V
    return O
```

问题出在哪？**中间结果 S 和 P 都是 N×N 的矩阵**。

举个例子：当 N = 4096，d = 128 时：

| 矩阵 | 形状 | 大小（FP16） |
|------|------|-------------|
| Q, K, V | [4096, 128] | 每个 1MB |
| S (QK^T) | [4096, 4096] | 32MB |
| P (softmax) | [4096, 4096] | 32MB |

N=4096 就用了 64MB 存中间结果。如果 N=16K 呢？就是 1GB。N=64K 呢？16GB。**序列长度一长，中间矩阵的显存占用是平方级增长的。**

更糟糕的是，这些中间矩阵还需要在 HBM（显存）和 SRAM（SM 内高速缓存）之间来回搬运，非常耗时。

> GPU 的 HBM 虽然容量大，但访问延迟高（几百个 cycle）。SRAM 速度快但容量小（每个 SM 只有几百 KB）。**显存带宽往往是算子性能的瓶颈。**

## FlashAttention 的核心思想

FlashAttention 的目标很明确：**不要把完整的 N×N 矩阵存到 HBM 里**。

怎么做到？两个关键技术：

### 1. 分块计算（Tiling）

把 Q、K、V 都切成小块，每次只加载一小块到 SRAM 里计算，算完就把结果写回去。这样中间结果始终在 SRAM 里，不需要写回 HBM。

```
K 被切成 K1, K2, K3, ..., Kb  （每块 Br 行）
V 被切成 V1, V2, V3, ..., Vb
Q 被切成 Q1, Q2, Q3, ..., Qb   （每块 Bc 列）

每次加载 Qi 和 Kj，计算一小块 S_ij = Qi @ Kj^T
然后在 SRAM 里做 softmax，再乘 Vj
累加结果到 Oi
```

用一张图来理解：

```
  Q (N×d)                    K^T (d×N)
┌─────────┐               ┌──────────────────┐
│ Q1      │  ───┐         │ K1^T  K2^T  K3^T │   ← K 被分块
│         │     │         └──────────────────┘
│ Q2      │     │
│         │     │  每次取一块 Qi 和一块 Kj
│ Q3      │     │  计算 S_ij = Qi @ Kj^T (Br × Bc)
│         │     │  在 SRAM 里完成 softmax 和乘 V
└─────────┘  ───┘  结果累加回 Oi
     ↑
  Q 被分块
```

但是，这里有个问题：**softmax 不是可以简单分块的运算**。

标准 softmax 是这样的：

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

分母是所有元素的指数和。如果你只看一块数据，你不知道全局的和是多少，也就没法算正确的 softmax 值。

这就引出了 FlashAttention 的第二个关键技术：**Online Softmax**。

### 2. Online Softmax

Online Softmax 是一种可以「流式」计算 softmax 的方法——你不需要一次看到所有数据，可以一块一块地算，而且结果和一次性算出来的完全一样。

#### 数学原理

假设我们要计算向量 x 的 softmax。定义两个统计量：

- `m = max(x)` ：最大值（用于数值稳定）
- `l = Σ exp(x_i - m)` ：指数和

softmax 的结果就是 `exp(x_i - m) / l`。

现在假设 x 被分成了两块：x = [x₁, x₂]。

先算第一块，得到：
- m₁ = max(x₁)
- l₁ = Σ exp(x₁_i - m₁)

再算第二块，得到：
- m₂ = max(x₂)
- l₂ = Σ exp(x₂_i - m₂)

合并两块的统计量：
- m_new = max(m₁, m₂)
- l_new = l₁ * exp(m₁ - m_new) + l₂ * exp(m₂ - m_new)

为什么这样是对的？因为：

```
l = Σ exp(x_i - m_new)
  = Σ_{i in 1} exp(x_i - m_new) + Σ_{i in 2} exp(x_i - m_new)
  = exp(m₁ - m_new) * Σ_{i in 1} exp(x_i - m₁) + exp(m₂ - m_new) * Σ_{i in 2} exp(x_i - m₂)
  = exp(m₁ - m_new) * l₁ + exp(m₂ - m_new) * l₂
```

完美！这样我们就可以一块一块地处理数据，只需要维护 `m`（当前最大值）和 `l`（当前指数和）两个标量就行。

#### 对输出也做增量更新

光有 softmax 的统计量还不够，我们还需要维护输出 O 的增量更新。

每处理一块 Kj、Vj，我们计算：

```
S_ij = Qi @ Kj^T / √d    # 当前块的 attention score
m_ij = rowmax(S_ij)      # 当前块每行的最大值
P_ij = exp(S_ij - m_ij)  # 当前块的指数值
l_ij = rowsum(P_ij)      # 当前块每行的指数和
```

然后更新全局统计量和输出：

```
m_new = max(m_old, m_ij)
l_new = l_old * exp(m_old - m_new) + l_ij * exp(m_ij - m_new)
O_new = O_old * (l_old / l_new) * exp(m_old - m_new) + (P_ij @ Vj) * exp(m_ij - m_new) / l_new
```

O 的更新公式看起来复杂，本质上就是：**旧的输出乘以一个缩放因子，加上新块的贡献**。缩放因子是为了保证最终 softmax 的归一化是正确的。

## 用 Triton 实现 FlashAttention

好了，原理讲完了，现在来写代码。

> 如果你不熟悉 Triton，可以把它理解为一个「Python 风格的 CUDA 编程框架」。你用 Python 写 kernel，Triton 编译器自动生成高效的 CUDA 代码。它的优势是不用手动管寄存器分配、共享内存布局这些底层细节，但性能往往能接近手写 CUDA。

### 完整实现

```python
import torch
import triton
import triton.language as tl

# ============================================================
# FlashAttention Forward Kernel
# ============================================================
@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qn, stride_qd,      # Q 的行/列步长
    stride_kn, stride_kd,      # K 的行/列步长
    stride_vn, stride_vd,      # V 的行/列步长
    stride_on, stride_od,      # O 的行/列步长
    N, D,                      # 序列长度，head dim
    BLOCK_M: tl.constexpr,     # Q 方向分块大小
    BLOCK_N: tl.constexpr,     # K/V 方向分块大小
):
    # ---------- 1. 计算当前 block 负责哪一块 Q ----------
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_d = tl.arange(0, D)                            # [D]

    # 构造 Q 块的指针（按行存储）
    q_ptrs = Q_ptr + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd

    # 加载 Q 块到 SRAM（只需加载一次）
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)  # [BLOCK_M, D]

    # ---------- 2. 初始化输出和统计量 ----------
    # m: 当前行的最大值 (BLOCK_M,)
    # l: 当前行的指数和 (BLOCK_M,)
    # o: 当前行的输出累加值 (BLOCK_M, D)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    # ---------- 3. 遍历 K/V 的所有块 ----------
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]

        # 加载 K 块
        k_ptrs = K_ptr + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=offs_n[None, :] < N, other=0.0)  # [D, BLOCK_N]

        # 加载 V 块
        v_ptrs = V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)  # [BLOCK_N, D]

        # ---------- 3.1 计算 S = Q @ K^T / sqrt(d) ----------
        # q: [BLOCK_M, D], k: [D, BLOCK_N]
        # s: [BLOCK_M, BLOCK_N]
        s = tl.dot(q, k) / tl.sqrt(tl.cast(D, tl.float32))

        # padding 位置设为 -inf，不参与 softmax
        s = tl.where(offs_n[None, :] < N, s, float('-inf'))

        # ---------- 3.2 Online Softmax 更新 ----------
        # 计算当前块的行最大值
        m_ij = tl.max(s, axis=1)  # [BLOCK_M]

        # 计算 exp(s - m_ij)
        p_ij = tl.exp(s - m_ij[:, None])  # [BLOCK_M, BLOCK_N]

        # 计算当前块的行和
        l_ij = tl.sum(p_ij, axis=1)  # [BLOCK_M]

        # 更新全局统计量
        m_new = tl.maximum(m_i, m_ij)
        l_new = l_i * tl.exp(m_i - m_new) + l_ij * tl.exp(m_ij - m_new)

        # ---------- 3.3 更新输出 ----------
        # 旧输出的缩放因子
        alpha = tl.exp(m_i - m_new)  # [BLOCK_M]

        # 新块的贡献: P_ij @ Vj
        # p_ij: [BLOCK_M, BLOCK_N], v: [BLOCK_N, D]
        # pv: [BLOCK_M, D]
        pv = tl.dot(p_ij, v)

        # 增量更新输出
        o_i = o_i * alpha[:, None] + pv * tl.exp(m_ij - m_new)[:, None]

        # 更新统计量供下一轮使用
        m_i = m_new
        l_i = l_new

    # ---------- 4. 最终归一化，写出结果 ----------
    o_i = o_i / l_i[:, None]

    # 写出 O
    o_ptrs = O_ptr + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o_i, mask=offs_m[:, None] < N)


# ============================================================
# Python 包装函数
# ============================================================
def flash_attn_forward(Q, K, V):
    """
    FlashAttention 前向传播
    Q, K, V: [N, D]，单 head
    返回: [N, D]
    """
    N, D = Q.shape
    O = torch.empty_like(Q)

    # 分块大小选择：
    # BLOCK_M 大一点，提高计算密度
    # BLOCK_N 受限于 SRAM 大小
    BLOCK_M = 128
    BLOCK_N = 64

    # grid 配置：每个 block 处理 BLOCK_M 行 Q
    grid = (triton.cdiv(N, BLOCK_M),)

    flash_attn_fwd_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        O.stride(0), O.stride(1),
        N, D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return O
```

### 代码逐段解析

让我们把 kernel 的核心逻辑拆开来看。

#### 分块策略

```
Q 方向（行方向）：BLOCK_M = 128
K 方向（列方向）：BLOCK_N = 64
```

每个 thread block 处理 128 行 Q，然后遍历所有 K 块（每块 64 列）。

为什么是这个比例？因为我们需要把 Q 块、K 块、V 块、中间的 S 和 P 都放进 SRAM。BLOCK_N 越小，占用的 SRAM 越少，但循环次数越多。这是一个经典的空间-时间权衡。

#### 加载 Q 块

```python
q = tl.load(q_ptrs, mask=offs_m[:, None] < N, other=0.0)
```

Q 块在循环外只加载一次——因为同一个 Q 块要和所有 K 块做计算，放在 SRAM 里复用。

#### 核心循环

```python
for start_n in range(0, N, BLOCK_N):
    # 加载一块 K 和一块 V
    # 计算 S_ij = Q_i @ K_j^T
    # 更新 m, l, o
```

每次迭代加载一小块 K 和 V，在 SRAM 里完成所有计算，然后只更新输出和统计量。**中间的 S 和 P 都不会写回 HBM**。

#### Online Softmax 更新

```python
m_ij = tl.max(s, axis=1)
p_ij = tl.exp(s - m_ij[:, None])
l_ij = tl.sum(p_ij, axis=1)

m_new = tl.maximum(m_i, m_ij)
l_new = l_i * tl.exp(m_i - m_new) + l_ij * tl.exp(m_ij - m_new)
```

这就是前面推导的 Online Softmax 公式的直接翻译。`m` 和 `l` 都是长度为 BLOCK_M 的向量，每行一个值。

#### 输出更新

```python
alpha = tl.exp(m_i - m_new)
pv = tl.dot(p_ij, v)
o_i = o_i * alpha[:, None] + pv * tl.exp(m_ij - m_new)[:, None]
```

旧的输出乘以 `alpha`（因为全局最大值变了，之前的归一化需要修正），再加上新块的贡献。

#### 最终归一化

```python
o_i = o_i / l_i[:, None]
```

循环结束后，`o_i` 里存的是「加权和」状态，最后除以 `l_i` 得到正确的 softmax 加权平均。

### 验证正确性

写好了 kernel，第一步永远是验证对不对：

```python
def test_flash_attention():
    torch.manual_seed(42)
    N, D = 1024, 64

    Q = torch.randn(N, D, device='cuda', dtype=torch.float16)
    K = torch.randn(N, D, device='cuda', dtype=torch.float16)
    V = torch.randn(N, D, device='cuda', dtype=torch.float16)

    # 标准实现（作为基准）
    def attention_ref(Q, K, V):
        S = Q @ K.T / (D ** 0.5)
        P = torch.softmax(S, dim=-1)
        return P @ V

    O_ref = attention_ref(Q, K, V)
    O_flash = flash_attn_forward(Q, K, V)

    # 比较结果
    print(f"最大误差: {(O_ref - O_flash).abs().max().item():.6f}")
    print(f"相对误差: {(O_ref - O_flash).norm() / O_ref.norm():.6f}")

    # 误差在 FP16 的精度范围内就算正确
    assert torch.allclose(O_ref.float(), O_flash.float(), atol=1e-2, rtol=1e-2)
    print("✅ FlashAttention 结果正确！")

test_flash_attention()
```

如果一切正常，你应该能看到误差在 1e-3 量级——这是 FP16 计算的正常精度损失。

## 性能对比讨论

FlashAttention 到底快在哪里？我们从几个角度来分析。

### 显存占用

| 实现方式 | 显存复杂度 | N=4096, d=128 |
|----------|-----------|---------------|
| 标准 Attention | O(N²) | ~64MB 中间结果 |
| FlashAttention | O(N) | ~几 KB 中间结果（在 SRAM 中） |

FlashAttention 的额外显存占用几乎可以忽略——因为中间结果都在 SRAM 里，不需要写回 HBM。

### 显存带宽

标准 Attention 的 HBM 访问量：

```
读 Q, K, V: 3 * N * d
写 S: N * N
读 S: N * N
写 P: N * N
读 P: N * N
写 O: N * d
= 4N² + 4Nd  ≈ 4N²（当 N >> d 时）
```

FlashAttention 的 HBM 访问量：

```
读 Q: N * d      （Q 只加载一次，存在 SRAM）
读 K, V: 2 * N * d  （K 和 V 在循环中读，共 1 次完整遍历）
写 O: N * d
= 4Nd
```

当 N=4096, d=128 时：
- 标准：4 × 4096² + 4 × 4096 × 128 ≈ **69M 元素**
- Flash：4 × 4096 × 128 ≈ **2M 元素**

差了 30 多倍！这就是 FlashAttention 快的根本原因——**大幅减少了 HBM 访问**。

### 计算量

计算量方面两者是一样的，都是 O(N²d)。FlashAttention 没有减少计算量，它只是把计算的「排列方式」变了，让计算和内存访问更高效。

> 这也是为什么 FlashAttention 在计算密集型场景（d 很大、N 很小）提升不明显，而在内存密集型场景（N 很大）提升巨大。

## 进阶：还有什么可以优化？

上面的实现是一个最基础的 FlashAttention，和工业级的实现相比还有不少可以优化的地方：

### 1. 多 head 支持
实际使用中 Q/K/V 形状是 `[batch, num_heads, seq_len, head_dim]`，需要在 kernel 里处理 batch 和 head 维度。

### 2. Causal Mask
自回归模型需要因果掩码（每个 token 只能看到前面的 token），可以通过在循环中调整起始位置来减少不必要的计算。

### 3. 反向传播
FlashAttention 的反向传播同样用分块 + Online Softmax 的思路，但更复杂一些——需要在前向时保存 m 和 l 用于反向。

### 4. 分块大小调优
不同的 GPU 架构、不同的 d 值，最优的 BLOCK_M/BLOCK_N 组合都不一样。Triton 提供了 autotune 功能可以自动搜索最优配置。

### 5. Warp 级优化
利用 Tensor Core 的排布、bank conflict 避免、寄存器分配优化等。工业级 FlashAttention（如 FlashAttention v2/v3）在这些方面做了大量细致的优化。

## 总结

让我们用一张图回顾 FlashAttention 的核心思想：

```
  标准 Attention                     FlashAttention
───────────────────              ───────────────────
  HBM                                HBM
 ┌───────┐                         ┌───────┐
 │ Q K V │                         │ Q K V │
 └───┬───┘                         └───┬───┘
     │  读取 Q,K,V                       │  分块读取
     ▼                                   ▼
 ┌───────────┐         SRAM          ┌─────────┐
 │ QK^T (N²) │ ────────读写──────▶  │ Qi Kj   │  ← 小块数据
 └─────┬─────┘                       │ S_ij    │    全部在
       │                             │ P_ij    │    SRAM 中
       ▼                             │ ...     │    计算
 ┌───────────┐                       └────┬────┘
 │ softmax  │                            │  结果写回
 │   (N²)    │                            ▼
 └─────┬─────┘                       ┌───────┐
       │                             │ O     │
       ▼                             └───────┘
 ┌───────────┐
 │  P @ V    │
 └───────────┘
```

FlashAttention 的核心就是一句话：**用分块 + Online Softmax，把 O(N²) 的中间结果从 HBM 搬到 SRAM 里，大幅减少显存访问，从而获得巨大的性能提升。**

它之所以成为大模型时代的基石技术，不是因为发明了什么新的算法，而是因为它**从硬件特性出发，重新思考了 Attention 的计算方式**。这种「面向硬件的算法设计」思路，值得我们每一个做 AI 系统的人学习。

---

**参考资料：**
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [Triton 官方文档](https://triton-lang.org/)

> 如果你觉得这篇文章对你有帮助，欢迎在评论区留言讨论。下一篇我们会讲讲 FlashAttention 的反向传播是怎么实现的。
