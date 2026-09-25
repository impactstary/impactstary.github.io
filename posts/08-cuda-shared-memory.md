# CUDA Shared Memory：用 Tiling 让矩阵乘法快 10 倍

GPU 的计算能力远超 CPU，但有一个前提：你得喂得饱它。矩阵乘法是 AI 计算中最核心的算子，也是最考验内存访问效率的。如果直接从全局内存（Global Memory）读写数据，GPU 的计算单元大部分时间都在等数据，性能上不去。

**Shared Memory** 是 GPU 性能优化的第一武器。通过把数据从全局内存搬到片上共享内存，配合 **Tiling（分块）** 技术，可以让数据被反复复用，大幅减少全局内存访问次数。

这篇文章从朴素的矩阵乘法出发，一步步带你理解 Shared Memory Tiling 的原理和实现。

## 一、为什么 Shared Memory 如此重要

### 1.1 GPU 内存层级

GPU 的内存是分层的，离计算单元越近，速度越快但容量越小：

```
┌─────────────────────────────────────────────────┐
│              Registers (寄存器)                 │  最快，每个 thread 私有
│            ~256 KB per SM, ~1 cycle             │
├─────────────────────────────────────────────────┤
│            Shared Memory (共享内存)              │  快，同一个 block 内共享
│          ~48-164 KB per SM, ~5 cycles           │
├─────────────────────────────────────────────────┤
│             L2 Cache (二级缓存)                 │  中速，所有 SM 共享
│             ~40-80 MB, ~50 cycles               │
├─────────────────────────────────────────────────┤
│           Global Memory (全局内存)              │  最慢，所有 SM 共享
│           几十 GB, ~300-500 cycles              │
└─────────────────────────────────────────────────┘
```

全局内存的延迟是 Shared Memory 的 **50-100 倍**。如果算法频繁访问全局内存，计算单元大部分时间都在等待数据。

### 1.2 计算密集 vs 内存密集

矩阵乘法 C = A × B 的计算量和内存访问量：
- 计算量：2 × M × N × K 次浮点运算
- 内存访问量：M × K + K × N + M × N 次读 + M × N 次写

计算强度（Arithmetic Intensity）= 计算量 / 内存访问量 ≈ **K**

当 K 很大时，计算强度高，是计算密集型；但如果每个元素都直接从全局内存读，实际上每个元素只读一次，计算强度其实很低。

Shared Memory 的作用就是：**让数据被多次复用，提高计算强度**。

## 二、Shared Memory 基础

### 2.1 什么是 Shared Memory

Shared Memory 是 **每个 Thread Block 私有的片上内存**，同一个 block 内的所有 thread 都可以访问它。

特点：
- 速度快（接近寄存器速度）
- 容量小（每 SM 几十 KB）
- 生命周期与 block 相同
- 需要程序员手动管理（不像 L2 Cache 是硬件自动的）

### 2.2 Bank Conflicts（Bank 冲突）

Shared Memory 被分成 32 个 bank（对应 warp 的 32 个 thread），每个 bank 一次只能服务一个访问。如果同一个 warp 中有多个 thread 访问同一个 bank，就会发生 **bank conflict**，访问被串行化。

```
无冲突（每个 thread 访问不同 bank）：
Thread:  0   1   2   3  ...  31
Bank:    0   1   2   3  ...  31
         ↓   ↓   ↓   ↓       ↓
        ───────────────────────  1 次访问完成

有冲突（两个 thread 访问同一个 bank）：
Thread:  0   1   2   3  ...  31
Bank:    0   0   2   3  ...  31
         ↓   ↓   ↓   ↓       ↓
        ─────                  第 1 次（bank 0 串行）
        ─────────────────────  第 2 次
        → 2 次访问才能完成
```

**避免 bank conflict 的方法**：
- 按行访问时，保证同一 warp 的 thread 访问连续的地址
- 或者在每行末尾加 padding（填充），错开 bank 映射

## 三、朴素矩阵乘法及其问题

### 3.1 朴素实现

```cpp
// 朴素矩阵乘法：C = A * B
// A: M x K, B: K x N, C: M x N
__global__ void matmul_naive(
    float* A, float* B, float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            // 每个 k 都要从全局内存读 A 和 B 的一个元素
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### 3.2 问题分析

对于每个输出元素 C[row][col]：
- 需要读 K 个 A 的元素 + K 个 B 的元素
- 全部从全局内存读取
- 每个 A 的元素被 N 个 thread 读（每一列）
- 每个 B 的元素被 M 个 thread 读（每一行）

总共的全局内存读取次数 = M × N × K × 2 次

```
内存访问模式：

A[row][k] —— row 相同的 thread 读同一行（连续，好）
B[k][col] —— col 相同的 thread 读不同行（不连续，差）

而且数据完全不复用，每次都从全局内存读！
```

### 3.3 性能瓶颈

- **全局内存带宽是瓶颈**：大量时间花在等数据上
- **计算单元利用率低**：GPU 的 ALU 大部分时间闲置
- **B 的访问不连续**：非合并访问（non-coalesced）进一步降低效率

## 四、Tiling 技术：用 Shared Memory 做数据复用

### 4.1 核心思想

Tiling（分块）的思路是：把输出矩阵分成小块（tile），每个 block 负责计算一个 tile。计算这个 tile 需要的数据，先从全局内存搬到 Shared Memory，然后在 Shared Memory 中反复复用。

```
Tiling 原理示意图：

          B 矩阵 (K x N)
          ┌───────────────────────┐
          │  B_tile (TILE x TILE) │  ← 第 k 个 B 的 tile
          └───────────────────────┘
               ↑
A 矩阵 (M x K) │
┌───────────┐  │
│ A_tile    │──┘  ← 第 k 个 A 的 tile
│ (TILE x   │
│  TILE)    │     C_tile = A_tile × B_tile 的累加
└──────────┘
               ↓
          ┌───────────────────────┐
          │  C_tile (TILE x TILE) │  ← 正在计算的 C 的 tile
          └───────────────────────┘
               C 矩阵 (M x N)

每一步：
  1. 把 A 的一个 tile 和 B 的一个 tile 从全局内存搬到 Shared Memory
  2. 用 Shared Memory 中的数据计算部分结果
  3. 移动到下一个 tile，重复 K/TILE 次
  4. 得到最终的 C_tile
```

### 4.2 数据复用分析

假设 TILE_SIZE = 32：

- 朴素实现：每个 C 的 tile 需要 K × 32 + K × 32 = 2K × 32 次全局内存读
- Tiling 实现：每步读 32×32 + 32×32 = 2×1024 次，共 K/32 步
  - 总计：2 × 1024 × (K/32) = 64K 次全局内存读
- **减少比例**：64K / (64K) = 1/32（全局内存访问减少 32 倍！）

这就是 Tiling 的威力：**用 Shared Memory 做数据缓存，让每个数据被复用 TILE_SIZE 次**。

## 五、Tiled Matrix Multiplication 完整实现

### 5.1 完整代码

```cpp
#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 32  // tile 的大小，也是每个 block 的 thread 数（32x32=1024）

/**
 * Tiled 矩阵乘法：C = A * B
 * A: M x K (行优先存储)
 * B: K x N (行优先存储)
 * C: M x N (行优先存储)
 *
 * Grid 维度：(ceil(N/TILE_SIZE), ceil(M/TILE_SIZE))
 * Block 维度：(TILE_SIZE, TILE_SIZE)
 */
__global__ void matmul_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Shared Memory：存储 A 的一个 tile 和 B 的一个 tile
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // 当前 thread 负责的输出元素坐标
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 遍历所有 tile，逐步累加
    for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; ++k) {
        // ============================================================
        // 第一步：从全局内存加载 tile 到 Shared Memory
        // 每个 thread 负责加载 A 和 B 各一个元素
        // ============================================================

        // 加载 A 的 tile: A[blockRow*TILE : (blockRow+1)*TILE][k*TILE : (k+1)*TILE]
        int a_row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int a_col = k * TILE_SIZE + threadIdx.x;
        if (a_row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[a_row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;  // 边界外填 0
        }

        // 加载 B 的 tile: B[k*TILE : (k+1)*TILE][blockCol*TILE : (blockCol+1)*TILE]
        int b_row = k * TILE_SIZE + threadIdx.y;
        int b_col = blockIdx.x * TILE_SIZE + threadIdx.x;
        if (b_row < K && b_col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;  // 边界外填 0
        }

        // ============================================================
        // 同步：确保所有 thread 都加载完数据
        // （不然计算时可能数据还没搬完）
        // ============================================================
        __syncthreads();

        // ============================================================
        // 第二步：用 Shared Memory 中的数据计算点积
        // 每个元素被 TILE_SIZE 个 thread 复用
        // ============================================================
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        // ============================================================
        // 再次同步：确保所有 thread 用完 Shared Memory 中的数据
        // （不然下一轮加载可能覆盖还没读完的数据）
        // ============================================================
        __syncthreads();
    }

    // 写回结果到全局内存
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 5.2 关键细节解读

#### __syncthreads() 为什么必须有？

Shared Memory 是 block 内所有 thread 共享的。如果不同步，可能出现：

- Thread A 还在加载数据，Thread B 已经开始计算 → 读到旧数据
- Thread A 开始加载下一轮数据，Thread B 还在用当前数据 → 数据被覆盖

`__syncthreads()` 是 **block 内的栅栏同步**：所有 thread 都到达这里之后，才继续往下执行。

#### #pragma unroll 的作用

告诉编译器展开循环，减少循环控制开销。对于小循环（TILE_SIZE = 32），展开后性能更好。

#### __restrict__ 关键字

告诉编译器指针之间没有重叠（no alias），编译器可以做更多优化。

### 5.3 调用方式

```cpp
int main() {
    int M = 1024, N = 1024, K = 1024;

    // 分配和初始化 host 内存...
    float *h_A, *h_B, *h_C;
    // ...

    // 分配 device 内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // 拷贝数据到 device...

    // 配置 Grid/Block
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE
    );

    // 启动 kernel
    matmul_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // 拷贝结果回 host...

    return 0;
}
```

## 六、性能对比

### 6.1 理论性能对比表

| 指标 | 朴素实现 | Tiled 实现 | 提升倍数 |
|------|---------|-----------|---------|
| 全局内存读次数 | 2 × M × N × K | 2 × M × N × K / TILE_SIZE | TILE_SIZE 倍 |
| Shared Memory 访问 | 0 | 2 × M × N × K | — |
| 计算量 | 2 × M × N × K | 2 × M × N × K | 相同 |
| 计算强度（FLOP/Byte）| ~4 | ~4 × TILE_SIZE | TILE_SIZE 倍 |

> TILE_SIZE = 32 时，理论上全局内存访问减少 32 倍，计算强度提升 32 倍。

### 6.2 实际性能参考（A100 上的大致数据）

| 实现方式 | TFLOPS | 占峰值比例 | 相对加速 |
|---------|--------|-----------|---------|
| CPU (单线程) | ~0.001 | — | 1x |
| CUDA 朴素 | ~0.5 | ~1% | 500x |
| CUDA Tiled | ~5-8 | ~10-15% | 5000-8000x |
| cuBLAS (优化版) | ~30+ | ~60%+ | 30000x+ |

> 注：实际性能取决于 GPU 型号、矩阵大小、数据类型等。这里只是数量级参考。

可以看到，Tiled 版本比朴素版本快大约 **10 倍**，但离 cuBLAS 等高度优化的库还有不小差距。

## 七、进一步优化方向

Tiled matrix multiplication 只是入门，真正的高性能矩阵乘法还有很多优化手段：

### 7.1 进一步优化清单

| 优化技术 | 原理 | 预期收益 |
|---------|------|---------|
| **更大的 tile** | 用 64×64 或更大的 tile，提高复用率 | 10-20% |
| **寄存器分块** | 每个 thread 计算多个输出元素，寄存器内累加 | 2-3x |
| **Warp-level 优化** | 利用 warp shuffle 指令通信 | 20-30% |
| **Double Buffering** | 预取下一个 tile，计算和访存重叠 | 20-40% |
| **Bank Conflict 消除** | 调整 Shared Memory 布局，避免冲突 | 10-20% |
| **FP16/Tensor Core** | 使用 Tensor Core 做混合精度矩阵乘 | 4-8x |
| **预取（Prefetch）** | 提前加载下一块数据到 L2 | 10-20% |

### 7.2 真实世界的矩阵乘法

cuBLAS、Tensor Core 等实际工业级实现的复杂度远超上面的示例代码。一个完整的 GEMM 实现可能包括：

- 根据矩阵大小选择不同的 kernel
- 多种 tile size 适配不同形状
- 多级流水线和预取
- 精细的寄存器分配
- Tensor Core 的 WMMA / MMA 指令
- 自动调优（auto-tuning）寻找最优配置

但所有这些优化的**起点和基础**，都是 Shared Memory Tiling 这个核心思想。

## 八、在 AI 框架中的体现

理解了 Tiling，再去看 AI 框架中的算子就会发现同样的模式：

- **FlashAttention**：把 Q、K、V 分块加载到 Shared Memory 中计算，本质上就是 attention 版本的 Tiling
- **Conv2D 优化**：Im2Col + GEMM，最终落到矩阵乘法上
- **Triton 算子**：Triton 语言中的 `tl.load` + `tl.dot` 背后，编译器自动做了 Tiling

Shared Memory Tiling 是 GPU 性能优化的"Hello World"，掌握它是深入理解所有高性能 GPU 算子的第一步。

## 九、总结

Shared Memory Tiling 的核心可以用一句话概括：

> **把数据从慢的全局内存搬到快的共享内存，让数据被多次复用，用计算换带宽。**

关键要点回顾：

1. **内存层级**：Global Memory 慢但大，Shared Memory 快但小
2. **Tiling 思想**：把计算分成小块，小块内数据放 Shared Memory 复用
3. **同步机制**：`__syncthreads()` 确保数据加载完成再计算
4. **性能收益**：全局内存访问量减少 TILE_SIZE 倍，通常 5-10 倍加速
5. **Bank Conflict**：合理安排访问模式，避免 Shared Memory 冲突

矩阵乘法是 AI 系统的基石算子，而 Shared Memory Tiling 是矩阵乘法优化的第一课。理解了它，你就掌握了打开 GPU 性能优化大门的钥匙。

---

**延伸阅读**：
- *CUDA C++ Programming Guide* — NVIDIA
- *Programming Massively Parallel Processors* — Kirk & Hwu
- [CUDA Matrix Multiplication Example](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory)
