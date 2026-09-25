---
title: CUDA 入门：从 vector_add 理解 GPU 编程模型
date: 2024-02-05
tags: [CUDA, GPU编程, 高性能计算]
description: 每个 CUDA 初学者写的第一个 kernel 都是 vector_add。这篇文章从最简单的向量加法出发，把 CUDA 编程模型的核心概念一次性讲清楚。
---

# CUDA 入门：从 vector_add 理解 GPU 编程模型

如果你开始接触深度学习底层优化，CUDA 是绕不过去的一道坎。

刚开始学 CUDA 的时候，我最大的困惑是：**CPU 代码我会写，为什么到了 GPU 上一切都不一样了？** 什么是 kernel？什么是 grid、block、thread？为什么要有这么多层级？

这篇文章，我们从最简单的 `vector_add` 出发，把 CUDA 编程模型的核心概念讲透。

## CPU vs GPU：为什么我们需要 CUDA

先回答最根本的问题：为什么不直接用 CPU 算，非要搞一套 CUDA 出来？

答案在于 **CPU 和 GPU 的架构设计目标完全不同**。

| | CPU | GPU |
|---|-----|-----|
| 核心数 | 几个到几十个 | 几千个 |
| 缓存 | 大（侧重延迟优化） | 小（侧重吞吐优化） |
| 控制逻辑 | 复杂（支持分支预测、乱序执行） | 简单（大量简单核心并行） |
| 擅长 | 串行任务、复杂逻辑 | 并行计算、数据密集型任务 |
| 比喻 | 几个资深工程师 | 一大群实习生 |

举个具体的例子：假设要把两个长度为 1000 万的 float 数组相加。

- **CPU 做法**：一个 for 循环，依次加 1000 万次。即使开多核，加速比也有限。
- **GPU 做法**：启动 1000 万个线程，每个线程只算一次加法，瞬间完成。

> GPU 不是什么都快。它的优势在于**数据并行**——同样的操作，作用在大量数据上。如果你的任务逻辑复杂、分支多、数据量小，GPU 可能反而更慢。

CUDA（Compute Unified Device Architecture）就是 NVIDIA 推出的一套 GPU 编程框架，让我们可以用类 C 的语言写 GPU 程序。

## CUDA 编程模型的核心概念

在看代码之前，先把几个关键概念搞明白。

### Kernel：在 GPU 上运行的函数

**Kernel** 是 CUDA 程序的核心——它就是一个在 GPU 上并行执行的函数。

和普通函数不同的是：
- 用 `__global__` 修饰符声明
- 调用时需要指定**启动多少个线程**
- 所有线程执行**同一段代码**（SIMT 模型）

```cpp
__global__ void vector_add(float *a, float *b, float *c, int n) {
    // 这里面的代码会被大量线程同时执行
    // 每个线程处理自己的那一份数据
}
```

### 线程层级：Grid → Block → Thread

CUDA 的线程不是扁平的，而是有层级结构的：

```
┌─────────────────────────────────────────────────────────┐
│                      Grid (1D/2D/3D)                    │
│                                                         │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│   │ Block 0  │  │ Block 1  │  │ Block 2  │  ...      │
│   │          │  │          │  │          │           │
│   │ T0 T1 T2 │  │ T0 T1 T2 │  │ T0 T1 T2 │  ...      │
│   │ T3 T4 T5 │  │ T3 T4 T5 │  │ T3 T4 T5 │           │
│   │ T6 T7 T8 │  │ T6 T7 T8 │  │ T6 T7 T8 │           │
│   └──────────┘  └──────────┘  └──────────┘           │
│                                                         │
│      每个 Block 最多 1024 个线程（取决于硬件）          │
└─────────────────────────────────────────────────────────┘
```

- **Thread（线程）**：最小的执行单元，有自己的寄存器和本地内存
- **Block（线程块）**：一组线程的集合，同一个 Block 内的线程可以通过共享内存通信、同步
- **Grid（网格）**：所有 Block 的集合，也就是一次 kernel 启动的所有线程

为什么要分这么多层？两个原因：

1. **硬件限制**：GPU 的 Streaming Multiprocessor（SM）一次调度的是一个 Block 的线程。Block 不能太大（目前最多 1024 个线程）。
2. **编程模型**：Block 之间是完全独立的，你不能假设它们的执行顺序。这种设计让 GPU 可以灵活调度，也让程序天然具备可扩展性。

### 内置变量：我是谁？我在哪？

每个线程都可以通过内置变量知道自己的位置：

| 变量 | 类型 | 含义 |
|------|------|------|
| `threadIdx.x` | uint3 | 线程在 Block 内的索引 |
| `blockIdx.x` | uint3 | Block 在 Grid 内的索引 |
| `blockDim.x` | uint3 | 每个 Block 的大小（线程数） |
| `gridDim.x` | uint3 | Grid 的大小（Block 数） |

`.x` 是因为这些变量都是 3 维的（`x`、`y`、`z`），方便处理图像、矩阵等 2D/3D 数据。

### 计算全局索引

最常用的操作——算出当前线程对应的数据下标：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

这个公式非常重要。理解了它，你就理解了 CUDA 线程模型的一半。

## 完整的 vector_add 代码

好了，概念讲完了，来看完整的代码。

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ============================================================
// Kernel 函数：向量加法
// ============================================================
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    // 计算当前线程对应的全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 防止越界：当数组长度不是线程数的整数倍时
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ============================================================
// 辅助函数：检查 CUDA 错误
// ============================================================
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA 错误: %s (行 %d)\n", \
                cudaGetErrorString(err), __LINE__); \
        exit(1); \
    }

// ============================================================
// 主函数
// ============================================================
int main() {
    int n = 1 << 20;  // 100 万元素
    size_t size = n * sizeof(float);

    // ---------- 1. 在 Host（CPU）上分配内存并初始化 ----------
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    for (int i = 0; i < n; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // ---------- 2. 在 Device（GPU）上分配内存 ----------
    float *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, size));

    // ---------- 3. 把数据从 Host 拷贝到 Device ----------
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // ---------- 4. 启动 Kernel ----------
    int block_size = 256;                    // 每个 block 256 个线程
    int grid_size = (n + block_size - 1) / block_size;  // 向上取整

    vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);

    // 检查 kernel 启动是否出错（kernel 本身不返回错误码）
    CHECK_CUDA_ERROR(cudaGetLastError());

    // 同步等待 kernel 执行完成
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // ---------- 5. 把结果从 Device 拷贝回 Host ----------
    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // ---------- 6. 验证结果 ----------
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
            errors++;
        }
    }
    printf("元素总数: %d\n", n);
    printf("错误数量: %d\n", errors);
    printf("结果: %s\n", errors == 0 ? "✅ 正确" : "❌ 错误");

    // ---------- 7. 释放内存 ----------
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
```

## 逐段讲解

让我们把代码拆开，一段一段讲。

### Kernel 函数

```cpp
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

- `__global__`：告诉编译器这是一个 kernel，在 GPU 上执行，从 CPU 端调用。
- `idx = blockIdx.x * blockDim.x + threadIdx.x`：经典的全局索引计算公式。
- `if (idx < n)`：边界检查。因为 `grid_size` 是向上取整的，最后一个 block 里可能有些线程是「多余」的，需要跳过。

### Host 端内存分配

```cpp
float *h_a = (float *)malloc(size);
```

`h_` 前缀是一种命名约定，表示 host 内存。对应的，`d_` 表示 device 内存。

### Device 端内存分配

```cpp
cudaMalloc(&d_a, size);
```

GPU 有自己的显存，不能直接用 `malloc`。`cudaMalloc` 在显存上分配空间，返回的指针指向显存地址。

> 注意：你不能在 CPU 端直接解引用 device 指针，那样会引发段错误。

### 数据拷贝

```cpp
cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
```

`cudaMemcpy` 是 CPU 和 GPU 之间搬运数据的主力函数。最后一个参数指定方向：
- `cudaMemcpyHostToDevice`：CPU → GPU
- `cudaMemcpyDeviceToHost`：GPU → CPU
- `cudaMemcpyDeviceToDevice`：GPU → GPU

### 启动 Kernel

```cpp
vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
```

`<<<M, N>>>` 是 CUDA 的语法扩展，表示启动 M 个 block，每个 block 有 N 个线程。

`grid_size = (n + block_size - 1) / block_size` 是一个常见的「向上取整」写法，确保总线程数不少于元素数。

### 错误检查

```cpp
CHECK_CUDA_ERROR(cudaGetLastError());
CHECK_CUDA_ERROR(cudaDeviceSynchronize());
```

Kernel 启动是**异步**的——CPU 发出启动命令后立即往下走，不会等 GPU 执行完。所以：
- `cudaGetLastError()` 检查启动参数有没有错
- `cudaDeviceSynchronize()` 等 GPU 执行完，顺便检查执行过程中有没有出错

> 写 CUDA 代码一定要养成错误检查的习惯，否则出了问题根本不知道哪错了。

## 编译和运行

假设你已经安装了 CUDA Toolkit，编译很简单：

```bash
nvcc vector_add.cu -o vector_add
./vector_add
```

输出大概长这样：

```
元素总数: 1048576
错误数量: 0
结果: ✅ 正确
```

## CUDA 内存层次概览

`vector_add` 只用到了全局内存（global memory），但 CUDA 实际上有一套复杂的内存层次：

```
┌─────────────────────────────────────────────────────┐
│                 Grid（所有线程）                     │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Block 0 │  │  Block 1 │  │  Block 2 │        │
│  │ ┌──────┐ │  │ ┌──────┐ │  │ ┌──────┐ │        │
│  │ │Shared│ │  │ │Shared│ │  │ │Shared│ │        │
│  │ └──────┘ │  │ └──────┘ │  │ └──────┘ │        │
│  └──────────┘  └──────────┘  └──────────┘        │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │            Global Memory（全局显存）          │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  每个线程还有自己的 Registers（寄存器）             │
│  和 Local Memory（本地内存，溢出用）               │
└─────────────────────────────────────────────────────┘
```

| 内存类型 | 位置 | 访问速度 | 大小 | 谁能访问 |
|----------|------|----------|------|----------|
| Register | SM 内 | 最快（1 cycle） | 很小（每个线程几十 KB） | 只有自己 |
| Shared Memory | SM 内 | 很快（~10 cycles） | 很小（每个 SM 几十 KB） | 同一个 Block 内的线程 |
| Global Memory | 显存 | 慢（~300-500 cycles） | 大（几十 GB） | 所有线程 + Host |
| Local Memory | 显存 | 慢 | - | 只有自己（寄存器溢出时用） |

**优化 CUDA 程序的核心思路之一，就是尽量减少对全局内存的访问，把数据放在更快的内存里。** 但这是后面的话题了。

## 关键概念总结表

最后，用一张表把这篇文章的核心概念收个尾：

| 概念 | 说明 | 类比 |
|------|------|------|
| Host | CPU 及其内存 | 项目经理 |
| Device | GPU 及其显存 | 一大群实习生 |
| Kernel | 在 GPU 上并行执行的函数 | 工作任务描述 |
| Thread | 最小执行单元 | 单个实习生 |
| Block | 一组线程，可共享内存、可同步 | 一个小组 |
| Grid | 一次启动的所有 Block | 整个项目组 |
| `__global__` | Kernel 函数的修饰符 | - |
| `<<<G, B>>>` | 启动配置：G 个 block，每个 B 个线程 | 「安排 G 组人，每组 B 个」 |
| `threadIdx.x` | 线程在 block 内的编号 | 组内工号 |
| `blockIdx.x` | block 在 grid 内的编号 | 组号 |
| `blockDim.x` | 每个 block 的线程数 | 每组人数 |
| `cudaMalloc` | 在 GPU 上分配内存 | 给实习生分配工位 |
| `cudaMemcpy` | Host 和 Device 之间拷贝数据 | 传资料/交成果 |

## 写在最后

`vector_add` 是最简单的 CUDA 程序，但它包含了 CUDA 编程模型最核心的思想：

1. 数据要在 Host 和 Device 之间显式搬运
2. Kernel 用海量线程并行执行同一段代码
3. 每个线程通过内置变量计算自己的位置，处理对应的数据
4. 需要注意边界检查和错误处理

掌握了这些，你就入门了。后面要学的——共享内存、同步、warp shuffle、bank conflict、coalescing 等等——都是在这个基础上做优化。

下一篇文章，我们会用 Triton 来写 FlashAttention，看看如何在更高的抽象层面上写出高性能的 GPU 算子。

> 如果你觉得这篇文章对你有帮助，欢迎在评论区留言讨论！
