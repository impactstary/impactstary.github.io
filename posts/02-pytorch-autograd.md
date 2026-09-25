---
title: 深入理解 PyTorch Autograd 的计算图与反向传播
date: 2024-01-22
tags: [PyTorch, 深度学习, Autograd, 反向传播]
description: 从底层原理出发，彻底搞懂 PyTorch 的自动微分机制：计算图是如何动态构建的？反向传播又是如何沿着图一路算回去的？
---

# 深入理解 PyTorch Autograd 的计算图与反向传播

如果你用过 PyTorch，一定对下面这段代码不陌生：

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x
y.backward()
print(x.grad)  # 输出: tensor(7.)
```

`x.grad` 为什么是 7？答案很简单：y = x² + 3x，求导得 dy/dx = 2x + 3，代入 x=2 得 7。

但你有没有想过，**PyTorch 是怎么知道这个求导关系的？** 它又是如何从 y 出发，一路把梯度传回 x 的？

这篇文章，我们就来把 Autograd（自动微分）这件事彻底搞明白。

## 什么是 Autograd，为什么我们需要它

### 手动求导的噩梦

想象一下，你要实现一个神经网络，有 100 层。如果每一层的梯度都要你手动推导、手写代码，那会是什么光景？

- 推导容易出错
- 改一次网络结构，梯度代码也要跟着改
- 代码量巨大，难以维护

**自动微分（Automatic Differentiation, AD）** 就是来解决这个问题的。它不是数值微分（用有限差分近似），也不是符号微分（用 sympy 那种数学表达式推导），而是**把所有运算分解成一系列基本操作，然后利用链式法则自动计算梯度**。

PyTorch 的 Autograd 就是一套自动微分系统。它的核心思路是：

> 在**前向传播**时，记录下所有运算，构建一张**计算图**；在**反向传播**时，沿着这张图从输出往输入走，利用**链式法则**逐层计算梯度。

## Tensor 的四个关键属性

要理解 Autograd，首先要搞懂 Tensor 的四个核心属性：

| 属性 | 类型 | 作用 |
|------|------|------|
| `data` | Tensor | 存储张量的实际数值 |
| `grad` | Tensor 或 None | 存储梯度值，初始为 None |
| `grad_fn` | Function 或 None | 指向创建该张量的函数（计算图中的节点） |
| `requires_grad` | bool | 标记是否需要计算梯度 |

用一个简单的例子来理解：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y + x

print(x.requires_grad)  # True  — 用户创建，手动开启
print(y.requires_grad)  # True  — 依赖于 x，自动开启
print(z.requires_grad)  # True  — 同理

print(x.grad_fn)        # None  — 叶子节点，没有创建函数
print(y.grad_fn)        # <MulBackward0>  — 由乘法创建
print(z.grad_fn)        # <AddBackward0>  — 由加法创建
```

几个要点：

1. **叶子张量（leaf tensor）**：由用户直接创建的张量，`grad_fn` 为 `None`。
2. 只要运算中有任何一个输入的 `requires_grad=True`，输出的 `requires_grad` 也为 `True`。
3. 每一次运算都会创建一个 `Function` 对象，它知道**如何做正向计算**，也知道**如何做反向计算**。

## 计算图是如何动态构建的

PyTorch 使用的是**动态计算图（Dynamic Computation Graph）**，也叫「定义即运行」（Define-by-Run）。这意味着：**计算图不是预先定义好的，而是在你每次前向传播时实时构建的**。

### 一个具体的例子

我们以 `y = x² + 3*x` 为例，看看计算图长什么样：

```
        x (leaf, requires_grad=True)
        |
        ├─────────────────┐
        |                 |
    PowBackward0     MulBackward0   (x^2)          (3 * x)
        |                 |
        |            MulBackward0  ← constant 3
        |                 |
        └──────┬──────────┘
               |
          AddBackward0      (x^2 + 3*x)
               |
               y
```

每一次张量运算，PyTorch 都会：

1. 执行正向计算，得到结果张量
2. 创建对应的 `Function` 对象（如 `PowBackward0`、`MulBackward0`）
3. 把结果张量的 `grad_fn` 指向这个 Function
4. Function 内部保存着**输入张量的引用**，从而形成一张有向无环图（DAG）

> 注意：这张图是**前向传播时实时构建**的。每调用一次 forward，就会构建一张新的图。这也是为什么 PyTorch 支持动态控制流（比如循环、条件判断）的原因——因为每一次执行都可以有不同的图结构。

## 沿着计算图走一遍反向传播

有了计算图，反向传播就变得很清晰了：从输出节点出发，沿着图的反方向走，每经过一个节点，就用链式法则把梯度乘上去，最终到达叶子节点。

我们还是用 `y = x² + 3*x`，x = 2 的例子。

### 第一步：初始化

调用 `y.backward()` 时，首先做的事是：给 y 一个初始梯度。因为 y 是标量，默认初始梯度是 1.0（也就是 dy/dy = 1）。

```
y.grad = 1.0
```

### 第二步：从 AddBackward0 往回传

y 是由加法得到的：`y = a + b`，其中 a = x²，b = 3x。

加法的反向传播规则很简单：**把梯度原封不动地分给两个输入**。因为 d(a+b)/da = 1，d(a+b)/db = 1。

```
a.grad = y.grad * 1 = 1.0
b.grad = y.grad * 1 = 1.0
```

### 第三步：分别处理 PowBackward0 和 MulBackward0

先看左边的 `a = x²`（PowBackward0）：

d(x²)/dx = 2x，所以：

```
x.grad += a.grad * 2*x = 1.0 * 4 = 4.0
```

再看右边的 `b = 3*x`（MulBackward0）：

乘法的反向传播规则是：`d(a*b)/da = b`，`d(a*b)/db = a`。这里一个输入是 3（常量，不需要梯度），一个是 x。

```
x.grad += b.grad * 3 = 1.0 * 3 = 3.0
```

### 第四步：汇总结果

x 收到了两路梯度，加起来：

```
x.grad = 4.0 + 3.0 = 7.0
```

和我们手动算的一样。完美！

### 用代码验证一下

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x
y.backward()

print(f"y = {y.item()}")          # y = 10.0
print(f"dy/dx = {x.grad.item()}") # dy/dx = 7.0
```

输出：

```
y = 10.0
dy/dx = 7.0
```

## 自定义 Autograd Function

虽然 PyTorch 内置了绝大多数运算的反向传播，但有时候你需要实现一个自定义的运算，并且希望它也能参与自动微分。这时候就需要用到 `torch.autograd.Function`。

### 完整示例：自定义 ReLU

让我们手写一个 ReLU 来体会一下：

```python
import torch
from torch.autograd import Function

class MyReLU(Function):
    @staticmethod
    def forward(ctx, input):
        """
        前向传播：计算 output = max(0, input)
        ctx 是一个上下文对象，用来保存前向的中间结果，供反向使用
        """
        # 保存输入，反向传播要用
        ctx.save_for_backward(input)
        # 计算 ReLU
        output = input.clamp(min=0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：根据 grad_output 计算输入的梯度
        grad_output 是从上游传过来的梯度
        """
        # 取出前向保存的输入
        input, = ctx.saved_tensors
        # ReLU 的导数：input > 0 时为 1，否则为 0
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

# 使用自定义 Function
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
y = MyReLU.apply(x)
loss = y.sum()
loss.backward()

print("x     :", x)
print("y     :", y)
print("x.grad:", x.grad)
```

输出：

```
x     : tensor([-2., -1.,  0.,  1.,  2.], requires_grad=True)
y     : tensor([0., 0., 0., 1., 2.], grad_fn=<MyReLUBackward>)
x.grad: tensor([0., 0., 0., 1., 1.])
```

### 自定义 Function 的要点

1. **必须继承 `torch.autograd.Function`**，并实现 `forward` 和 `backward` 两个静态方法。
2. **`forward` 的第一个参数是 `ctx`**（context），用来在前向和反向之间传递数据。
3. **用 `ctx.save_for_backward(...)` 保存张量**，反向时通过 `ctx.saved_tensors` 取出。
4. **`backward` 返回的梯度数量必须和 `forward` 的输入数量一致**。不需要梯度的输入返回 `None`。
5. **使用时调用 `MyFunction.apply(...)`**，而不是直接实例化。

## 常见陷阱与技巧

### 1. 梯度累加，不是覆盖

默认情况下，`backward()` 会把梯度**累加**到 `.grad` 上，而不是覆盖。这是为了支持多次 backward（比如 RNN）。

但在训练循环中，这意味着你需要手动清零梯度：

```python
optimizer.zero_grad()  # 清零
loss.backward()        # 计算梯度
optimizer.step()       # 更新参数
```

忘记清零的话，梯度会越来越大，模型就乱了。

### 2. 叶子节点才能有 grad

只有叶子张量（`is_leaf=True`）的 `grad` 才会被保留。中间变量的梯度在反向传播结束后会被释放，以节省内存。

如果你想查看中间变量的梯度，可以用 `register_hook`：

```python
x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y + 1

# 给 y 注册一个 hook，打印梯度
y.register_hook(lambda grad: print(f"y 的梯度: {grad}"))

z.backward()
# 输出: y 的梯度: 1.0
```

### 3. 用 `torch.no_grad()` 关闭梯度追踪

推理阶段不需要计算梯度，用 `torch.no_grad()` 可以节省内存和计算：

```python
x = torch.tensor(2.0, requires_grad=True)

with torch.no_grad():
    y = x * 2  # y 的 requires_grad 为 False
    print(y.requires_grad)  # False
```

还有一个类似的 `torch.inference_mode()`，比 `no_grad()` 更彻底，连版本计数器都不更新，速度更快。

### 4. 计算图只构建一次

每次 `backward()` 之后，计算图就会被释放（除非设置 `retain_graph=True`）。再次调用 `backward()` 会报错。

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()  # 第一次，正常
y.backward()  # 第二次，报错！
```

如果需要多次 backward，第一次调用时加上 `retain_graph=True`：

```python
y.backward(retain_graph=True)
y.backward()  # 这次就正常了
```

### 5. 非标量的 backward

只有标量才能调用 `backward()`。如果输出是向量，需要传入一个和输出形状相同的 `gradient` 参数：

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# y 是向量，需要传入 gradient
y.backward(gradient=torch.tensor([1.0, 1.0, 1.0]))
print(x.grad)  # tensor([2., 4., 6.])
```

这相当于计算了 `sum(y)` 对 x 的梯度。

## 总结

我们用一张图来回顾一下 Autograd 的完整流程：

```
  前向传播 (Forward Pass)          反向传播 (Backward Pass)
─────────────────────────        ─────────────────────────
                                     ┌──────────┐
   x ──[pow]──> x² ──┐               │  dL/dy   │  = 1.0
                     ├─[add]─> y     └────┬─────┘
   x ──[mul]──> 3x ──┘                    │
                                          ▼
   构建计算图 (DAG)                 链式法则求梯度
   每个运算 → Function 节点         从输出向叶子节点传播
   每个张量 → 图中的边              梯度累加到 .grad
```

掌握 Autograd 是深入理解 PyTorch 的第一步。它不仅仅是一个「自动求导工具」，更是整个 PyTorch 框架的基石。理解了它，你才能更好地调试模型、写出高效的自定义算子，甚至去研究框架本身的实现。

下一篇文章，我们会从 Python 层面深入到 CUDA 层面，看看这些算子在 GPU 上到底是怎么跑的。

> 如果你觉得这篇文章对你有帮助，欢迎在评论区留言讨论。发现错误也欢迎指出，一起进步！
