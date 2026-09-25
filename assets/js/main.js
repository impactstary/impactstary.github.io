/* ============================================================
   Haitao Chen — Notes on AI Systems · Main JS
   Handles: post listing, search, category filtering
   ============================================================ */

// --- Post metadata ---
const POSTS = [
    {
        id: "hello-world",
        title: "你好，世界 — 这个博客的由来",
        date: "2026-09-19",
        category: "随笔",
        tags: ["博客", "学习"],
        excerpt: "为什么要开这个技术博客？以及接下来打算写些什么。",
        file: "01-hello-world.md"
    },
    {
        id: "pytorch-autograd",
        title: "深入理解 PyTorch Autograd 的计算图与反向传播",
        date: "2026-09-20",
        category: "PyTorch",
        tags: ["PyTorch", "深度学习", "反向传播"],
        excerpt: "从 loss.backward() 出发，一步步拆解 autograd 的计算图构建与反向传播执行过程。",
        file: "02-pytorch-autograd.md"
    },
    {
        id: "cuda-vector-add",
        title: "CUDA 入门：从 vector_add 理解 GPU 编程模型",
        date: "2026-09-22",
        category: "CUDA",
        tags: ["CUDA", "GPU", "并行计算"],
        excerpt: "第一个 CUDA 程序的完整解析：Thread、Block、Grid 的层级关系与内存模型。",
        file: "03-cuda-vector-add.md"
    },
    {
        id: "triton-flash-attention",
        title: "用 Triton 手写 FlashAttention：从原理到实现",
        date: "2026-09-25",
        category: "Triton",
        tags: ["Triton", "FlashAttention", "算子优化"],
        excerpt: "FlashAttention 的核心思想是 tiling + online softmax，用 Triton 实现它比 CUDA 简洁多少？",
        file: "04-triton-flash-attention.md"
    },
    {
        id: "vllm-paged-attention",
        title: "vLLM 核心：PagedAttention 是如何让推理吞吐翻倍的",
        date: "2026-09-28",
        category: "推理框架",
        tags: ["vLLM", "PagedAttention", "推理优化"],
        excerpt: "借鉴操作系统虚拟内存的思路，PagedAttention 完美解决了 KV Cache 的内存碎片问题。",
        file: "05-vllm-paged-attention.md"
    },
    {
        id: "megatron-3d-parallel",
        title: "Megatron-LM 3D 并行：数据并行、张量并行、流水线并行的组合艺术",
        date: "2026-10-02",
        category: "训练框架",
        tags: ["Megatron-LM", "分布式训练", "3D并行"],
        excerpt: "大模型训练的三驾马车：DP、TP、PP 如何协同工作？各自的通信开销和适用场景是什么？",
        file: "06-megatron-3d-parallel.md"
    },
    {
        id: "cpp-move-semantics",
        title: "现代 C++ 核心：移动语义与右值引用完全指南",
        date: "2026-09-15",
        category: "C++",
        tags: ["C++", "移动语义", "性能优化"],
        excerpt: "左值、右值、将亡值、移动构造、移动赋值、std::move、完美转发... 一文搞懂。",
        file: "07-cpp-move-semantics.md"
    },
    {
        id: "cuda-shared-memory",
        title: "CUDA Shared Memory：用 Tiling 让矩阵乘法快 10 倍",
        date: "2026-09-30",
        category: "CUDA",
        tags: ["CUDA", "Shared Memory", "矩阵乘法"],
        excerpt: "Shared Memory 是 GPU 性能优化的第一武器。用 tiled matrix multiplication 讲清楚数据复用原理。",
        file: "08-cuda-shared-memory.md"
    },
    {
        id: "sglang-radix-attention",
        title: "SGLang 杀手锏：RadixAttention 前缀缓存的原理与实现",
        date: "2026-10-05",
        category: "推理框架",
        tags: ["SGLang", "RadixAttention", "推理优化"],
        excerpt: "当大量请求共享相同的系统提示词时，RadixAttention 能带来多大的性能提升？",
        file: "09-sglang-radix-attention.md"
    }
];

// --- Categories ---
const CATEGORIES = ["全部", "PyTorch", "CUDA", "Triton", "推理框架", "训练框架", "C++", "随笔"];

let currentCategory = "全部";
let currentSearch = "";

// --- DOM Ready ---
document.addEventListener("DOMContentLoaded", function() {
    initCategories();
    renderPosts();
    initSearch();
});

// --- Categories ---
function initCategories() {
    const nav = document.getElementById("category-grid");
    if (!nav) return;

    const counts = {};
    POSTS.forEach(p => {
        counts[p.category] = (counts[p.category] || 0) + 1;
    });
    counts["全部"] = POSTS.length;

    nav.innerHTML = CATEGORIES.map(name => {
        const isActive = name === currentCategory ? "active" : "";
        return `<button class="chip ${isActive}" data-category="${name}">${name}<span class="chip-count">${counts[name] || 0}</span></button>`;
    }).join("");

    nav.querySelectorAll(".chip").forEach(chip => {
        chip.addEventListener("click", function() {
            currentCategory = this.dataset.category;
            nav.querySelectorAll(".chip").forEach(c => c.classList.remove("active"));
            this.classList.add("active");
            renderPosts();
        });
    });
}

// --- Search ---
function initSearch() {
    const input = document.getElementById("search-input");
    if (!input) return;

    input.addEventListener("input", function() {
        currentSearch = this.value.toLowerCase().trim();
        renderPosts();
    });
}

// --- Filter posts ---
function getFilteredPosts() {
    return POSTS.filter(post => {
        if (currentCategory !== "全部" && post.category !== currentCategory) {
            return false;
        }
        if (currentSearch) {
            const searchTarget = (
                post.title + " " +
                post.excerpt + " " +
                post.tags.join(" ") + " " +
                post.category
            ).toLowerCase();
            if (!searchTarget.includes(currentSearch)) {
                return false;
            }
        }
        return true;
    }).sort((a, b) => new Date(b.date) - new Date(a.date));
}

// --- Render posts ---
function renderPosts() {
    const list = document.getElementById("post-list");
    const noResults = document.getElementById("no-results");
    if (!list) return;

    const posts = getFilteredPosts();

    if (posts.length === 0) {
        list.innerHTML = "";
        noResults.style.display = "block";
        return;
    }

    noResults.style.display = "none";
    list.innerHTML = posts.map(post => `
        <a href="post.html?id=${post.id}" class="post-item">
            <span class="post-item-meta">${formatDate(post.date)} · ${post.category}</span>
            <h2 class="post-item-title">${escapeHtml(post.title)}</h2>
            <p class="post-item-excerpt">${escapeHtml(post.excerpt)}</p>
            <span class="post-item-tags">${post.tags.map(t => `<span>${t}</span>`).join("")}</span>
        </a>
    `).join("");
}

// --- Helpers ---
function formatDate(dateStr) {
    const date = new Date(dateStr);
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, "0");
    const day = String(date.getDate()).padStart(2, "0");
    return `${year}-${month}-${day}`;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(text));
    return div.innerHTML;
}

// Export for post.js
window.__POSTS__ = POSTS;
