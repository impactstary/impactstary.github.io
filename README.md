# Notes on AI Systems

> Haitao Chen 的个人技术博客 — AI 基础设施方向的学习与实践

一个极简风格的静态博客，基于纯 HTML/CSS/JS + Markdown，可直接部署到 GitHub Pages。

## ✨ 特性

- 🎨 **极简技术风** — 干净的阅读体验，专注内容
- 📝 **Markdown 写作** — 直接写 Markdown 文件即可发布
- 💻 **代码高亮** — Prism.js 驱动，支持 Python/C++/CUDA/SQL 等
- 🔍 **全文搜索** — 标题、摘要、标签实时搜索
- 🏷️ **分类标签** — 按分类浏览文章
- 📱 **响应式设计** — 手机、平板、桌面完美适配
- 🚀 **零构建** — 纯静态，无需 Node.js，直接部署

## 📁 目录结构

```
haitao-tech-notes/
├── index.html              # 首页（文章列表 + 搜索 + 分类）
├── post.html               # 文章详情页模板
├── about.html              # 关于页面
├── posts/                  # Markdown 文章
│   ├── 01-hello-world.md
│   ├── 02-pytorch-autograd.md
│   └── ...
├── assets/
│   ├── css/
│   │   ├── style.css       # 主样式
│   │   └── prism.css       # 代码高亮样式
│   └── js/
│       ├── main.js         # 首页逻辑（搜索、分类、文章列表）
│       ├── post.js         # 文章页逻辑（Markdown 渲染）
│       └── prism.js        # 代码高亮库
└── README.md
```

## 🚀 快速开始

### 本地预览

直接用浏览器打开 `index.html` 即可，或使用简单的 HTTP 服务器：

```bash
# Python 3
python3 -m http.server 8000

# 然后访问 http://localhost:8000
```

> ⚠️ 注意：直接双击打开 `index.html` 时，由于浏览器的 CORS 限制，Markdown 文章可能无法加载。
> 建议使用本地 HTTP 服务器预览。

### 部署到 GitHub Pages

本博客按用户站点（`<username>.github.io`）方式部署：

1. **创建仓库**：仓库命名为 `impactstary.github.io`（必须与 GitHub 用户名一致），设为 Public

2. **推送代码**：
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/impactstary/impactstary.github.io.git
   git push -u origin main
   ```

3. **开启 Pages**：
   - 进入仓库 Settings → Pages
   - Source 选择 `Deploy from a branch`
   - Branch 选择 `main`，目录选择 `/ (root)`
   - 点击 Save

4. **等待部署**：几十秒后，博客就会在 `https://impactstary.github.io` 上线

## ✍️ 写新文章

### 第一步：创建 Markdown 文件

在 `posts/` 目录下新建 `.md` 文件，比如 `10-my-new-post.md`。

```markdown
---
title: 我的新文章
date: 2026-10-10
category: CUDA
tags: [CUDA, 性能优化]
---

# 标题

这里是正文内容...
```

### 第二步：在 main.js 中添加文章元数据

打开 `assets/js/main.js`，在 `POSTS` 数组中添加：

```javascript
{
    id: "my-new-post",
    title: "我的新文章",
    date: "2026-10-10",
    category: "CUDA",
    tags: ["CUDA", "性能优化"],
    excerpt: "文章摘要，显示在首页列表中。",
    file: "10-my-new-post.md"
},
```

### 第三步：提交并推送

```bash
git add .
git commit -m "add new post: my-new-post"
git push
```

几分钟后 GitHub Pages 会自动更新。

## 🎨 自定义

### 修改博客名称

编辑 `index.html`、`post.html`、`about.html` 中的 `<title>` 和 `.site-title`。

### 修改分类

编辑 `assets/js/main.js` 中的 `CATEGORIES` 数组。

### 修改配色

编辑 `assets/css/style.css` 顶部的 `:root` CSS 变量。

## 📄 License

MIT License — 博客代码和原创文章均自由使用，转载请注明出处。
