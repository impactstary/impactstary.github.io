/* ============================================================
   Haitao Chen — Notes on AI Systems · Post Page JS
   Handles: load markdown, render with marked.js, highlight code
   ============================================================ */

document.addEventListener("DOMContentLoaded", function() {
    const params = new URLSearchParams(window.location.search);
    const postId = params.get("id");

    if (!postId) {
        showError("未指定文章 ID");
        return;
    }

    const post = (window.__POSTS__ || []).find(p => p.id === postId);
    if (!post) {
        showError("未找到该文章");
        return;
    }

    // Set page title
    document.title = post.title + " — Haitao Chen";

    // Render header
    document.getElementById("post-title").textContent = post.title;
    document.getElementById("post-meta").textContent =
        formatDate(post.date) + " · " + post.category;
    document.getElementById("post-tags").innerHTML =
        post.tags.map(t => `<span class="tag">${t}</span>`).join("");

    // Load and render markdown
    loadMarkdown(post.file);
});

function loadMarkdown(filename) {
    const contentEl = document.getElementById("post-content");

    fetch("posts/" + filename)
        .then(response => {
            if (!response.ok) throw new Error("文章加载失败");
            return response.text();
        })
        .then(text => {
            // Configure marked
            if (typeof marked !== "undefined") {
                marked.setOptions({
                    breaks: true,
                    gfm: true,
                    highlight: function(code, lang) {
                        if (typeof Prism !== "undefined" && Prism.languages[lang]) {
                            return Prism.highlight(code, Prism.languages[lang], lang);
                        }
                        return code;
                    }
                });
                contentEl.innerHTML = marked.parse(text);
            } else {
                // Fallback: plain text
                contentEl.innerHTML = "<pre>" + escapeHtml(text) + "</pre>";
            }

            // Re-highlight all code blocks with Prism
            if (typeof Prism !== "undefined") {
                Prism.highlightAll();
            }

            // Add IDs to headings for anchor links
            addHeadingIds(contentEl);
        })
        .catch(err => {
            showError("文章加载失败：" + err.message);
        });
}

function showError(msg) {
    document.getElementById("post-title").textContent = "出错了";
    document.getElementById("post-content").innerHTML =
        '<p style="color:#ef4444;">' + msg + '</p>';
}

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

function addHeadingIds(container) {
    const headings = container.querySelectorAll("h2, h3, h4");
    const usedIds = {};
    headings.forEach(h => {
        let id = h.textContent.toLowerCase()
            .replace(/[^\w\u4e00-\u9fa5\s-]/g, "")
            .replace(/\s+/g, "-")
            .replace(/-+/g, "-");
        if (usedIds[id]) {
            usedIds[id]++;
            id = id + "-" + usedIds[id];
        } else {
            usedIds[id] = 1;
        }
        h.id = id;
    });
}
