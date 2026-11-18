#!/usr/bin/env python3
"""Build the Streaming HD weekly site from weekly/*.md into a static directory.

Each weekly report is one Markdown file with a small front-matter block. The same
file renders on GitHub (math in ```math fences and $`...`$) and on the site, where
math is typeset in the browser by KaTeX.

    python site/build.py --out _site
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import re
import shutil
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
WEEKLY = ROOT / "weekly"
STATIC = Path(__file__).resolve().parent / "static"
REPO = "https://github.com/pILLOW-1/streaming_hd"
SITE_NAME = "Streaming HD"
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"
FONTS = ("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Instrument+Serif:ital@0;1"
         "&family=JetBrains+Mono:wght@400;500&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,700;"
         "1,8..60,400&display=swap")


def front_matter(text: str) -> tuple[dict, str]:
    """Parse a leading '---' block of 'key: value' lines."""
    meta: dict = {}
    if text.startswith("---\n"):
        head, text = text[4:].split("\n---\n", 1)
        for line in head.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip()
    return meta, text


def strip_title_block(body: str, cover: str | None) -> str:
    """Drop the leading H1, its italic deck line and the cover image; the site renders them from metadata."""
    lines = body.lstrip("\n").splitlines()
    while lines and (lines[0].startswith("# ") or not lines[0].strip()
                     or (lines[0].startswith("*") and lines[0].endswith("*"))
                     or (cover and lines[0].startswith("![") and f"({cover})" in lines[0])):
        lines.pop(0)
    return "\n".join(lines)


def protect(body: str) -> tuple[str, dict]:
    """Replace math and kicker comments by tokens so Markdown leaves them untouched."""
    store: dict = {}

    def keep(kind: str, value: str) -> str:
        token = f"ZZ{kind}{len(store)}ZZ"
        store[token] = (kind, value)
        return token

    body = re.sub(r"```math\n(.*?)\n```", lambda m: "\n\n" + keep("D", m.group(1)) + "\n\n", body, flags=re.S)
    body = re.sub(r"\$`(.+?)`\$", lambda m: keep("I", m.group(1)), body)
    body = re.sub(r"<!--\s*kicker:\s*(.*?)\s*-->", lambda m: "\n\n" + keep("K", m.group(1)) + "\n\n", body)
    return body, store


def restore(out: str, store: dict) -> str:
    for token, (kind, value) in store.items():
        if kind == "D":
            out = out.replace(f"<p>{token}</p>", f'<div class="math-display">\\[{html.escape(value)}\\]</div>')
        elif kind == "I":
            out = out.replace(token, f'<span class="math-inline">\\({html.escape(value)}\\)</span>')
        else:
            kicker = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(value))
            out = out.replace(f"<p>{token}</p>", f'<p class="section-kicker">{kicker}</p>')
    return out


def rewrite(out: str) -> str:
    """Figures with captions, repo links to GitHub, assets relative to the post page, scrollable tables."""
    out = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)" ?/?></p>',
                 r'<figure><img alt="\1" src="\2" loading="lazy"><figcaption>\1</figcaption></figure>', out)
    out = re.sub(r'(src|href)="assets/', r'\1="../assets/', out)
    out = re.sub(r'href="\.\./([^"]+)"', lambda m: f'href="{REPO}/blob/main/{m.group(1)}"', out)
    out = out.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    return out


def repo_link(path: str) -> str:
    return f"{REPO}/blob/main/{path[3:]}" if path.startswith("../") else path


def page(title: str, body: str, depth: int, description: str = "") -> str:
    up = "../" * depth
    home = up or "./"  # directory URLs, never .../index.html
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description)}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<link rel="stylesheet" href="{up}static/style.css">
<link rel="icon" href="{up}static/mark.svg" type="image/svg+xml">
</head>
<body>
<header class="site-header"><div class="site-header-inner">
  <a class="brand" href="{home}"><img src="{up}static/mark.svg" alt="" width="22" height="22"><span>{SITE_NAME}</span></a>
  <nav><a href="{home}">Weekly</a><a href="{REPO}">Repository</a></nav>
</div></header>
{body}
<footer class="site-footer"><div class="site-footer-inner">
  <p>{SITE_NAME} — weekly notes on streaming hallucination detection.</p>
  <a href="{REPO}" aria-label="GitHub repository">GitHub</a>
</div></footer>
<script defer src="{KATEX}/katex.min.js"></script>
<script defer src="{KATEX}/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{{delimiters:[{{left:'\\\\[',right:'\\\\]',display:true}},{{left:'\\\\(',right:'\\\\)',display:false}}],throwOnError:false}})"></script>
</body>
</html>
"""


def build_post(src: Path, meta: dict, body: str) -> tuple[str, list]:
    body = strip_title_block(body, meta.get("cover"))
    body, store = protect(body)
    md = markdown.Markdown(extensions=["tables", "fenced_code", "footnotes", "toc", "sane_lists", "attr_list"],
                           extension_configs={"toc": {"permalink": False}})
    out = restore(rewrite(md.convert(body)), store)
    sections = [(t["id"], t["name"]) for t in md.toc_tokens]
    return out, sections


def render_post(meta: dict, content: str, sections: list, src: Path) -> str:
    date = dt.date.fromisoformat(meta["date"])
    toc = "\n".join(f'<li><a href="#{sid}">{html.escape(name)}</a></li>' for sid, name in sections)
    links = [part.split("|", 1) for part in meta.get("links", "").split(";") if "|" in part]
    links.append(("Markdown source", f"../weekly/{src.name}"))
    buttons = "".join(f'<a class="pill" href="{html.escape(repo_link(url.strip()))}">{html.escape(label.strip())} →</a>'
                      for label, url in links)
    cover = meta.get("cover")
    hero = f'<figure class="hero"><img src="../{html.escape(cover)}" alt=""></figure>' if cover else ""
    return f"""<div class="ready-bar"><div class="ready-inner">Report · {date:%B %-d, %Y}</div></div>
<div class="post-layout">
  <aside class="post-aside">
    <p class="crumb"><a href="../../">Weekly</a> / Report</p>
    <p class="aside-label">Contents</p>
    <ol class="toc">{toc}</ol>
  </aside>
  <article class="post">
    <p class="post-kicker">{html.escape(meta.get('kicker', 'A weekly report'))} · {date:%B %Y}</p>
    <h1 class="post-title">{html.escape(meta['title'])}</h1>
    <p class="post-deck">{html.escape(meta.get('deck', ''))}</p>
    <p class="post-meta">{SITE_NAME} · {date:%B %-d, %Y}</p>
    <div class="pills">{buttons}</div>
    {hero}
    <div class="prose">{content}</div>
    <div class="post-end"><span>End of report</span><a href="../../">Browse all reports →</a></div>
  </article>
</div>"""


def render_index(posts: list) -> str:
    def card(p: dict, featured: bool) -> str:
        date = dt.date.fromisoformat(p["date"])
        img = f'<img src="{html.escape(p["cover"])}" alt="" loading="lazy">' if p.get("cover") else ""
        if featured:
            return f"""<a class="feature" href="{p['href']}">
  <div class="feature-img">{img}</div>
  <div class="feature-text"><p class="date-caps">{date:%b %-d, %Y}</p>
  <h2>{html.escape(p['title'])}</h2><p>{html.escape(p.get('summary', ''))}</p></div></a>"""
        return f"""<a class="row" href="{p['href']}"><div class="row-img">{img}</div>
  <div class="row-text"><h3>{html.escape(p['title'])}</h3><p>{html.escape(p.get('summary', ''))}</p></div>
  <p class="row-date">{date:%b %-d, %Y}</p></a>"""

    items = [card(p, i == 0) for i, p in enumerate(posts)]
    return f'<main class="index"><h1 class="index-title">Weekly</h1>\n{items[0] if items else ""}\n' \
           f'<div class="rows">{"".join(items[1:])}</div></main>'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "_site")
    out = ap.parse_args().out.resolve()
    if out.exists():
        shutil.rmtree(out)
    (out / "weekly").mkdir(parents=True)
    shutil.copytree(STATIC, out / "static")
    if (WEEKLY / "assets").exists():
        shutil.copytree(WEEKLY / "assets", out / "weekly" / "assets")
    posts = []
    for src in sorted(WEEKLY.glob("*.md")):
        if src.name.lower() == "readme.md":
            continue
        meta, body = front_matter(src.read_text(encoding="utf-8"))
        slug = src.stem
        content, sections = build_post(src, meta, body)
        target = out / "weekly" / slug / "index.html"
        target.parent.mkdir(parents=True)
        target.write_text(page(f"{meta['title']} · {SITE_NAME}", render_post(meta, content, sections, src), 2,
                               meta.get("summary", "")), encoding="utf-8")
        posts.append(dict(meta, href=f"weekly/{slug}/",
                          cover=f"weekly/{meta['cover']}" if meta.get("cover") else None))
        print("built", target.relative_to(out))
    posts.sort(key=lambda p: p["date"], reverse=True)
    (out / "index.html").write_text(page(f"Weekly · {SITE_NAME}", render_index(posts), 0,
                                         "Weekly research notes on streaming hallucination detection."), encoding="utf-8")
    (out / ".nojekyll").write_text("")
    print("built index.html with", len(posts), "reports")


if __name__ == "__main__":
    main()
