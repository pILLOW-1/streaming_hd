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
GITHUB_ICON = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.52-1.33-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.84 1.19 3.1 0 4.42-2.7 5.39-5.26 5.68.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5Z"/></svg>')
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"
FONTS = ("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Instrument+Serif:ital@0;1"
         "&family=JetBrains+Mono:wght@400;500;600&display=swap")


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
    # a paragraph starting with "↳ " right after a figure continues its caption (links allowed)
    out = re.sub(r'</figcaption></figure>\s*<p>↳ (.*?)</p>', r' \1</figcaption></figure>', out, flags=re.S)
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
<script>if(location.pathname.endsWith("/index.html"))history.replaceState(null,"",location.pathname.slice(0,-10)+location.search+location.hash)</script>
<meta name="description" content="{html.escape(description)}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<link rel="stylesheet" href="{up}static/style.css">
<link rel="icon" href="{up}static/mark.svg" type="image/svg+xml">
</head>
<body>
<header class="site-header">
  <a class="brand" href="{home}"><span class="brand-mark"><img src="{up}static/mark.svg" alt=""></span><span>{SITE_NAME}</span></a>
  <nav class="site-nav"><a href="{home}">Weekly</a><a href="{REPO}">Repository</a></nav>
</header>
<main class="site-main">
{body}
</main>
<footer class="site-footer"><div class="site-footer__row">
  <div class="site-footer__left">
    <span>{SITE_NAME} — weekly notes on streaming hallucination detection.</span>
    <a class="site-footer__credit" href="https://www.agenticresearch.sh/">Site style adapted from agenticresearch.sh</a>
  </div>
  <div class="site-footer__social"><a href="{REPO}" aria-label="GitHub repository">{GITHUB_ICON}</a></div>
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
    show_hero = cover and meta.get("hero", "cover") != "none"  # "hero: none" when the cover already opens §1
    hero = f'<figure class="hero"><img src="../{html.escape(cover)}" alt=""></figure>' if show_hero else ""
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
    """Each report has a compact row and an expanded card; the open item shows its card (first by default)."""
    def item(p: dict, is_open: bool) -> str:
        date = dt.date.fromisoformat(p["date"])
        title, note, href = html.escape(p["title"]), html.escape(p.get("summary", "")), p["href"]
        img = f'<img src="{html.escape(p["cover"])}" alt="" loading="lazy" decoding="async">' if p.get("cover") else ""
        art = "idx-art" if img else "idx-art idx-art--none"
        return f"""<li class="idx-item{' is-open' if is_open else ''}">
  <div class="idx-item__shape idx-item__shape--row"><div class="idx-item__clip"><div class="idx-row">
    <a class="idx-row__artlink" aria-hidden="true" tabindex="-1" href="{href}"><span class="{art} idx-row__art">{img}</span></a>
    <div class="idx-row__main"><h2 class="idx-row__title"><a href="{href}">{title}</a></h2><p class="idx-row__note">{note}</p></div>
    <div class="idx-row__right"><span class="idx-row__meta">{date:%b %-d, %Y}</span></div>
  </div></div></div>
  <div class="idx-item__shape idx-item__shape--card"><div class="idx-item__clip"><div class="idx-card">
    <a class="idx-card__artlink" aria-hidden="true" tabindex="-1" href="{href}"><span class="{art} idx-card__art">{img}</span></a>
    <div class="idx-card__body"><p class="idx-card__meta"><span>{date:%b %-d, %Y}</span></p>
      <h2 class="idx-card__title"><a href="{href}">{title}</a></h2><p class="idx-card__note">{note}</p></div>
  </div></div></div>
</li>"""

    items = "\n".join(item(p, i == 0) for i, p in enumerate(posts))
    # Hover or keyboard focus opens one item at a time; the hidden shape is made inert.
    script = """<script>
(function(){var items=[].slice.call(document.querySelectorAll('.idx-item'));
function sync(){items.forEach(function(it){var o=it.classList.contains('is-open');
it.querySelector('.idx-item__shape--row .idx-row').inert=o;it.querySelector('.idx-item__shape--card .idx-card').inert=!o;});}
function open(it){items.forEach(function(x){x.classList.toggle('is-open',x===it);});sync();}
items.forEach(function(it){it.addEventListener('mouseenter',function(){open(it);});it.addEventListener('focusin',function(){open(it);});});
sync();})();
</script>"""
    return f'<section class="blog-index"><header class="blog-index__head"><h1>Weekly</h1></header>\n' \
           f'<ul class="idx-list">{items}</ul></section>{script}'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "_site")
    ap.add_argument("--weekly", type=Path, default=WEEKLY, help="report folder (default: weekly/)")
    args = ap.parse_args()
    out, weekly = args.out.resolve(), args.weekly.resolve()
    if out.exists():
        shutil.rmtree(out)
    (out / "weekly").mkdir(parents=True)
    shutil.copytree(STATIC, out / "static")
    if (weekly / "assets").exists():
        shutil.copytree(weekly / "assets", out / "weekly" / "assets")
    posts = []
    for src in sorted(weekly.glob("*.md")):
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
