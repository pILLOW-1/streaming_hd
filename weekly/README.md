# Weekly reports

One Markdown file per report, named by date. The site at <https://pillow-1.github.io/streaming_hd/> is built from these files by [`site/build.py`](../site/build.py) and published by a GitHub Actions workflow on every push to `main` that changes `weekly/` or `site/`.

| Date | Report |
|---|---|
| 2026-11-21 | [The Challenge of Streaming HD](2026-11-21.md) |
| 2026-11-20 | [Errors Are Introduced, Then Carried](2026-11-20.md) |

## Writing a report

- Start the file with a front-matter block: `title`, `deck`, `date`, `kicker`, `summary`, `cover`, `links` (`Label|path; Label|path`) and optionally `hero: none` when the cover image already appears in the text.
- Put figures in `assets/<date>/` and the scripts that draw them in `scripts/`.
- Write math as on GitHub: display math in a `math` code fence, inline math as `` $`...`$ ``.
- An HTML comment `<!-- kicker: The **word** -->` before a section heading adds the small label shown above it on the site.
