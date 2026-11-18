# Weekly reports

One Markdown file per report, named by date. The site at <https://pillow-1.github.io/streaming_hd/> is built from these files by [`site/build.py`](../site/build.py) and published to the `gh-pages` branch by [`site/publish.sh`](../site/publish.sh) after the report is committed to `main`.

| Date | Report |
|---|---|
| 2026-11-18 | [Errors Are Introduced, Then Carried](2026-11-18.md) |

## Writing a report

- Start the file with a front-matter block: `title`, `deck`, `date`, `kicker`, `summary`, `cover` and `links` (`Label|path; Label|path`).
- Put figures in `assets/<date>/` and the scripts that draw them in `scripts/`.
- Write math as on GitHub: display math in a `math` code fence, inline math as `` $`...`$ ``.
- An HTML comment `<!-- kicker: The **word** -->` before a section heading adds the small label shown above it on the site.
