"""Render a QAReport for a terminal and for a browser."""
from __future__ import annotations

import html
import json
from pathlib import Path

from .severity import QAReport, Severity

_BADGE = {
    Severity.BLOCKER: ("#8c1c13", "#fdecea"),
    Severity.MAJOR: ("#8a5200", "#fdf3e2"),
    Severity.MINOR: ("#3d5a80", "#eef3f8"),
    Severity.INFO: ("#4a5159", "#f0f2f4"),
}


def to_text(report: QAReport) -> str:
    c = report.counts
    width = 74
    lines = [
        "=" * width,
        f"  DATA QA REPORT — {report.dataset_name}",
        f"  {report.run_at_utc}",
        "=" * width,
        f"  train {report.row_counts.get('train_rows', 0):,} rows   "
        f"test {report.row_counts.get('test_rows', 0):,} rows   "
        f"{report.row_counts.get('columns', 0)} columns",
        "",
        f"  {c['passed']}/{c['total']} checks passed   "
        f"BLOCKER {c['blocker']}   MAJOR {c['major']}   MINOR {c['minor']}",
        "=" * width,
        "",
    ]

    for r in report.sorted_results:
        mark = "PASS" if r.passed else "FAIL"
        tag = "" if r.passed else f" [{r.severity.value}]"
        lines.append(f"  {mark}{tag}  {r.check_id}  {r.name}")
        lines.append(f"        {r.summary}")
        if not r.passed and r.remedy:
            for i, chunk in enumerate(_wrap(r.remedy, 64)):
                prefix = "        -> " if i == 0 else "           "
                lines.append(prefix + chunk)
        lines.append("")

    verdict = "FIT TO TRAIN" if report.fit_to_train else "NOT FIT TO TRAIN"
    lines += ["=" * width, f"  VERDICT: {verdict}"]
    if not report.fit_to_train:
        lines.append(f"  {c['blocker']} blocking issue(s). Any model scored on this split")
        lines.append("  will report a number that cannot be believed.")
    lines.append("=" * width)
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words, out, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


def to_html(report: QAReport) -> str:
    c = report.counts
    verdict_ok = report.fit_to_train
    rows = []
    for r in report.sorted_results:
        fg, bg = _BADGE[r.severity]
        badge = (
            '<span class="pass">PASS</span>' if r.passed
            else f'<span class="sev" style="color:{fg};background:{bg}">{r.severity.value}</span>'
        )
        ev = html.escape(json.dumps(r.evidence, indent=2)) if r.evidence else ""
        remedy = (
            f'<p class="remedy">{html.escape(r.remedy)}</p>'
            if r.remedy and not r.passed else ""
        )
        details = (
            f'<details><summary>evidence</summary><pre>{ev}</pre></details>' if ev else ""
        )
        rows.append(f"""
        <tr class="{'ok' if r.passed else 'bad'}">
          <td class="badge">{badge}</td>
          <td>
            <div class="cid">{html.escape(r.check_id)}</div>
            <div class="cname">{html.escape(r.name)}</div>
            <div class="csum">{html.escape(r.summary)}</div>
            {remedy}{details}
          </td>
        </tr>""")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data QA Report — {html.escape(report.dataset_name)}</title>
<style>
  :root {{ --fg:#1a1d21; --muted:#6b7280; --line:#e3e6ea; --bg:#ffffff; --card:#f7f8fa; }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{ --fg:#e6e8ea; --muted:#9aa3ad; --line:#2c3238; --bg:#15181b; --card:#1d2126; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
    font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:900px; margin:0 auto; padding:32px 16px 64px; }}
  h1 {{ font-size:22px; margin:0 0 4px; }}
  .meta {{ color:var(--muted); font-size:13px; margin-bottom:20px; }}
  .verdict {{ padding:14px 18px; border-radius:8px; font-weight:600; margin-bottom:8px;
    background:{'#e7f5ec' if verdict_ok else '#fdecea'};
    color:{'#1b5e38' if verdict_ok else '#8c1c13'}; }}
  .verdict small {{ display:block; font-weight:400; margin-top:4px; }}
  .tiles {{ display:flex; gap:10px; flex-wrap:wrap; margin:16px 0 24px; }}
  .tile {{ flex:1 1 120px; background:var(--card); border:1px solid var(--line);
    border-radius:8px; padding:12px 14px; }}
  .tile b {{ display:block; font-size:22px; }}
  .tile span {{ color:var(--muted); font-size:12px; text-transform:uppercase;
    letter-spacing:.06em; }}
  table {{ width:100%; border-collapse:collapse; }}
  td {{ border-top:1px solid var(--line); padding:14px 8px; vertical-align:top; }}
  td.badge {{ width:86px; }}
  .pass, .sev {{ font-size:11px; font-weight:700; letter-spacing:.05em;
    padding:3px 7px; border-radius:4px; }}
  .pass {{ color:#1b5e38; background:#e7f5ec; }}
  .cid {{ font:600 11px ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--muted); }}
  .cname {{ font-weight:600; margin:1px 0 3px; }}
  .csum {{ color:var(--muted); font-size:14px; }}
  .remedy {{ margin:8px 0 0; padding-left:11px; border-left:2px solid var(--line);
    font-size:14px; }}
  details {{ margin-top:8px; }}
  summary {{ cursor:pointer; color:var(--muted); font-size:12px; }}
  pre {{ background:var(--card); border:1px solid var(--line); border-radius:6px;
    padding:10px; overflow:auto; font-size:12px; max-height:300px; }}
  tr.ok .cname {{ font-weight:500; }}
</style></head><body><div class="wrap">
  <h1>Data QA Report</h1>
  <div class="meta">{html.escape(report.dataset_name)} &middot; {report.run_at_utc} &middot;
    {report.row_counts.get('train_rows',0):,} train / {report.row_counts.get('test_rows',0):,} test rows</div>
  <div class="verdict">{'FIT TO TRAIN' if verdict_ok else 'NOT FIT TO TRAIN'}
    <small>{'No blocking issues found.' if verdict_ok else
      f"{c['blocker']} blocking issue(s). A model scored on this split will report a number that cannot be believed."}</small></div>
  <div class="tiles">
    <div class="tile"><b>{c['passed']}/{c['total']}</b><span>checks passed</span></div>
    <div class="tile"><b>{c['blocker']}</b><span>blocker</span></div>
    <div class="tile"><b>{c['major']}</b><span>major</span></div>
    <div class="tile"><b>{c['minor']}</b><span>minor</span></div>
  </div>
  <table>{''.join(rows)}</table>
</div></body></html>"""


def write_html(report: QAReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_html(report), encoding="utf-8")
    return path
