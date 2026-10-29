"""Export a static, serverless version of the demo (for GitHub Pages or any static host).

    python -m agent_judge export-site          # writes docs/index.html + docs/data.js

The page lets anyone browse every transcript with the saved verdicts of each judge, the
human label, and the actual test outcome, plus the full results report. Everything is
pre-computed, so it needs no server, GPU, or API key.
"""

from __future__ import annotations

import json
from pathlib import Path

from .data import latest_by_id, load_jsonl, load_trajectories
from .render import render_transcript
from .rubric import DIMENSIONS


def _rater_name(judge: str, prompt: str) -> str:
    if judge.startswith("mock:heuristic"):
        return "Heuristic baseline (no LLM)"
    return f"{judge.split(':', 1)[-1]} · prompt {prompt}"


def _inline(text: str) -> str:
    import html
    import re

    t = html.escape(text, quote=False)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"_([^_]+)_(?=\W|$)", r"<i>\1</i>", t)
    return t


def markdown_to_html(md: str) -> str:
    """Tiny converter for the report's markdown: headings, bullets, tables, paragraphs."""
    out, para, i = [], [], 0
    lines = md.splitlines()

    def flush():
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
            para.clear()

    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            flush()
        elif line.startswith("#"):
            flush()
            level = min(len(line) - len(line.lstrip("#")) + 1, 4)
            out.append(f"<h{level}>{_inline(line.lstrip('#').strip())}</h{level}>")
        elif line.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            body = [r for r in rows[1:] if not all(set(c) <= set("-: ") for c in r)]
            h = "".join(f"<th>{_inline(c)}</th>" for c in rows[0])
            b = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body)
            out.append(f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>')
            continue
        elif line.lstrip().startswith("- "):
            flush()
            items = []
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                items.append(f"<li>{_inline(lines[i].lstrip()[2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        else:
            para.append(line.strip())
        i += 1
    flush()
    return "\n".join(out)


def build_data(sample_path: str | Path, labels_path: str | Path, results_dir: str | Path) -> dict:
    results_dir = Path(results_dir)
    trajs = load_trajectories(sample_path)
    runs = []
    for p in sorted(results_dir.glob("judge_*.jsonl")):
        recs = {i: r for i, r in latest_by_id(load_jsonl(p)).items() if r.get("scores")}
        if recs:
            first = next(iter(recs.values()))
            runs.append((_rater_name(first.get("judge", p.stem), first.get("prompt", "?")), recs))
    runs.sort(key=lambda x: x[0].startswith("Heuristic"))
    human = {i: r for i, r in latest_by_id(load_jsonl(labels_path)).items() if r.get("scores")}

    items = []
    for n, t in enumerate(trajs):
        verdicts = []
        for name, recs in runs:
            r = recs.get(t.id)
            if r:
                verdicts.append({"rater": name, "scores": [r["scores"].get(d) for d in DIMENSIONS],
                                 "p": r.get("resolved_probability"),
                                 "why": r.get("rationale", "") if not name.startswith("Heuristic") else ""})
        h = human.get(t.id)
        if h:
            verdicts.append({"rater": "Human label", "scores": [h["scores"].get(d) for d in DIMENSIONS],
                             "p": h.get("resolved_probability"), "why": h.get("notes", ""), "human": True})
        items.append({"n": n + 1, "issue": t.instance_id, "passed": t.resolved, "verdicts": verdicts,
                      "transcript": render_transcript(t)})
    report = results_dir / "report.md"
    return {"dimensions": DIMENSIONS, "primary": runs[0][0] if runs else None, "items": items,
            "report": markdown_to_html(report.read_text(encoding="utf-8")) if report.exists() else ""}


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Judge Explorer</title>
<meta name="description" content="How well can an LLM grade an AI coding agent? Browse 500 real SWE-agent runs with judge verdicts, human labels and test outcomes.">
<style>
:root{--bg:#fafaf9;--panel:#fff;--text:#1c1917;--muted:#57534e;--line:#e7e5e4;--accent:#4f46e5;--good:#15803d;--bad:#b91c1c;--code:#f5f5f4}
@media (prefers-color-scheme:dark){:root{--bg:#0c0a09;--panel:#1c1917;--text:#f5f5f4;--muted:#a8a29e;--line:#292524;--accent:#a5b4fc;--good:#4ade80;--bad:#f87171;--code:#141210}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:1.7rem;margin:0 0 6px;letter-spacing:-.01em}
.lede{color:var(--muted);margin:0 0 20px;max-width:760px}
a{color:var(--accent)}
nav{display:flex;gap:4px;border-bottom:1px solid var(--line);margin-bottom:18px}
nav button{background:none;border:0;border-bottom:2px solid transparent;padding:8px 12px;font:inherit;color:var(--muted);cursor:pointer}
nav button[aria-selected=true]{color:var(--text);border-color:var(--accent);font-weight:600}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:end;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:16px}
.controls label{display:flex;flex-direction:column;gap:4px;font-size:.85rem;color:var(--muted)}
select{font:inherit;padding:7px 8px;border:1px solid var(--line);border-radius:7px;background:var(--panel);color:var(--text);min-width:220px;max-width:100%}
.nav2{display:flex;gap:6px}
.nav2 button{font:inherit;padding:7px 12px;border:1px solid var(--line);border-radius:7px;background:var(--panel);color:var(--text);cursor:pointer}
.count{color:var(--muted);font-size:.85rem;margin-left:auto}
.outcome{font-weight:700;margin:4px 0 10px}
.outcome.pass{color:var(--good)}.outcome.fail{color:var(--bad)}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-size:.8rem;color:var(--muted);font-weight:600;white-space:nowrap}
th.num{text-align:center}
td:first-child{min-width:150px}
#report h2{font-size:1.15rem;margin:26px 0 8px}#report h3{font-size:1rem;margin:20px 0 6px}
#report .tablewrap{margin:10px 0 6px}#report td,#report th{white-space:nowrap}
code{font:12.5px ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--code);padding:1px 4px;border-radius:4px}
tr:last-child td{border-bottom:0}
td.num{font-variant-numeric:tabular-nums;text-align:center}
.ok{color:var(--good);font-weight:600}.no{color:var(--bad);font-weight:600}
.why{margin:12px 0 0;color:var(--text)}
.why b{color:var(--muted)}
pre{background:var(--code);border:1px solid var(--line);border-radius:10px;padding:14px;white-space:pre-wrap;word-break:break-word;font:12.5px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;max-height:70vh;overflow:auto}
h2{font-size:1.1rem;margin:22px 0 8px}
#report table{background:var(--panel)}
#report{overflow-x:auto}
footer{color:var(--muted);font-size:.85rem;margin-top:36px}
@media (max-width:600px){.controls label{width:100%}select{min-width:0;width:100%}.count{margin-left:0}}
</style>
</head>
<body>
<main>
<h1>How well can an LLM grade an AI coding agent?</h1>
<p class="lede">Each transcript is a real SWE-agent attempt at a GitHub issue, together with the hidden truth of whether its patch passed the repository's tests.
Compare a local <b>qwen2.5:14b</b> judge, a no-LLM heuristic, and human labels against that outcome.
<a href="https://github.com/krishnagiri1/agent-judge">Code and write-up on GitHub</a>.</p>
<nav role="tablist">
<button role="tab" aria-selected="true" data-tab="browse">Browse verdicts</button>
<button role="tab" aria-selected="false" data-tab="results">Results</button>
</nav>
<section id="browse">
<div class="controls">
<label>Show<select id="filter">
<option value="all">All transcripts</option>
<option value="human">Hand-labeled</option>
<option value="wrong">Judge was wrong</option>
<option value="right">Judge was right</option>
</select></label>
<label>Transcript<select id="pick"></select></label>
<div class="nav2"><button id="prev" aria-label="Previous">‹ Prev</button><button id="next" aria-label="Next">Next ›</button></div>
<span class="count" id="count"></span>
</div>
<div id="verdicts"></div>
<h2>Transcript (exactly what the judge saw)</h2>
<pre id="transcript"></pre>
</section>
<section id="results" hidden><div id="report"></div></section>
<footer>Data: nebius/SWE-agent-trajectories (CC-BY-4.0). Built by Venkata Krishna Giri Koneru.</footer>
</main>
<script src="data.js"></script>
<script>
const D = window.AGENT_JUDGE_DATA, dims = D.dimensions;
const $ = id => document.getElementById(id);
const primaryP = it => { const v = it.verdicts.find(v => v.rater === D.primary); return v ? v.p : null; };
const judgeRight = it => { const p = primaryP(it); return p == null || p === 50 ? null : (p > 50) === it.passed; };
let list = [];
function applyFilter() {
  const f = $("filter").value;
  list = D.items.filter(it => f === "all" || (f === "human" && it.verdicts.some(v => v.human)) ||
    (f === "wrong" && judgeRight(it) === false) || (f === "right" && judgeRight(it) === true));
  $("pick").innerHTML = list.map((it, i) => `<option value="${i}">#${String(it.n).padStart(3, "0")} · ${it.issue}</option>`).join("");
  $("count").textContent = `${list.length} transcript${list.length === 1 ? "" : "s"}`;
  show(0);
}
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function show(i) {
  if (!list.length) { $("verdicts").innerHTML = "<p>No transcripts match this filter.</p>"; $("transcript").textContent = ""; return; }
  i = Math.max(0, Math.min(list.length - 1, i)); $("pick").value = i;
  const it = list[i];
  let h = `<p class="outcome ${it.passed ? "pass" : "fail"}">Actual test outcome: ${it.passed ? "✓ the patch passed" : "✗ the patch failed"}</p>`;
  h += `<div class="tablewrap"><table><thead><tr><th>Rater</th>${dims.map(d => `<th class="num">${d.replace("_", " ")}</th>`).join("")}<th class="num">P(pass)</th><th class="num">Right?</th></tr></thead><tbody>`;
  for (const v of it.verdicts) {
    const right = v.p == null || v.p === 50 ? "–" : ((v.p > 50) === it.passed ? '<span class="ok">✓</span>' : '<span class="no">✗</span>');
    h += `<tr><td>${v.human ? "<b>Human label</b>" : esc(v.rater)}</td>${v.scores.map(s => `<td class="num">${s ?? "–"}</td>`).join("")}<td class="num">${v.p == null ? "–" : Math.round(v.p) + "%"}</td><td class="num">${right}</td></tr>`;
  }
  h += "</tbody></table></div>";
  for (const v of it.verdicts.filter(v => v.why)) h += `<p class="why"><b>${esc(v.rater)}:</b> ${esc(v.why)}</p>`;
  $("verdicts").innerHTML = h;
  $("transcript").textContent = it.transcript;
}
$("filter").onchange = applyFilter;
$("pick").onchange = e => show(+e.target.value);
$("prev").onclick = () => show(+$("pick").value - 1);
$("next").onclick = () => show(+$("pick").value + 1);
document.querySelectorAll("nav button").forEach(b => b.onclick = () => {
  document.querySelectorAll("nav button").forEach(x => x.setAttribute("aria-selected", x === b));
  $("browse").hidden = b.dataset.tab !== "browse"; $("results").hidden = b.dataset.tab !== "results";
});
$("report").innerHTML = D.report || "<p>No report yet.</p>";
applyFilter();
</script>
</body>
</html>
"""


def export_site(sample_path: str | Path, labels_path: str | Path, results_dir: str | Path,
                out_dir: str | Path = "docs") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = build_data(sample_path, labels_path, results_dir)
    (out / "data.js").write_text("window.AGENT_JUDGE_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n",
                                 encoding="utf-8")
    (out / "index.html").write_text(PAGE, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")  # serve files as-is on GitHub Pages
    return out / "index.html"
