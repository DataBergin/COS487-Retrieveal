"""Build a self-contained annotation page for the Part 2 relevance judgments.

For every Test query it shows the model's TOP paragraph (rank 1 of the given run),
you press 0 / 1 / 2, and the page exports <team>_qrel_part2_test.tsv in the required
format:  qid <tab> 0 <tab> paragraph_id <tab> relevance

The page deliberately hides the model, rank and score so judgments are made on the
text alone. Progress autosaves in the browser (localStorage), keyed by the run file.

    python -m apps.neural.reporting.build_annotation data/runs/JackSearchParty_paragraph_dense_test.tsv
"""
import argparse
import json
import os

from ..common import PARAGRAPHS_TSV, ROOT, load_queries

PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Relevance annotation</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body{font:16px/1.55 -apple-system,"Segoe UI",Roboto,sans-serif;background:#fafbfd;color:#1f2430;margin:0}
 main{max-width:900px;margin:0 auto;padding:20px}
 .bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
 .prog{flex:1;min-width:200px;height:10px;background:#e4e8f0;border-radius:6px;overflow:hidden}
 .prog>div{height:100%;background:#2b5fd9}
 .card{background:#fff;border:1px solid #dde1ea;border-radius:10px;padding:16px 18px;margin:12px 0}
 .lbl{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:#5b6475;margin-bottom:4px}
 .q{font-size:19px;font-weight:600}
 .p{white-space:pre-wrap;max-height:46vh;overflow:auto}
 button{font:inherit;padding:10px 18px;border-radius:8px;border:1px solid #c8cedb;background:#fff;cursor:pointer}
 button.r{min-width:150px;text-align:left}
 button.sel{background:#2b5fd9;color:#fff;border-color:#2b5fd9}
 .row{display:flex;gap:10px;flex-wrap:wrap}
 .rubric{font-size:14px;color:#5b6475}
 .small{font-size:13px;color:#5b6475}
 input{font:inherit;padding:6px 8px;width:80px}
</style></head><body><main>
<div class="bar"><b id="count"></b><div class="prog"><div id="bar" style="width:0"></div></div>
 <span class="small" id="done"></span></div>
<div class="rubric"><b>2</b> fully addresses the query &middot; <b>1</b> useful context but does not fully address it &middot; <b>0</b> no useful information.
 Keys: <b>0 / 1 / 2</b> to judge, <b>&larr; &rarr;</b> to move.</div>
<div class="card"><div class="lbl">Query <span id="qid"></span></div><div class="q" id="query"></div></div>
<div class="card"><div class="lbl">Retrieved paragraph <span id="pid"></span></div><div class="p" id="para"></div></div>
<div class="row" id="btns"></div>
<div class="bar" style="margin-top:16px">
 <button id="prev">&larr; Back</button><button id="next">Next &rarr;</button>
 <span class="small">Jump to #</span><input id="jump" type="number" min="1">
 <button id="skip">Next unjudged</button>
 <button id="export" style="margin-left:auto">Export TSV</button>
</div>
<div class="small" id="msg"></div>
<script>
const DATA = __DATA__;
const TEAM = "__TEAM__";
const KEY = "annot::" + "__RUNKEY__";
let judg = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0; const el = id => document.getElementById(id);
const labels = {2:"2  Fully relevant", 1:"1  Partially relevant", 0:"0  Non-relevant"};
[2,1,0].forEach(v => { const b=document.createElement("button"); b.className="r"; b.id="b"+v; b.textContent=labels[v];
  b.onclick=()=>set(v); el("btns").appendChild(b); });
function render(){
  const d = DATA[i]; el("count").textContent = (i+1)+" / "+DATA.length;
  el("qid").textContent="("+d.qid+")"; el("query").textContent=d.query;
  el("pid").textContent="("+d.pid+")"; el("para").textContent=d.text;
  const n = Object.keys(judg).length; el("done").textContent = n+" judged";
  el("bar").style.width = (100*n/DATA.length)+"%";
  [2,1,0].forEach(v => el("b"+v).classList.toggle("sel", judg[d.qid]===v));
}
function save(){ localStorage.setItem(KEY, JSON.stringify(judg)); }
function set(v){ judg[DATA[i].qid]=v; save(); if(i<DATA.length-1) i++; render(); }
function go(k){ i=Math.max(0,Math.min(DATA.length-1,k)); render(); }
el("prev").onclick=()=>go(i-1); el("next").onclick=()=>go(i+1);
el("jump").onchange=e=>go(parseInt(e.target.value)-1);
el("skip").onclick=()=>{ for(let k=1;k<=DATA.length;k++){ const j=(i+k)%DATA.length; if(judg[DATA[j].qid]===undefined){go(j);return;} } el("msg").textContent="All queries judged."; };
document.addEventListener("keydown",e=>{ if(e.target.tagName==="INPUT")return;
  if(e.key==="0"||e.key==="1"||e.key==="2") set(parseInt(e.key));
  else if(e.key==="ArrowRight") go(i+1); else if(e.key==="ArrowLeft") go(i-1); });
el("export").onclick=()=>{
  const missing = DATA.filter(d=>judg[d.qid]===undefined).length;
  if(missing && !confirm(missing+" queries are not judged yet and will be left out. Export anyway?")) return;
  const lines = DATA.filter(d=>judg[d.qid]!==undefined).map(d=>d.qid+"\t0\t"+d.pid+"\t"+judg[d.qid]);
  const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([lines.join("\n")+"\n"],{type:"text/tab-separated-values"}));
  a.download=TEAM+"_qrel_part2_test.tsv"; a.click(); el("msg").textContent="Exported "+lines.length+" judgments.";
};
const firstUnjudged = DATA.findIndex(d=>judg[d.qid]===undefined); go(firstUnjudged<0?0:firstUnjudged);
</script></main></body></html>
"""


def top1(run_path):
    top = {}
    with open(run_path, encoding="utf-8") as f:
        for line in f:
            q, _, doc, rank = line.split()[:4]
            if rank == "1":
                top[q] = doc
    return top


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="paragraph-level TREC run on the Test queries")
    p.add_argument("--team", default="JackSearchParty")
    p.add_argument("--out", default="docs/annotation/annotate.html")
    a = p.parse_args()

    run = a.run if os.path.isabs(a.run) else os.path.join(ROOT, a.run)
    top = top1(run)
    qids, qtexts = load_queries("test")
    need = set(top.values())
    text = {}
    with open(PARAGRAPHS_TSV, encoding="utf-8") as f:
        for line in f:
            pid, _, t = line.rstrip("\n").partition("\t")
            if pid in need:
                text[pid] = t
    data = [{"qid": q, "query": qt, "pid": top[q], "text": text.get(top[q], "[paragraph text missing]")}
            for q, qt in zip(qids, qtexts) if q in top]
    assert len(data) == len(qids), "run is missing top-1 results for some Test queries"
    html = (PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
            .replace("__TEAM__", a.team).replace("__RUNKEY__", os.path.basename(run)))
    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote %s with %d queries" % (out, len(data)))


if __name__ == "__main__":
    main()
