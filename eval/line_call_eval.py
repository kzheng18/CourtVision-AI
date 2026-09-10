"""
Line-call evaluation for CourtVision-AI.

Turns "I think the in/out calls got better" into a number. You hand-label the
true IN/OUT call for a set of bounces from one match; this scores the pipeline's
predicted calls against them and writes a visual HTML report.

Workflow
--------
1. Run the pipeline and export its detected bounces:

       from eval.line_call_eval import export_predictions
       export_predictions(ball_bounce_frames, "eval/preds.json")

   (ball_bounce_frames is what detect_ball_bounces(...) returns.)

2. Make a label template from those predictions, then edit the "truth" field
   for each bounce by watching the video ("IN" or "OUT"):

       python eval/line_call_eval.py template eval/preds.json eval/labels.json

   You can also add bounces the system missed, and delete rows that were not
   real bounces.

3. Score and render the report:

       python eval/line_call_eval.py score eval/preds.json eval/labels.json \
           --html eval/report.html --title "Practice set — Aug 20"

OUT is treated as the positive class: an OUT call is the consequential one
(it ends the point), and calling a good ball OUT is the costliest error.
"""

import os
import json
import argparse

IN, OUT = "IN", "OUT"


# ---------------------------------------------------------------------------
# Export / labeling helpers
# ---------------------------------------------------------------------------

def export_predictions(ball_bounce_frames, out_path):
    """Dump detect_ball_bounces(...) output to a predictions JSON."""
    preds = []
    for b in ball_bounce_frames:
        preds.append({
            "frame": int(b["frame"]),
            "x_m": None if b.get("x_m") is None else round(float(b["x_m"]), 3),
            "y_m": None if b.get("y_m") is None else round(float(b["y_m"]), 3),
            "pred": OUT if not b["is_in_bounds"] else IN,
        })
    _write_json(out_path, preds)
    print(f"Wrote {len(preds)} predictions → {out_path}")
    return preds


def make_template(predictions_path, out_path):
    """Create a label template from predictions: copy calls into 'truth' to edit."""
    preds = _read_json(predictions_path)
    labels = [{"frame": p["frame"], "truth": p["pred"],
               "x_m": p.get("x_m"), "y_m": p.get("y_m")} for p in preds]
    _write_json(out_path, labels)
    print(f"Wrote label template with {len(labels)} rows → {out_path}\n"
          f"Edit each 'truth' to the real call (IN/OUT); add missed bounces, "
          f"remove non-bounces.")
    return labels


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _match(preds, labels, frame_tol):
    """
    Match predictions to labels by nearest frame within frame_tol.
    Returns (matched, missed, extra):
      matched : list of {frame, pred, truth, x_m, y_m}
      missed  : labels with no prediction (system missed a real bounce)
      extra   : predictions with no label (system saw a bounce that wasn't real)
    """
    used_pred = set()
    matched, missed = [], []
    for lab in labels:
        best, best_d = None, frame_tol + 1
        for i, p in enumerate(preds):
            if i in used_pred:
                continue
            d = abs(p["frame"] - lab["frame"])
            if d < best_d:
                best, best_d, best_i = p, d, i
        if best is not None:
            used_pred.add(best_i)
            matched.append({
                "frame": lab["frame"],
                "pred": best["pred"],
                "truth": lab["truth"],
                "x_m": lab.get("x_m", best.get("x_m")),
                "y_m": lab.get("y_m", best.get("y_m")),
            })
        else:
            missed.append(lab)
    extra = [p for i, p in enumerate(preds) if i not in used_pred]
    return matched, missed, extra


def _metrics(matched):
    """Confusion counts + accuracy and per-class precision/recall/F1."""
    tp = sum(1 for m in matched if m["truth"] == OUT and m["pred"] == OUT)
    fp = sum(1 for m in matched if m["truth"] == IN and m["pred"] == OUT)
    fn = sum(1 for m in matched if m["truth"] == OUT and m["pred"] == IN)
    tn = sum(1 for m in matched if m["truth"] == IN and m["pred"] == IN)
    total = max(1, tp + fp + fn + tn)

    def prf(p_tp, p_fp, p_fn):
        prec = p_tp / (p_tp + p_fp) if (p_tp + p_fp) else None
        rec = p_tp / (p_tp + p_fn) if (p_tp + p_fn) else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
        return prec, rec, f1

    out_p, out_r, out_f = prf(tp, fp, fn)
    in_p, in_r, in_f = prf(tn, fn, fp)  # IN as positive: TN acts as its TP
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "total": tp + fp + fn + tn,
        "accuracy": (tp + tn) / total,
        "out": {"precision": out_p, "recall": out_r, "f1": out_f},
        "in": {"precision": in_p, "recall": in_r, "f1": in_f},
    }


def score(predictions_path, labels_path, frame_tol=5):
    preds = _read_json(predictions_path)
    labels = _read_json(labels_path)
    matched, missed, extra = _match(preds, labels, frame_tol)
    m = _metrics(matched)
    m.update({
        "matched": matched,
        "missed": missed,
        "extra": extra,
        "n_labels": len(labels),
        "n_preds": len(preds),
        "frame_tol": frame_tol,
    })
    return m


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _read_json(p):
    with open(p) as f:
        return json.load(f)


def _write_json(p, obj):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_CSS = """
:root{
  --ground:#f4f6f2; --surface:#ffffff; --surface-2:#eef1eb; --sunk:#e7ebe4;
  --ink:#16201b; --muted:#5f6d64; --line:#dbe2d9;
  --accent:#2f6f4e; --in:#1f9d57; --out:#cf5433; --warn:#c2891d;
  --shadow:0 1px 2px rgba(20,40,30,.06),0 8px 24px rgba(20,40,30,.06);
}
:root:not([data-theme="light"]){}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0f1411; --surface:#171e19; --surface-2:#1d2620; --sunk:#131a15;
    --ink:#e9ede9; --muted:#93a498; --line:#29342c;
    --accent:#5cb684; --in:#37c47d; --out:#e77b56; --warn:#e0ab4a;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"]{
  --ground:#0f1411; --surface:#171e19; --surface-2:#1d2620; --sunk:#131a15;
  --ink:#e9ede9; --muted:#93a498; --line:#29342c;
  --accent:#5cb684; --in:#37c47d; --out:#e77b56; --warn:#e0ab4a;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"Spline Sans",system-ui,-apple-system,sans-serif;
  line-height:1.55; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:960px; margin:0 auto; padding:40px 24px 64px}
.eyebrow{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:12px;
  letter-spacing:.18em; text-transform:uppercase; color:var(--accent);
  margin:0 0 8px;
}
h1{
  font-family:"Archivo",system-ui,sans-serif; font-weight:800;
  font-size:clamp(28px,4.5vw,42px); line-height:1.05; letter-spacing:-.02em;
  text-wrap:balance; margin:0 0 6px;
}
.sub{color:var(--muted); margin:0 0 28px; font-size:15px}
.headline{
  display:flex; flex-wrap:wrap; align-items:baseline; gap:14px;
  padding:22px 24px; margin:0 0 28px; border-radius:16px;
  background:var(--surface); border:1px solid var(--line); box-shadow:var(--shadow);
}
.headline .big{
  font-family:"Archivo",sans-serif; font-weight:800; letter-spacing:-.03em;
  font-size:clamp(44px,9vw,72px); line-height:.9;
  font-variant-numeric:tabular-nums;
}
.headline .cap{font-size:14px; color:var(--muted); max-width:32ch}
.grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:14px; margin:0 0 28px}
.tile{
  background:var(--surface); border:1px solid var(--line); border-radius:14px;
  padding:16px 16px 14px; box-shadow:var(--shadow); position:relative; overflow:hidden;
}
.tile .k{font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.1em;
  text-transform:uppercase; color:var(--muted); margin:0 0 8px}
.tile .v{font-family:"Archivo",sans-serif; font-weight:700; font-size:30px;
  font-variant-numeric:tabular-nums; line-height:1}
.tile .u{font-size:13px; color:var(--muted); margin-top:4px}
.tile.stripe::before{content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--accent)}
.tile.good::before{background:var(--in)} .tile.bad::before{background:var(--out)} .tile.warn::before{background:var(--warn)}
.section-h{font-family:"Archivo",sans-serif; font-weight:700; font-size:19px;
  letter-spacing:-.01em; margin:36px 0 14px}
.cols{display:grid; grid-template-columns:1fr 1fr; gap:24px; align-items:start}
@media(max-width:720px){.cols{grid-template-columns:1fr}}
.panel{background:var(--surface); border:1px solid var(--line); border-radius:16px;
  padding:20px; box-shadow:var(--shadow)}
.cm{display:grid; grid-template-columns:auto 1fr 1fr; gap:6px; font-size:14px}
.cm .cell{border-radius:10px; padding:12px 10px; text-align:center; background:var(--surface-2)}
.cm .hd{font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); display:flex; align-items:center; justify-content:center}
.cm .n{font-family:"Archivo",sans-serif; font-weight:700; font-size:26px; font-variant-numeric:tabular-nums}
.cm .lbl{font-size:11px; color:var(--muted); margin-top:2px}
.cm .correct{background:color-mix(in srgb,var(--in) 16%,var(--surface))}
.cm .error{background:color-mix(in srgb,var(--out) 16%,var(--surface))}
table{width:100%; border-collapse:collapse; font-size:14px}
th{font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); text-align:left; font-weight:500;
  padding:8px 10px; border-bottom:1px solid var(--line)}
td{padding:9px 10px; border-bottom:1px solid var(--line); font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
.pill{display:inline-block; font-family:"IBM Plex Mono",monospace; font-size:11px;
  font-weight:600; letter-spacing:.04em; padding:2px 9px; border-radius:999px}
.pill.in{color:var(--in); background:color-mix(in srgb,var(--in) 16%,transparent)}
.pill.out{color:var(--out); background:color-mix(in srgb,var(--out) 16%,transparent)}
.legend{display:flex; flex-wrap:wrap; gap:16px; font-size:13px; color:var(--muted); margin-top:14px}
.legend span{display:inline-flex; align-items:center; gap:7px}
.dot{width:11px; height:11px; border-radius:50%; display:inline-block}
.court-wrap{display:flex; justify-content:center}
svg.court{width:100%; max-width:280px; height:auto}
.empty{color:var(--muted); font-size:14px; padding:6px 2px}
.foot{margin-top:40px; padding-top:18px; border-top:1px solid var(--line);
  color:var(--muted); font-size:12.5px; font-family:"IBM Plex Mono",monospace}
"""


def _fmt_pct(x):
    return "—" if x is None else f"{x*100:.0f}%"


def _court_svg(matched):
    """Top-down singles-court plot of bounces. Dot color = true call; a ring
    marks calls the system got wrong."""
    W, L = 10.97, 23.77           # doubles court meters
    SL, SR = 1.37, 9.60           # singles sidelines
    NET, SVC = 11.885, 6.40       # net y, service line offset
    S = 17.0                       # px per meter
    pad = 16
    w = W * S + 2 * pad
    h = L * S + 2 * pad
    def X(m): return pad + m * S
    def Y(m): return pad + m * S
    e = []
    e.append(f'<svg class="court" viewBox="0 0 {w:.0f} {h:.0f}" '
             f'xmlns="http://www.w3.org/2000/svg" role="img" '
             f'aria-label="Court plot of bounce calls">')
    e.append(f'<rect x="{pad}" y="{pad}" width="{W*S:.0f}" height="{L*S:.0f}" '
             f'rx="3" fill="color-mix(in srgb,var(--accent) 8%,var(--surface))" '
             f'stroke="var(--line)"/>')
    ln = 'stroke="var(--muted)" stroke-width="1.2" opacity=".55"'
    # singles sidelines
    e.append(f'<line x1="{X(SL):.1f}" y1="{Y(0):.1f}" x2="{X(SL):.1f}" y2="{Y(L):.1f}" {ln}/>')
    e.append(f'<line x1="{X(SR):.1f}" y1="{Y(0):.1f}" x2="{X(SR):.1f}" y2="{Y(L):.1f}" {ln}/>')
    # service lines + center service line
    e.append(f'<line x1="{X(SL):.1f}" y1="{Y(NET-SVC):.1f}" x2="{X(SR):.1f}" y2="{Y(NET-SVC):.1f}" {ln}/>')
    e.append(f'<line x1="{X(SL):.1f}" y1="{Y(NET+SVC):.1f}" x2="{X(SR):.1f}" y2="{Y(NET+SVC):.1f}" {ln}/>')
    e.append(f'<line x1="{X((SL+SR)/2):.1f}" y1="{Y(NET-SVC):.1f}" x2="{X((SL+SR)/2):.1f}" y2="{Y(NET+SVC):.1f}" {ln}/>')
    # net
    e.append(f'<line x1="{pad}" y1="{Y(NET):.1f}" x2="{w-pad:.0f}" y2="{Y(NET):.1f}" '
             f'stroke="var(--ink)" stroke-width="2" opacity=".75"/>')
    # bounces
    for m in matched:
        if m.get("x_m") is None or m.get("y_m") is None:
            continue
        cx, cy = X(m["x_m"]), Y(m["y_m"])
        col = "var(--in)" if m["truth"] == IN else "var(--out)"
        wrong = m["pred"] != m["truth"]
        if wrong:
            e.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8.5" fill="none" '
                     f'stroke="var(--warn)" stroke-width="2.5"/>')
        e.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{col}" '
                 f'stroke="var(--surface)" stroke-width="1.5"/>')
    e.append('</svg>')
    return "\n".join(e)


def render_html(results, out_path, title="Line-call evaluation"):
    r = results
    acc = r["accuracy"]
    matched = r["matched"]
    wrong = [m for m in matched if m["pred"] != m["truth"]]

    tiles = [
        ("Bounces scored", str(r["total"]), "matched to labels", "stripe"),
        ("OUT precision", _fmt_pct(r["out"]["precision"]), "of OUT calls were right",
         "bad" if (r["out"]["precision"] or 1) < 0.9 else "good"),
        ("OUT recall", _fmt_pct(r["out"]["recall"]), "of real OUTs were caught",
         "bad" if (r["out"]["recall"] or 1) < 0.9 else "good"),
        ("Missed bounces", str(len(r["missed"])), "real bounces not detected",
         "warn" if r["missed"] else "good"),
        ("Phantom bounces", str(len(r["extra"])), "detected, not real",
         "warn" if r["extra"] else "good"),
    ]
    tiles_html = "\n".join(
        f'<div class="tile stripe {cls}"><p class="k">{k}</p>'
        f'<div class="v">{v}</div><div class="u">{u}</div></div>'
        for k, v, u, cls in tiles
    )

    cm = f"""
    <div class="cm">
      <div class="hd"></div><div class="hd">called IN</div><div class="hd">called OUT</div>
      <div class="hd">truly IN</div>
      <div class="cell correct"><div class="n">{r['tn']}</div><div class="lbl">correct</div></div>
      <div class="cell error"><div class="n">{r['fp']}</div><div class="lbl">good ball called out</div></div>
      <div class="hd">truly OUT</div>
      <div class="cell error"><div class="n">{r['fn']}</div><div class="lbl">out ball missed</div></div>
      <div class="cell correct"><div class="n">{r['tp']}</div><div class="lbl">correct</div></div>
    </div>"""

    if wrong:
        row_html = []
        for m in sorted(wrong, key=lambda z: z["frame"]):
            xy = "—" if m.get("x_m") is None else "{:.2f}, {:.2f}".format(m["x_m"], m["y_m"])
            pred_cls, truth_cls = m["pred"].lower(), m["truth"].lower()
            row_html.append(
                '<tr><td>{f}</td>'
                '<td><span class="pill {pc}">{p}</span></td>'
                '<td><span class="pill {tc}">{t}</span></td>'
                '<td>{xy}</td></tr>'.format(
                    f=m["frame"], pc=pred_cls, p=m["pred"],
                    tc=truth_cls, t=m["truth"], xy=xy)
            )
        rows = "\n".join(row_html)
        disagree = f"""<table>
      <thead><tr><th>Frame</th><th>Called</th><th>Truth</th><th>Court x,y (m)</th></tr></thead>
      <tbody>{rows}</tbody></table>"""
    else:
        disagree = '<p class="empty">No disagreements — every scored call matched the label.</p>'

    court = _court_svg(matched)

    html = f"""<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=Spline+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>{_CSS}</style>
<div class="wrap">
  <p class="eyebrow">CourtVision · line-call eval</p>
  <h1>{title}</h1>
  <p class="sub">Predicted IN/OUT calls scored against hand labels · {r['n_labels']} labels · {r['n_preds']} detections · ±{r['frame_tol']}-frame match</p>

  <div class="headline">
    <div class="big">{_fmt_pct(acc)}</div>
    <div class="cap"><strong>call accuracy</strong> on {r['total']} bounces matched to ground truth. OUT is scored as the positive class — calling a good ball out is the costliest miss.</div>
  </div>

  <div class="grid">{tiles_html}</div>

  <div class="cols">
    <div>
      <div class="section-h">Confusion matrix</div>
      <div class="panel">{cm}</div>
    </div>
    <div>
      <div class="section-h">Where they landed</div>
      <div class="panel">
        <div class="court-wrap">{court}</div>
        <div class="legend">
          <span><span class="dot" style="background:var(--in)"></span>true IN</span>
          <span><span class="dot" style="background:var(--out)"></span>true OUT</span>
          <span><span class="dot" style="background:var(--warn)"></span>ring = wrong call</span>
        </div>
      </div>
    </div>
  </div>

  <div class="section-h">Disagreements</div>
  <div class="panel">{disagree}</div>

  <p class="foot">Generated by eval/line_call_eval.py · accuracy = (correct IN + correct OUT) / scored · missed &amp; phantom bounces are detection errors, tracked separately from call accuracy.</p>
</div>"""

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Wrote report → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(r):
    print(f"\n  Call accuracy : {_fmt_pct(r['accuracy'])}  ({r['tp']+r['tn']}/{r['total']})")
    print(f"  OUT precision : {_fmt_pct(r['out']['precision'])}   OUT recall: {_fmt_pct(r['out']['recall'])}")
    print(f"  IN  precision : {_fmt_pct(r['in']['precision'])}   IN  recall: {_fmt_pct(r['in']['recall'])}")
    print(f"  Confusion     : TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']}")
    print(f"  Missed bounces: {len(r['missed'])}   Phantom: {len(r['extra'])}\n")


def main():
    ap = argparse.ArgumentParser(description="Line-call evaluation for CourtVision-AI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("template", help="make a label template from predictions")
    t.add_argument("predictions"); t.add_argument("out")

    s = sub.add_parser("score", help="score predictions against labels")
    s.add_argument("predictions"); s.add_argument("labels")
    s.add_argument("--frame-tol", type=int, default=5)
    s.add_argument("--html", default=None)
    s.add_argument("--title", default="Line-call evaluation")

    a = ap.parse_args()
    if a.cmd == "template":
        make_template(a.predictions, a.out)
    elif a.cmd == "score":
        r = score(a.predictions, a.labels, frame_tol=a.frame_tol)
        _print_summary(r)
        if a.html:
            render_html(r, a.html, title=a.title)


if __name__ == "__main__":
    main()
