"""Assemble media/level2/m4-slides.html and m4-explore.html (Quantum Computing Intermediate, Module 4: running on real
hardware) from the Level 1 deck frame (media/level1/m1-slides.html), tools/l2_m4_slides_body.html,
tools/l2_m4_explore_src.html and the saved runs content/level2/m4_saved_runs.json (every number on the slides and in
the explorer comes from that file)."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "media/level2"
OUT.mkdir(parents=True, exist_ok=True)
MAROON, GREY, INK, GOLD, MUTED, OK = "#99004C", "#B9AEB4", "#24313D", "#C9A227", "#6A626B", "#2E7D4F"
SAVED = json.loads((ROOT / "content/level2/m4_saved_runs.json").read_text())
DEV, TEACH = SAVED["device"], SAVED["teach"]
EDGES = [tuple(e) for e in DEV["edges"]]
NB = {q: set() for q in range(DEV["num_qubits"])}
for a, b in EDGES:
    NB[a].add(b); NB[b].add(a)


def pos(q):
    """Heron layout: blocks of 20 qubits, 16 in a row and 4 bridge qubits below it."""
    k, i = divmod(q, 20)
    if i < 16:
        return i, 2 * k
    up = min(n for n in NB[q] if n < q)
    return pos(up)[0], 2 * k + 1


def lattice(qubits, highlight=(), w=600, h=300, sx=None, label=True, aria=""):
    qs = [q for q in qubits]
    xs = [pos(q)[0] for q in qs]; ys = [pos(q)[1] for q in qs]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    m = 26
    sx = sx or (w - 2 * m) / max(1, x1 - x0)
    sy = (h - 2 * m) / max(1, y1 - y0)
    P = lambda q: (m + (pos(q)[0] - x0) * sx, m + (pos(q)[1] - y0) * sy)
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="{aria}">']
    S = set(qs)
    for a, b in EDGES:
        if a in S and b in S:
            (ax, ay), (bx, by) = P(a), P(b)
            hl = a in highlight and b in highlight
            p.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{MAROON if hl else GREY}" stroke-width="{5 if hl else 3}"/>')
    r = min(12, sx * 0.27)
    for q in qs:
        x, y = P(q)
        hl = q in highlight
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r + (1.5 if hl else 0):.1f}" fill="{MAROON if hl else "#fff"}" stroke="{MAROON if hl else INK}" stroke-width="2"/>')
        if label and (hl or r >= 12):
            p.append(f'<text x="{x:.1f}" y="{y + 4:.1f}" font-size="{11 if q >= 100 else 12}" text-anchor="middle" fill="{"#fff" if hl else INK}">{q}</text>')
    p.append('</svg>')
    return ''.join(p)


def stages(w=1130, h=150):
    names = ["init", "layout", "routing", "translation", "optimization", "scheduling"]
    sub = ["prepare, split multi-qubit gates", "pick physical qubits", "add SWAPs", "rewrite in native gates", "cancel, merge, resynthesize", "timing (optional)"]
    bw, gap = 160, 34
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="The six transpiler stages: init, layout, routing, translation, optimization, scheduling">',
         f'<defs><marker id="ar" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{MAROON}"/></marker></defs>']
    for i, (n, s) in enumerate(zip(names, sub)):
        x = 6 + i * (bw + gap)
        p.append(f'<rect x="{x}" y="20" width="{bw}" height="70" rx="8" fill="#F8F4F6" stroke="{MAROON}" stroke-width="2"/>')
        p.append(f'<text x="{x + bw / 2}" y="62" font-size="22" text-anchor="middle" fill="{MAROON}">{n}</text>')
        p.append(f'<text x="{x + bw / 2}" y="118" font-size="15" text-anchor="middle" fill="{MUTED}">{s}</text>')
        if i < 5:
            p.append(f'<line x1="{x + bw + 3}" y1="55" x2="{x + bw + gap - 5}" y2="55" stroke="{MAROON}" stroke-width="3" marker-end="url(#ar)"/>')
    p.append('</svg>')
    return ''.join(p)


def bars(groups, series, colors, names, w=520, h=330, top=1.0, label_fmt="{:.2f}", xlabel=""):
    """Grouped bar chart: groups = x labels, series = list of value lists."""
    x0, x1, yb, yt = 50, w - 10, h - 70, 20
    gw = (x1 - x0) / len(groups)
    bw = gw * 0.8 / len(series)
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="{xlabel}">']
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        if v > top: continue
        y = yb - (yb - yt) * v / top
        p.append(f'<line x1="{x0 - 4}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{INK if v == 0 else "#E4DCE0"}" stroke-width="{2 if v == 0 else 1}"/>')
        p.append(f'<text x="{x0 - 8}" y="{y + 5:.1f}" font-size="15" text-anchor="end" fill="{INK}">{v:g}</text>')
    for g, lab in enumerate(groups):
        for s, vals in enumerate(series):
            v = vals[g]
            hh = (yb - yt) * v / top
            x = x0 + g * gw + gw * 0.1 + s * bw
            p.append(f'<rect x="{x:.1f}" y="{yb - hh:.1f}" width="{bw * 0.92:.1f}" height="{hh:.1f}" fill="{colors[s]}"/>')
            p.append(f'<text x="{x + bw * 0.46:.1f}" y="{yb - hh - 5:.1f}" font-size="13" text-anchor="middle" fill="{INK}">{label_fmt.format(v)}</text>')
        p.append(f'<text x="{x0 + g * gw + gw / 2:.1f}" y="{yb + 20}" font-size="16" text-anchor="middle" fill="{INK}">{lab}</text>')
    lx = x0
    for s, n in enumerate(names):
        p.append(f'<rect x="{lx}" y="{h - 26}" width="14" height="14" fill="{colors[s]}"/><text x="{lx + 20}" y="{h - 14}" font-size="15" fill="{INK}">{n}</text>')
        lx += 30 + 8 * len(n)
    p.append(f'<text x="{x1}" y="{yb + 40}" font-size="14" text-anchor="end" fill="{MUTED}">{xlabel}</text>')
    p.append('</svg>')
    return ''.join(p)


def sr(counts, w="101"):
    return counts.get(w, 0) / sum(counts.values())


ideal = TEACH["ideal"]["2"]
R2 = TEACH["runs"]["2"]
rows = ["<table style=\"font-size:21px\"><tr><th>level</th><th>depth</th><th>gates</th><th>cz</th><th>rz</th><th>sx</th><th>x</th><th>final physical qubits</th></tr>"]
for l in "0123":
    r = R2[l]; o = r["ops"]
    rows.append(f"<tr><td>{l}</td><td>{r['depth']}</td><td>{r['size']}</td><td><strong>{r['two_qubit']}</strong></td><td>{o.get('rz', 0)}</td><td>{o.get('sx', 0)}</td><td>{o.get('x', 0)}</td><td>{r['layout']}</td></tr>")
TABLE_LEVELS = "".join(rows) + "</table>"
TABLE_DEVICE = ("<table style=\"font-size:21px\">"
                f"<tr><th>cz error</th><td>{DEV['median_cz_error']:.5f}</td></tr>"
                f"<tr><th>sx error</th><td>{DEV['median_sx_error']:.5f}</td></tr>"
                f"<tr><th>readout error</th><td>{DEV['median_readout_error']:.4f}</td></tr>"
                f"<tr><th>T1</th><td>{DEV['median_t1_us']:.0f} µs</td></tr>"
                f"<tr><th>T2</th><td>{DEV['median_t2_us']:.0f} µs</td></tr></table>")
est_med, est_act, noisy = [], [], []
for l in "0123":
    r = R2[l]
    est_med.append(ideal * (1 - DEV["median_cz_error"]) ** r["two_qubit"] * (1 - DEV["median_readout_error"]) ** 3)
    est_act.append(ideal * math.prod(1 - e for e in r["cz_errors"]) * math.prod(1 - e for e in r["readout_errors"]))
    noisy.append(sr(r["counts"]))
rows = ["<table style=\"font-size:21px\"><tr><th>level</th><th>cz</th><th>estimate (median errors)</th><th>estimate (actual qubits)</th><th>noise model</th></tr>"]
for i, l in enumerate("0123"):
    rows.append(f"<tr><td>{l}</td><td>{R2[l]['two_qubit']}</td><td>{est_med[i]:.3f}</td><td>{est_act[i]:.3f}</td><td><strong>{noisy[i]:.3f}</strong></td></tr>")
TABLE_BUDGET = "".join(rows) + f"</table><p class=\"small\">Ideal (no noise): {ideal:.4f}.</p>"
rows = ["<table style=\"font-size:21px\"><tr><th>iterations</th><th>cz</th><th>ideal</th><th>noise model</th></tr>"]
it_ideal, it_noisy = [], []
for t in "0123":
    r = TEACH["runs"][t]["3"]
    it_ideal.append(TEACH["ideal"][t]); it_noisy.append(sr(r["counts"]))
    rows.append(f"<tr><td>{t}</td><td>{r['two_qubit']}</td><td>{TEACH['ideal'][t]:.4f}</td><td><strong>{sr(r['counts']):.4f}</strong></td></tr>")
TABLE_ITER = "".join(rows) + "</table>"

# the numbers quoted in the slide text must match the saved runs
assert (R2["0"]["two_qubit"], R2["3"]["two_qubit"], R2["0"]["depth"], R2["3"]["depth"]) == (51, 37, 313, 134)
assert R2["0"]["layout"] == [1, 0, 2] and R2["3"]["layout"] == [113, 114, 119]
assert R2["2"]["ops"] == R2["3"]["ops"] and noisy[3] > noisy[2]
assert TEACH["logical_two_qubit_level_none"]["ops"] == {"h": 23, "x": 16, "ccx": 4, "measure": 3}
assert round(est_med[3], 3) == 0.881 and round(DEV["median_cz_error"], 5) == 0.00155 and round(DEV["median_readout_error"], 4) == 0.0045
assert {113, 114} <= NB[113] | {113} and 119 in NB[113] and max(len(v) for v in NB.values()) == 3
assert [round(x, 3) for x in (it_ideal[1], it_noisy[1], it_ideal[2], it_noisy[2])] == [0.781, 0.752, 0.945, 0.892]

m1 = (ROOT / "media/level1/m1-slides.html").read_text().split("\n")
head = "\n".join(m1[:56]).replace("<title>Module 1 Slides: From Bits to Qubits</title>",
                                  "<title>Module 4 Slides: Running on Real Hardware</title>")
assert "Module 4" in head
head = head.replace("  svg text { font-family:",
                    "  .slide code { font-family: Consolas, \"Courier New\", monospace; font-size:0.9em; }\n  svg text { font-family:")
body = (ROOT / "tools/l2_m4_slides_body.html").read_text()
title_q = [q for q in range(80, 140) if 8 <= pos(q)[0] <= 15]
body = body.replace("/*LATTICE_TITLE*/", lattice(title_q, highlight=(113, 114, 119), w=440, h=300,
                                                       aria="Part of the heavy-hex lattice of FakePittsburgh, with qubits 113, 114 and 119 highlighted"))
body = body.replace("/*LATTICE_TOP*/", lattice(range(0, 60), highlight=(0, 1, 2), w=640, h=330, label=False,
                                                     aria="Qubits 0 to 59 of FakePittsburgh: three rows of 16 qubits joined by bridge qubits; qubits 0, 1 and 2 highlighted"))
body = body.replace("/*STAGES*/", stages())
body = body.replace("/*NQ*/", str(DEV["num_qubits"])).replace("/*PAIRS*/", str(DEV["coupled_pairs"]))
body = body.replace("/*TABLE_LEVELS*/", TABLE_LEVELS).replace("/*TABLE_DEVICE*/", TABLE_DEVICE)
body = body.replace("/*TABLE_BUDGET*/", TABLE_BUDGET).replace("/*TABLE_ITER*/", TABLE_ITER)
body = body.replace("/*SR3*/", f"{noisy[3]:.3f}").replace("/*C3*/", f"{R2['3']['counts']['101']:,}").replace("/*ACT3*/", f"{est_act[3]:.3f}")
body = body.replace("/*BARS*/", bars(["level 0", "level 1", "level 2", "level 3"], [est_act, noisy], [GREY, MAROON],
                                     ["estimate (actual qubits)", "noise model"], xlabel="success rate for 101, 2 iterations"))
body = body.replace("/*ITERBARS*/", bars(["0", "1", "2", "3"], [it_ideal, it_noisy], [GREY, MAROON], ["ideal", "noise model"],
                                         xlabel="P(101) against the number of iterations"))
bar = "\n".join(m1[365:373]).replace("1 / 19", "1 / 18")
ctrl = "\n".join(m1[393:])
out = head + "\n" + body + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
assert "`" not in out and "${" not in out and "—" not in out and "/*" not in body
(OUT / "m4-slides.html").write_text(out)
print("wrote media/level2/m4-slides.html", len(out))

src = ROOT / "tools/l2_m4_explore_src.html"
if src.exists():
    keep = {"device": {k: DEV[k] for k in ("name", "num_qubits", "median_cz_error", "median_readout_error")},
            "teach": {"ideal": TEACH["ideal"],
                      "runs": {t: {l: {k: TEACH["runs"][t][l][k] for k in ("depth", "two_qubit", "layout", "cz_errors", "readout_errors")}
                                   | {"success": sr(TEACH["runs"][t][l]["counts"])} for l in "0123"} for t in "0123"}}}
    exp = src.read_text().replace("/*DATA*/", json.dumps(keep, separators=(",", ":")))
    assert "`" not in exp and "${" not in exp and "—" not in exp and "/*DATA*/" not in exp
    (OUT / "m4-explore.html").write_text(exp)
    print("wrote media/level2/m4-explore.html", len(exp))
