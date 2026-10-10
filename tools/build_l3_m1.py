"""Assemble media/level3/m1-slides.html and m1-explore.html (Quantum Computing Advanced, Module 1: noise and open
quantum systems) from the Level 1 deck frame (media/level1/m1-slides.html), tools/l3_m1_slides_body.html and
tools/l3_m1_explore_src.html. Every number on the slides is computed here with qsim (content/level3/qsim.py), the
reference solutions of checks/l3_m1_check.py and content/level3/l3_m1_device_data.json, the same way the lab
computes them (Steps 3, 5, 6, 7 and 8)."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.modules["qiskit"] = None                       # use qsim, as the browser lab does
sys.path[:0] = [str(ROOT / "content/level3"), str(ROOT / "checks")]
import qsim  # noqa: E402
from qsim import QuantumCircuit, Statevector, AerSimulator, NoiseModel, thermal_relaxation_error, ReadoutError  # noqa: E402
import l3_m1_check as ref  # noqa: E402

OUT = ROOT / "media/level3"
OUT.mkdir(parents=True, exist_ok=True)
MAROON, GREY, INK, GOLD, MUTED, OK, PURPLE = "#99004C", "#B9AEB4", "#24313D", "#C9A227", "#6A626B", "#2E7D4F", "#6E2C4F"
X, Y, Z = ref.X, ref.Y, ref.Z
dm, bv = ref.reference_density_matrix, ref.reference_bloch_vector

# ---------------------------------------------------------------- the numbers
# Step 3: the mixture of |0> and |+>
plus = np.array([1, 1]) / math.sqrt(2)
rho_p = 0.75 * dm([1, 0]) + 0.25 * dm(plus)
r_p = bv(rho_p)
pur_p = float(np.trace(rho_p @ rho_p).real)
assert np.allclose(r_p, [0.25, 0, 0.75]) and abs(pur_p - 0.8125) < 1e-12 and abs(pur_p - (1 + r_p @ r_p) / 2) < 1e-12

# Steps 5 and 6: T1 = 100, T2 = 80
T1, T2, DT = 100.0, 80.0, 10.0
nm = NoiseModel()
nm.add_all_qubit_quantum_error(thermal_relaxation_error(T1, T2, DT), ["id"])
sim = AerSimulator(noise_model=nm)
waits = np.arange(0, 31, 2)
p1m = []
for k in waits:
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    for _ in range(int(k)):
        qc.id(0)
    qc.measure(0, 0)
    p1m.append(sim.run(qc, shots=1000, seed_simulator=int(100 + k)).result().get_counts().get("1", 0) / 1000)
t_wait, p1m = waits * DT, np.array(p1m)
use = p1m > 0.05
slope, intercept = np.polyfit(t_wait[use], np.log(p1m[use]), 1)
T1_FIT = -1 / slope
assert round(T1_FIT, 1) == 101.3, T1_FIT

# Step 7: device data
dev = json.loads((ROOT / "content/level3/l3_m1_device_data.json").read_text())
t1 = np.array([q["t1_us"] for q in dev["qubits"]])
t2 = np.array([q["t2_us"] for q in dev["qubits"]])
odd = [q["qubit"] for q in dev["qubits"] if q["t2_us"] > 2 * q["t1_us"]]
assert odd == [36] and round(float(np.median(t1)), 2) == 298.95
N_SX = np.median(t1) * 1000 / dev["sx_duration_ns"]

# Step 8: tomography


def tomography(prep, shots, seed, noise_model=None):
    s = AerSimulator(noise_model=noise_model)
    r = []
    for k, basis in enumerate("XYZ"):
        qc = QuantumCircuit(1, 1)
        qc.compose(prep, inplace=True)
        if basis == "X":
            qc.h(0)
        elif basis == "Y":
            qc.sdg(0)
            qc.h(0)
        qc.measure(0, 0)
        c = s.run(qc, shots=shots, seed_simulator=seed + k).result().get_counts()
        r.append((c.get("0", 0) - c.get("1", 0)) / shots)
    return np.array(r)


prep = QuantumCircuit(1)
prep.ry(1.1, 0)
prep.rz(0.7, 0)
psi = Statevector(prep).data
r_true = bv(dm(psi))
r_est = tomography(prep, 1000, 11)
ro = NoiseModel()
ro.add_all_qubit_readout_error(ReadoutError([[0.97, 0.03], [0.03, 0.97]]))
r_ro = tomography(prep, 10000, 21, ro)
fid = lambda r: (1 + r_true @ r) / 2  # noqa: E731   <psi|rho|psi> for rho = (I + r.sigma)/2
assert round(float(np.linalg.norm(r_est)), 3) == 1.010 and round(fid(r_est), 4) == 1.0040 and round(float(np.linalg.norm(r_ro)), 3) == 0.941


# ---------------------------------------------------------------- drawings
def disc(w, h, cx, cy, R, aria):
    """An x-z slice of the Bloch ball: z up, x to the right."""
    P = lambda x, z: (cx + R * x, cy - R * z)  # noqa: E731
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="{aria}">',
         f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="#F8F4F6" stroke="{INK}" stroke-width="2.5"/>',
         f'<line x1="{cx - R}" y1="{cy}" x2="{cx + R}" y2="{cy}" stroke="{GREY}" stroke-width="1.5" stroke-dasharray="5 5"/>',
         f'<line x1="{cx}" y1="{cy - R}" x2="{cx}" y2="{cy + R}" stroke="{GREY}" stroke-width="1.5" stroke-dasharray="5 5"/>']
    return p, P


def dot(P, x, z, col, r=7):
    a, b = P(x, z)
    return f'<circle cx="{a:.1f}" cy="{b:.1f}" r="{r}" fill="{col}"/>'


def label(P, x, z, txt, dx=0, dy=0, fs=20, col=INK, anchor="middle"):
    a, b = P(x, z)
    return f'<text x="{a + dx:.1f}" y="{b + dy:.1f}" font-size="{fs}" text-anchor="{anchor}" fill="{col}">{txt}</text>'


def relax_path(x0, z0, T1_, T2_, tmax, n=80):
    ts = np.linspace(0, tmax, n)
    return [(x0 * math.exp(-t / T2_), 1 - (1 - z0) * math.exp(-t / T1_)) for t in ts]


def title_disc():
    p, P = disc(400, 400, 200, 200, 165, "A state relaxing inside the Bloch ball: it spirals in from the surface toward the top, the state |0⟩")
    th = 2.2
    path = relax_path(math.sin(th), math.cos(th), T1, T2, 400)
    pts = " ".join("%.1f,%.1f" % P(x, z) for x, z in path)
    p.append(f'<polyline points="{pts}" fill="none" stroke="{MAROON}" stroke-width="4"/>')
    for t in (0, 50, 100, 200):
        x, z = math.sin(th) * math.exp(-t / T2), 1 - (1 - math.cos(th)) * math.exp(-t / T1)
        p.append(dot(P, x, z, MAROON, 8))
    p += [label(P, 0, 1, "|0⟩", dy=-14), label(P, 0, -1, "|1⟩", dy=32), label(P, 1, 0, "x", dx=18, dy=7, col=MUTED),
          dot(P, 0, 1, INK, 6), dot(P, 0, -1, INK, 6), '</svg>']
    return "".join(p)


def ball():
    p, P = disc(470, 470, 235, 235, 175, "The x-z slice of the Bloch ball with |0⟩, |1⟩, |+⟩, |−⟩ on the edge, I/2 at the center, and the mixture of |0⟩ and |+⟩ inside")
    p.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="2.5" stroke-dasharray="7 5"/>' % (*P(0, 1), *P(1, 0), MAROON))
    for (x, z), t, dx, dy in (((0, 1), "|0⟩", 0, -16), ((0, -1), "|1⟩", 0, 34), ((1, 0), "|+⟩", 34, 7), ((-1, 0), "|−⟩", -34, 7)):
        p += [dot(P, x, z, INK), label(P, x, z, t, dx, dy, 22)]
    p += [dot(P, 0, 0, MUTED, 8), label(P, 0, 0, "I/2", 0, 30, 20, MUTED)]
    p += [dot(P, r_p[0], r_p[2], MAROON, 10), label(P, r_p[0], r_p[2], "75% |0⟩, 25% |+⟩", -14, 24, 15, MAROON, "end")]
    p.append(label(P, 0, -1, "pure states: on the edge; mixed: inside", 0, 70, 16, MUTED))
    p[0] = p[0].replace('height="470"', 'height="490"').replace("0 0 470 470", "0 0 470 490")
    p.append('</svg>')
    return "".join(p)


def axes(w, h, x0, x1, yb, yt, xmax, xticks, ylab, xlab, ymin=0.0, ymax=1.0, yticks=(0, 0.25, 0.5, 0.75, 1)):
    Xf = lambda t: x0 + (x1 - x0) * t / xmax  # noqa: E731
    Yf = lambda v: yb - (yb - yt) * (v - ymin) / (ymax - ymin)  # noqa: E731
    p = []
    for v in yticks:
        p.append(f'<line x1="{x0}" y1="{Yf(v):.1f}" x2="{x1}" y2="{Yf(v):.1f}" stroke="#E4DCE0"/>'
                 f'<text x="{x0 - 8}" y="{Yf(v) + 5:.1f}" font-size="14" text-anchor="end" fill="{INK}">{v:g}</text>')
    p.append(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="{INK}" stroke-width="2"/>')
    for t in xticks:
        p.append(f'<text x="{Xf(t):.1f}" y="{yb + 20}" font-size="14" text-anchor="middle" fill="{INK}">{t:g}</text>')
    p.append(f'<text x="{(x0 + x1) / 2}" y="{h - 8}" font-size="15" text-anchor="middle" fill="{MUTED}">{xlab}</text>')
    p.append(f'<text x="16" y="{(yb + yt) / 2}" font-size="15" text-anchor="middle" fill="{MUTED}" transform="rotate(-90 16 {(yb + yt) / 2})">{ylab}</text>')
    return p, Xf, Yf


def poly(Xf, Yf, xs, ys, col, width=3, dash=""):
    pts = " ".join(f"{Xf(a):.1f},{Yf(b):.1f}" for a, b in zip(xs, ys))
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{width}"{d}/>'


def curves(w=560, h=380):
    p, Xf, Yf = axes(w, h, 70, w - 20, h - 60, 20, 400, range(0, 401, 100), "value", "time (microseconds)")
    ts = np.linspace(0, 400, 161)
    p.insert(0, f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="P(1) starting from |1⟩ falls as exp(−t/T1); ⟨X⟩ starting from |+⟩ falls as exp(−t/T2); T1 = 100, T2 = 80 microseconds">')
    p.append(poly(Xf, Yf, ts, np.exp(-ts / T1), MAROON, 3.5))
    p.append(poly(Xf, Yf, ts, np.exp(-ts / T2), INK, 3.5, "9 6"))
    p.append(f'<line x1="{Xf(T1):.1f}" y1="{Yf(0):.1f}" x2="{Xf(T1):.1f}" y2="{Yf(math.exp(-1)):.1f}" stroke="{GREY}" stroke-width="1.5"/>')
    p.append(f'<line x1="300" y1="97" x2="316" y2="97" stroke="{GREY}" stroke-width="2"/><text x="324" y="102" font-size="15">t = T1</text>')
    p.append(f'<rect x="300" y="40" width="16" height="5" fill="{MAROON}"/><text x="324" y="48" font-size="15">P(1) from |1⟩</text>')
    p.append(f'<line x1="300" y1="70" x2="316" y2="70" stroke="{INK}" stroke-width="3" stroke-dasharray="6 4"/><text x="324" y="75" font-size="15">⟨X⟩ from |+⟩</text>')
    p.append('</svg>')
    return "".join(p)


def t1fit(w=560, h=380):
    p, Xf, Yf = axes(w, h, 70, w - 20, h - 60, 20, 300, range(0, 301, 50), "P(1)", "wait (microseconds)")
    p.insert(0, f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="Measured P(1) after waits from 0 to 300 microseconds, the curve exp(−t/100) and the fitted curve">')
    ts = np.linspace(0, 300, 121)
    p.append(poly(Xf, Yf, ts, np.exp(-ts / T1), GREY, 3))
    p.append(poly(Xf, Yf, ts, np.exp(intercept + slope * ts), MAROON, 3, "9 6"))
    for t, v, u in zip(t_wait, p1m, use):
        p.append(f'<circle cx="{Xf(t):.1f}" cy="{Yf(v):.1f}" r="6" fill="{INK if u else "#fff"}" stroke="{INK}" stroke-width="2"/>')
    p.append(f'<circle cx="290" cy="45" r="6" fill="{INK}"/><text x="304" y="50" font-size="15">1,000 shots each</text>')
    p.append(f'<line x1="282" y1="72" x2="298" y2="72" stroke="{GREY}" stroke-width="3"/><text x="304" y="77" font-size="15">exp(−t / 100)</text>')
    p.append(f'<line x1="282" y1="99" x2="298" y2="99" stroke="{MAROON}" stroke-width="3" stroke-dasharray="6 4"/><text x="304" y="104" font-size="15">fit: T1 = {T1_FIT:.1f} µs</text>')
    if not use.all():
        p.append(f'<circle cx="290" cy="126" r="6" fill="#fff" stroke="{INK}" stroke-width="2"/><text x="304" y="131" font-size="15">below 0.05, not fitted</text>')
    p.append('</svg>')
    return "".join(p)


def device(w=560, h=400):
    x0, x1, yb, yt = 70, w - 20, h - 60, 20
    xmax, ymax = 500, 700
    Xf = lambda t: x0 + (x1 - x0) * t / xmax  # noqa: E731
    Yf = lambda v: yb - (yb - yt) * v / ymax  # noqa: E731
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-label="T2 against T1 for the {dev["num_qubits"]} qubits of FakePittsburgh, with the line T2 = 2 T1; one qubit lies above it">']
    for v in range(0, 701, 100):
        p.append(f'<line x1="{x0}" y1="{Yf(v):.1f}" x2="{x1}" y2="{Yf(v):.1f}" stroke="#E4DCE0"/>'
                 f'<text x="{x0 - 8}" y="{Yf(v) + 5:.1f}" font-size="14" text-anchor="end">{v}</text>')
    for v in range(0, 501, 100):
        p.append(f'<text x="{Xf(v):.1f}" y="{yb + 20}" font-size="14" text-anchor="middle">{v}</text>')
    p.append(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="{INK}" stroke-width="2"/>')
    p.append(f'<line x1="{Xf(0):.1f}" y1="{Yf(0):.1f}" x2="{Xf(350):.1f}" y2="{Yf(700):.1f}" stroke="{INK}" stroke-width="2" stroke-dasharray="7 5"/>')
    p.append(f'<text x="{Xf(318):.1f}" y="{Yf(680):.1f}" font-size="15" text-anchor="end">T2 = 2 T1</text>')
    for a, b, q in zip(t1, t2, dev["qubits"]):
        bad = q["qubit"] in odd
        p.append(f'<circle cx="{Xf(min(a, xmax)):.1f}" cy="{Yf(min(b, ymax)):.1f}" r="{6 if bad else 4}" fill="{GOLD if bad else MAROON}" fill-opacity="{1 if bad else 0.65}"/>')
    q36 = dev["qubits"][odd[0]]
    p.append(f'<line x1="{Xf(q36["t1_us"]) + 4:.1f}" y1="{Yf(q36["t2_us"]) - 4:.1f}" x2="{Xf(25):.1f}" y2="{Yf(250):.1f}" stroke="{INK}" stroke-width="1.2"/>'
             f'<text x="{Xf(27):.1f}" y="{Yf(258):.1f}" font-size="14">qubit {odd[0]}</text>')
    p.append(f'<text x="{(x0 + x1) / 2}" y="{h - 18}" font-size="15" text-anchor="middle" fill="{MUTED}">T1 (microseconds)</text>')
    p.append(f'<text x="16" y="{(yb + yt) / 2}" font-size="15" text-anchor="middle" fill="{MUTED}" transform="rotate(-90 16 {(yb + yt) / 2})">T2 (microseconds)</text>')
    p.append('</svg>')
    return "".join(p)


def tomo_table():
    rows = [("exact", r_true), ("1,000 shots", r_est), ("10,000 shots, 3% readout", r_ro)]
    s = '<table style="font-size:19px;white-space:nowrap"><tr><th></th><th>x</th><th>y</th><th>z</th><th>|r|</th><th>F</th></tr>'
    for name, r in rows:
        s += f"<tr><td>{name}</td>" + "".join(f"<td>{v:+.3f}</td>" for v in r) + f"<td>{np.linalg.norm(r):.3f}</td><td>{fid(r):.3f}</td></tr>"
    return s + "</table>"


m1 = (ROOT / "media/level1/m1-slides.html").read_text().split("\n")
head = "\n".join(m1[:56]).replace("<title>Module 1 Slides: From Bits to Qubits</title>",
                                  "<title>Module 1 Slides: Noise and Open Quantum Systems</title>")
assert "Noise and Open" in head
head = head.replace("  svg text { font-family:",
                    "  .slide code { font-family: Consolas, \"Courier New\", monospace; font-size:0.9em; }\n"
                    "  .slide table td, .slide table th { padding:4px 10px; text-align:right; }\n  svg text { font-family:")
body = (ROOT / "tools/l3_m1_slides_body.html").read_text()
rep = {"/*TITLE_DISC*/": title_disc(), "/*BALL*/": ball(), "/*CURVES*/": curves(), "/*T1FIT*/": t1fit(),
       "/*DEVICE*/": device(), "/*TOMO*/": tomo_table(),
       "/*RP*/": f"{r_p[0]:g}, {r_p[1]:g}, {r_p[2]:g}", "/*RP_LEN*/": f"{np.linalg.norm(r_p):.3f}",
       "/*RP_SQ*/": f"{r_p @ r_p:g}", "/*RP_PUR*/": f"{pur_p:g}",
       "/*Z500*/": f"{1 - 2 * math.exp(-5):.3f}", "/*C500*/": f"{math.exp(-500 / T2):.4f}",
       "/*T1FIT_VAL*/": f"{T1_FIT:.1f}", "/*BACKEND*/": "FakePittsburgh", "/*CAL_DATE*/": "17 April 2026",
       "/*NQ*/": str(dev["num_qubits"]), "/*MED_T1*/": f"{np.median(t1):.0f}", "/*MED_T2*/": f"{np.median(t2):.0f}",
       "/*SX*/": f"{dev['sx_duration_ns']:.0f}", "/*N_SX*/": f"{N_SX:,.0f}", "/*QSIM_V*/": qsim.__version__}
for k_, v in rep.items():
    assert k_ in body, k_
    body = body.replace(k_, v)
n_slides = body.count('<section class="slide')
bar = "\n".join(m1[365:373]).replace("1 / 19", f"1 / {n_slides}")
ctrl = "\n".join(m1[393:])
out = head + "\n" + body + "\n" + bar + "\n<script>\n(function () {\n" + ctrl
assert "`" not in out and "${" not in out and "—" not in out and "/*" not in body
(OUT / "m1-slides.html").write_text(out)
print("wrote media/level3/m1-slides.html", len(out), "bytes,", n_slides, "slides")
print(f"rho_p r = {r_p}, |r| = {np.linalg.norm(r_p):.4f}, purity {pur_p}; T1 fit {T1_FIT:.2f}; median T1 {np.median(t1):.2f}, "
      f"T2 {np.median(t2):.2f}, sx per T1 {N_SX:.0f}; tomography |r| {np.linalg.norm(r_est):.4f} F {fid(r_est):.4f}, "
      f"readout |r| {np.linalg.norm(r_ro):.4f} F {fid(r_ro):.4f}")

src = ROOT / "tools/l3_m1_explore_src.html"
exp = src.read_text()
assert "`" not in exp and "${" not in exp and "—" not in exp
(OUT / "m1-explore.html").write_text(exp)
print("wrote media/level3/m1-explore.html", len(exp))
