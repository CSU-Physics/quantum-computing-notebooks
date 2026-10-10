"""Build the Level 3 Module 5 browser notebook (comparing quantum hardware from published data: an error budget for
a circuit, how many two-qubit gates a device allows, the spread of one device's calibration, coherence against gate
time, and the platforms side by side). The check code is checks/l3_m5_check.py, published next to the notebook by
l3_hidden. Data: content/level3/l3_m5_data.json (tools/make_l3_m5_data.py)."""
import shutil
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level3"
from l3_hidden import hidden_check  # noqa: E402
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}
NAME = "QC-L3-M5-lab-hardware-data.ipynb"
VERSION = "2026-10-10"


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md(f"""# Module 5 lab: comparing quantum hardware from published data

**Quantum Computing Advanced · Module 5 · about 50 minutes** · notebook version {VERSION}

Every company and laboratory publishes numbers for its quantum computer: qubits, gate errors, readout errors, coherence times. The numbers are real, but they are rarely measured the same way. In this lab you work with figures published for eleven devices and experiments, from superconducting chips to trapped ions, neutral atoms, silicon spin qubits and photons, and with the full calibration of two IBM computers:

1. look at the published figures and what kind of number each one is;
2. write `circuit_success()`: the chance that a circuit runs without any error, from its gate counts and the error rates;
3. write `max_two_qubit_gates()`: how many two-qubit gates a device allows before that chance falls below one half;
4. write `summarize()`: the median, best and worst of a list of calibration values, and see how much one chip's qubits differ;
5. compare coherence time with gate time;
6. put the platforms side by side;
7. get your verification value.

The **Module 5 lab check** in Canvas asks for results from this notebook, so keep it open in a second tab. Run each code cell with **Shift + Enter**, in order."""),

md("""## Step 0: start Python

Run the cell below. It loads NumPy, Matplotlib and the data file `l3_m5_data.json`, which sits next to this notebook."""),
code("""import json
import math
import numpy as np
import matplotlib.pyplot as plt

data = json.load(open("l3_m5_data.json"))
platforms = data["platforms"]
devices = data["devices"]
print("Ready. NumPy", np.__version__, "|", len(platforms), "published devices and experiments |",
      "calibration of", ", ".join(devices))"""),

md("""## Step 1: the published figures

Each row gives one device or experiment, with its figures exactly as the source states them. `e1`, `e2` and `e_ro` are the error rates of a single-qubit gate, a two-qubit gate and a readout (an error of 0.001 is a fidelity of 99.9%). `T1` and `T2` are in microseconds, the two-qubit gate time `t2q` too. A dash means the source does not give that figure. Run the cell."""),
code("""def show(x, fmt):
    return "-" if x is None else format(x, fmt)

print(f"{'device':34s}{'qubits':>7s}{'e1':>9s}{'e2':>9s}{'e_ro':>9s}{'T2 (us)':>11s}{'t2q (us)':>10s}")
for p in platforms:
    print(f"{p['name']:34s}{show(p['qubits'], 'd'):>7s}{show(p['e1'], '.2g'):>9s}{show(p['e2'], '.2g'):>9s}"
          f"{show(p['e_ro'], '.2g'):>9s}{show(p['t2_us'], '.4g'):>11s}{show(p['t2q_us'], '.3g'):>10s}")
print()
for p in platforms:
    print(f"{p['name']}: {p['kind']}. Source: {p['source']}.")"""),
md("""**What to notice.** The figures are of different kinds. IBM's are medians over every qubit of one chip on one day. Google's are means with all qubits operating at the same time, which is harder than one pair at a time. Quantinuum's come from randomized benchmarking; IonQ's two-qubit figure is not corrected for preparation and measurement errors; the silicon figures are bounds ("above 99%"); PsiQuantum's two-qubit figure is a fusion, not a gate, and leaves out photon loss. Some experiments report no gate errors at all: the Caltech array shows 6,100 atoms and their coherence, not computation. Comparisons across rows are useful, but only as rough guides."""),

md("""## Step 2: an error budget for a circuit

An illustrative model: every operation fails independently, with the error rate of its kind. A circuit with n1 single-qubit gates, n2 two-qubit gates and nm measurements then runs without any error with probability

P = (1 − e1)^n1 · (1 − e2)^n2 · (1 − e_ro)^nm.

The model ignores idle errors, crosstalk and leakage, and real errors do not always ruin the result. Published gate errors are usually average infidelities, not literal failure probabilities, so P is a toy comparison of how fast errors add up, not a prediction of what a device will do or a ranking of real computers. Write `circuit_success(e1, e2, e_ro, n1, n2, nm)`. The cell then applies it to two circuits on every device that has all three error rates and enough qubits: a 20-qubit GHZ state (1 H gate, 19 CNOTs, 20 measurements) and a 50-qubit circuit of 20 layers (1,000 single-qubit and 500 two-qubit gates, 50 measurements)."""),
code("""def circuit_success(e1, e2, e_ro, n1, n2, nm):
    \"\"\"Probability that n1 single-qubit gates, n2 two-qubit gates and nm measurements all run without error.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete circuit_success() first.")


CIRCUITS = {"GHZ, 20 qubits": (1, 19, 20, 20), "20 layers, 50 qubits": (1000, 500, 50, 50)}
for cname, (n1, n2, nm, nq) in CIRCUITS.items():
    print(cname)
    for p in platforms:
        if None in (p["e1"], p["e2"], p["e_ro"]) or (p["qubits"] or 0) < nq:
            continue
        print(f"   {p['name']:32s} P = {circuit_success(p['e1'], p['e2'], p['e_ro'], n1, n2, nm):.3f}")"""),
md("""**What to notice.** In this model, for the GHZ state every listed device keeps most of its runs error free, from 0.659 (ibm_miami) to 0.976 (Helios). For the 20-layer circuit the differences grow: 0.641 for Helios, 0.355 for ibm_boston, below 0.1 for Willow and ibm_miami. Small differences in the error rate become large differences in deep circuits, because the error rate is raised to the number of gates. Two-qubit gates dominate: 500 of them at 0.3% error leave only (0.997)^500 ≈ 0.22."""),

md("""## Step 3: how many two-qubit gates?

A handy single number for a device: the largest number of two-qubit gates n with (1 − e2)^n still at least one half. Taking logarithms, n ≤ ln(0.5)/ln(1 − e2), so n is that value **rounded down**. For small e2 it is about 0.69/e2: 690 gates at 0.1% error.

Write `max_two_qubit_gates(e2, target=0.5)` and return a whole number. Use the exact logarithm `math.log(1 - e2)`."""),
code("""def max_two_qubit_gates(e2, target=0.5):
    \"\"\"The largest whole number n with (1 - e2) ** n >= target.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete max_two_qubit_gates() first.")


for p in platforms:
    if p["e2"] is not None:
        print(f"{p['name']:34s} e2 = {p['e2']:<9.4g} at most {max_two_qubit_gates(p['e2']):4d} two-qubit gates")"""),
md("""**What to notice.** From 68 gates (silicon) to 877 (Helios). The silicon row uses e2 = 0.01, a conservative assumed value: the paper reports fidelities above 99%, so the real error is below 1% and the model's count for these devices would be higher, not lower. None of these devices can run the millions of gates that useful algorithms such as Shor's need: that requires error correction (Module 2), which turns many physical qubits into fewer, better logical qubits. These numbers say how deep a circuit can be today **without** it, and why mitigation (Module 3) matters now."""),

md("""## Step 4: one chip, many qubits

A single median hides a lot. IBM publishes the calibration of every qubit and every pair of its devices. Here you look at two: ibm_boston (Heron r3, 156 qubits) and ibm_miami (Nighthawk r1, 120 qubits), both calibrated on 17 April 2026.

Write `summarize(values, higher_is_better)`. It returns a dict with the `"median"`, the `"best"` and the `"worst"` value. For T1 and T2 a higher value is better; for error rates a lower one is. The cell leaves out error rates of 1, which mark a qubit or pair that IBM reports as unusable that day."""),
code("""def summarize(values, higher_is_better):
    \"\"\"{'median': ..., 'best': ..., 'worst': ...} of a list of numbers.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete summarize() first.")


def usable(values):
    return [x for x in values if x < 1]


for name, d in devices.items():
    q = d["qubit_data"]
    print(f"{name}: {d['qubits']} qubits, {len(d['pairs'])} connected pairs, calibrated {d['calibrated']}")
    print(f"   {'':12s}{'median':>11s}{'best':>11s}{'worst':>11s}")
    for label, vals, hib in (("T1 (us)", q["t1_us"], True), ("T2 (us)", q["t2_us"], True),
                             ("e1", usable(q["e1"]), False), ("e2 (CZ)", usable(d["e2"]), False),
                             ("e_ro", usable(q["e_ro"]), False)):
        s = summarize(vals, hib)
        print(f"   {label:12s}{s['median']:11.4g}{s['best']:11.4g}{s['worst']:11.4g}")
    print(f"   left out as unusable: {sum(x >= 1 for x in d['e2'])} of {len(d['e2'])} pairs, "
          f"{sum(x >= 1 for x in q['e1'])} of {d['qubits']} qubits")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, (name, d) in zip(axes, devices.items()):
    e2 = np.array(usable(d["e2"]))
    ax.hist(np.log10(e2), bins=30, color="#99004C")
    ax.axvline(np.log10(np.median(e2)), color="#24313D", ls="--", label=f"median {np.median(e2):.2g}")
    ax.set_xlabel("log10 of the CZ error of each pair")
    ax.set_ylabel("pairs")
    ax.set_title(name)
    ax.legend()
plt.tight_layout()
plt.show()"""),
md("""**What to notice.** On ibm_boston the median CZ error is 0.0013, but the best pair reaches 0.00073 and the worst 0.16, more than a hundred times the median. T1 runs from 6.8 µs to 404 µs around a median of 292 µs. A headline "best" number describes one pair, not the chip; a median describes a typical pair; and a circuit placed on the worst qubits would do far worse than either. That is why the transpiler (Level 2) chooses qubits using the calibration, and why the same device can do better or worse from one day to the next."""),

md("""## Step 5: coherence against gate time

Long coherence alone does not make a good qubit; what counts is how many gates fit into it. Run the cell: it divides T2 by the two-qubit gate time where a source gives both, and also shows how long 1,000 layers of two-qubit gates take."""),
code("""print(f"{'device':34s}{'T2 / t2q':>10s}{'1,000 layers take':>20s}")
for p in platforms:
    if p["t2_us"] and p["t2q_us"]:
        t = 1000 * p["t2q_us"] * 1e-6
        print(f"{p['name']:34s}{p['t2_us'] / p['t2q_us']:10.0f}{t:17.3g} s")"""),
md("""**What to notice.** IonQ's ions keep their coherence for about 1 s, thousands of times longer than IBM's qubits (353 µs on ibm_boston), but their gates are also about 9,000 times slower (600 µs against 68 ns). The ratios end up similar: about 1,700 two-qubit gate times within T2 for IonQ Aria and ibm_miami, about 5,200 for ibm_boston. Speed matters on its own, too: 1,000 layers take 68 µs on ibm_boston and 0.6 s on Aria, and error correction needs many rounds of measurement. Superconducting qubits are fast, ions and atoms are slow but very uniform and can connect any pair."""),

md("""## Step 6: the platforms side by side

The last cell plots every device with a qubit count and a two-qubit error. Remember Step 1: the points are not all the same kind of number."""),
code("""colors = {"superconducting": "#99004C", "trapped": "#24313D", "neutral": "#C08A00", "silicon": "#2E7D32", "photons": "#6A5ACD"}
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for p in platforms:
    if p["qubits"] and p["e2"]:
        c = next(v for k, v in colors.items() if p["platform"].startswith(k))
        ax.scatter(p["qubits"], p["e2"], s=60, color=c)
        ax.annotate(p["name"], (p["qubits"], p["e2"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("qubits")
ax.set_ylabel("two-qubit error")
ax.set_title("Published two-qubit errors (lower is better)")
plt.tight_layout()
plt.show()"""),
md("""**What to notice.** No platform wins on every axis. Trapped ions have the lowest errors and connect every pair, but run slowly and have about 100 qubits per trap so far. Superconducting chips are fast and made with chip-fabrication tools, with errors a few times higher. Neutral atoms reach thousands of trapped atoms, but moving and reading out atoms is slow. Silicon spin qubits are tiny and made in the same 300 mm fabs as transistors, but the largest arrays have only about a dozen qubits. Photons work at room temperature apart from the detectors, but lose photons. Each company is betting on a different route to error-corrected machines."""),

md("""## Step 7: your personal check

Open the **Module 5 lab check** in Canvas. Question 1 shows your own N1 (single-qubit gates, 100 to 1,000), N2 (two-qubit gates, 20 to 300) and NM (measurements, 5 to 50). Type them below and run the next two cells. The check cell tests `circuit_success()`, `max_two_qubit_gates()` and `summarize()`. Only if every test passes does it print your **verification value**: 1,000 times the success probability of your circuit with ibm_boston's median error rates (usable qubits and pairs only), rounded to a whole number. For example, N1, N2, NM = 500, 100, 20 gives 739."""),
code("""N1 = 0    # your number from Canvas, for example 500
N2 = 0    # for example 100
NM = 0    # for example 20"""),
code(hidden_check(['l3_m5_check'], 'check_l3_module5', '''

passed, messages, value = check_l3_module5(circuit_success, max_two_qubit_gates, summarize, N1, N2, NM)
for m in messages:
    print(m)
if passed:
    print("\\nAll tests passed. Your verification value is", value)
else:
    print("\\nNot yet: fix the item above and run this cell again.")''')),

md("""## What you should notice

- Published hardware figures are real but of different kinds: medians, means, best values, bounds, measured one pair at a time or all at once. Read what each number is before comparing.
- An error budget P = (1 − e1)^n1 (1 − e2)^n2 (1 − e_ro)^nm shows why two-qubit errors decide how deep a circuit can be: about 0.69/e2 gates halve the chance of an error-free run.
- One chip's qubits differ by large factors; medians describe a device better than its best pair.
- Coherence counts only relative to gate speed, and speed matters on its own.
- Every platform trades something: speed, errors, connectivity, qubit count, temperature, manufacturability.

**Next in Canvas:** the Module 5 lab check, the quiz and the time log."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl3m5c{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / NAME)
print("wrote", OUT / NAME)
