"""Build the optional Colab notebook of Level 3 Module 6: one capstone's reference circuits on a real IBM computer
(or, without an account, on FakeManilaV2), compared with the ideal values and the course noise model.
Writes colab/level3/QC-L3-M6-hardware-colab.ipynb. Not graded. The course-model numbers in the notebook are computed
here from the capstone check modules (exact density matrices)."""
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "level3"), str(ROOT / "checks")]
import l3_m6_kernel_check as KC  # noqa: E402
import l3_m6_mit_check as MC  # noqa: E402
import l3_m6_rep_check as RC  # noqa: E402
import l3_m6_spin_check as SC  # noqa: E402

OUT = ROOT / "colab" / "level3"
NAME = "QC-L3-M6-hardware-colab.ipynb"
VERSION = "2026-10-09"
META = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"}, "colab": {"provenance": []}}
P2, RO = 0.01, 0.02


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


# ---------------------------------------------------------------- course-model predictions (exact)
d3 = RC.record_distribution(RC.reference_memory_circuit(3), P2, RO, P2)
rates = [0.0] * 4
for k, p in d3.items():
    for i, e in enumerate(RC.reference_detection_events(k, 3)):
        if e != (0, 0):
            rates[i] += p
rep_course = {"logical": RC.logical_error(d3, 3), "bare": RC.record_distribution(RC.bare_circuit(3), P2, RO, P2).get("1", 0.0),
              "rates": [round(r, 4) for r in rates]}
ms = MC.study(3, P2, RO)
mit_course = {"raw": round(ms["raw"][1], 4), "readout": round(ms["ro"][1], 4), "zne_lin": round(ms["zne_lin"], 4)}
ss = SC.study(8, P2, RO)
spin_course = {"raw": round(ss["raw"][1], 4), "corrected": round(ss["corrected"][1], 4), "mitigated": round(ss["mitigated"], 4),
               "trotter": round(ss["ideal"], 4), "exact": round(SC.exact_magnetization(3, 0.7, 2.0), 4)}
Xtr, ytr, Xte, yte = KC.capstone_data(str(ROOT / "content" / "level3" / "l3_m4a_data.json"))
COURSE = {"repetition": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rep_course.items()},
          "mitigation": mit_course, "spin": spin_course}
DATA = {"X_train": Xtr.tolist(), "y_train": ytr.tolist(), "X_test": Xte.tolist(), "y_test": yte.tolist()}
print(COURSE)

cells = [
md(f"""# Module 6 (optional, Colab): your capstone on a real IBM quantum computer

**Quantum Computing Advanced · Module 6 · optional, not graded · about 30 minutes plus the queue** · notebook version {VERSION}

This notebook runs the reference circuits of **one capstone** on a real IBM quantum computer, applies the same analysis as your capstone notebook (written out here with Qiskit), and compares the result with the ideal value and with the **course noise model** (P2 = 0.01, READOUT = 0.02). Nothing in it is graded, and the badge does not depend on it.

It contains working versions of the capstone circuits, so **finish your own capstone notebook first**.

- **With an IBM Quantum account** (the free Open Plan is enough) the circuits run on a real computer. Each experiment is one job of a few thousand shots per circuit; that is usually well under a minute of the Open Plan's 10 minutes every 28 days, but check your remaining time first. The queue can take from seconds to hours.
- **Without an account** they run on `FakeManilaV2`, a 5-qubit model of an older IBM device, simulated in Colab, through the same code.

Run each cell with **Shift + Enter**, in order."""),

md("## Step 0: install Qiskit\n\nAbout a minute. The versions are fixed so that the code below works as written (the same as in Module 3)."),
code("%pip install -q qiskit==2.5.2 qiskit-aer==0.17.2 qiskit-ibm-runtime==0.50.0 scikit-learn"),

md("""## Step 1: choose the computer

To use a real IBM computer, store your API key in Colab's **Secrets** (the key icon on the left) under the name `IBM_QUANTUM_TOKEN`, and, if your account has one, the instance CRN under `IBM_QUANTUM_INSTANCE`. Allow this notebook to read them. **Never paste the key into a cell**: a notebook can be shared, and its cells with it. Without these secrets the cell picks `FakeManilaV2`."""),
code("""import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
import qiskit, qiskit_ibm_runtime
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.executor_sampler import Sampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
warnings.filterwarnings("ignore", category=DeprecationWarning)

token = instance = None
try:
    from google.colab import userdata
    try:
        token = userdata.get("IBM_QUANTUM_TOKEN")
        instance = userdata.get("IBM_QUANTUM_INSTANCE")
    except Exception:
        pass
except ImportError:
    pass

if token:
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance or None)
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=5)
    REAL = True
else:
    backend = FakeManilaV2()
    REAL = False
print("qiskit", qiskit.__version__, "| qiskit-ibm-runtime", qiskit_ibm_runtime.__version__)
print("computer:", backend.name, "(real hardware)" if REAL else "(simulated fake device)")


def run(circuits, shots, optimization_level=1):
    \"\"\"Transpile the circuits for the computer, run them in one job, return a list of counts.\"\"\"
    pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level, seed_transpiler=7)
    isa = pm.run(circuits)
    job = Sampler(mode=backend).run(isa, shots=shots)
    if REAL:
        print("job ID:", job.job_id(), "- waiting for the queue (the result also appears under Workloads on the platform)...")
    result = job.result()
    two_q = [sum(1 for i in c.data if i.operation.num_qubits == 2 and i.operation.name != "barrier") for c in isa]
    print(f"{len(isa)} circuits, {shots} shots each; two-qubit gates after transpiling: {two_q}")
    return [r.data.c.get_counts() for r in result]


def readout_calibration(n, shots=4000):
    \"\"\"Two calibration circuits (all 0, all 1): each qubit's chance e0 of reading 1 for a 0 and e1 of reading 0 for a 1.
    Returns the circuits; readout_errors() turns their counts into the lists e0, e1.\"\"\"
    c0 = QuantumCircuit(n, n)
    c0.measure(range(n), range(n))
    c1 = QuantumCircuit(n, n)
    c1.x(range(n))
    c1.measure(range(n), range(n))
    return [c0, c1]


def readout_errors(counts0, counts1):
    n = len(next(iter(counts0)))
    s0, s1 = sum(counts0.values()), sum(counts1.values())
    e0 = [sum(v for k, v in counts0.items() if k[n - 1 - j] == "1") / s0 for j in range(n)]
    e1 = [sum(v for k, v in counts1.items() if k[n - 1 - j] == "0") / s1 for j in range(n)]
    return e0, e1


def bootstrap_sigma(counts, estimator, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    keys = list(counts)
    k = np.array([counts[x] for x in keys], dtype=float)
    shots = int(k.sum())
    vals = [estimator({x: int(c) for x, c in zip(keys, rng.multinomial(shots, k / shots)) if c > 0}) for _ in range(n_boot)]
    return float(np.std(vals, ddof=1))"""),

md("""## Step 2: choose your capstone

Set `EXPERIMENT` to your capstone: `"repetition"`, `"mitigation"`, `"spin"` or `"kernel"`. Then run the cell of Step 3 for that experiment only (the other cells check `EXPERIMENT` and do nothing).

| Experiment | Circuits | Shots |
|---|---|---|
| `repetition` | memory experiment, 3 rounds, logical 0 and 1; a bare qubit, 0 and 1 | 4 × 4,000 |
| `mitigation` | ⟨X X X⟩ of the GHZ state at folding scales 1, 3, 5; 2 readout calibrations | 5 × 4,000 |
| `spin` | 3 spins, T = 2, 8 Trotter steps, RZZ folded at scales 1, 3, 5; 2 readout calibrations | 5 × 4,000 |
| `kernel` | the 28 training and 32 test overlap circuits | 60 × 1,000 |"""),
code(f"""EXPERIMENT = "repetition"     # "repetition", "mitigation", "spin" or "kernel"

COURSE = {COURSE!r}     # the course noise model's exact predictions (from the capstone notebooks)"""),

md("""## Step 3a: repetition code

The memory experiment of your capstone, with 3 rounds of syndrome measurement. Mid-circuit measurements and resets are supported on IBM's current computers. The transpiler removes the `id` gates (they do nothing on hardware), but the data qubits still wait while the ancillas are measured, which is the idle error the course model put on them."""),
code("""def memory_circuit(rounds, bit=0):
    qc = QuantumCircuit(5, 2 * rounds + 3)
    if bit:
        qc.x(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    for r in range(rounds):
        qc.barrier()
        qc.cx(0, 3); qc.cx(1, 3); qc.cx(1, 4); qc.cx(2, 4)
        qc.measure(3, 2 * r); qc.measure(4, 2 * r + 1)
        qc.reset(3); qc.reset(4)
    for d in range(3):
        qc.measure(d, 2 * rounds + d)
    return qc


def bare_circuit(rounds, bit=0):
    qc = QuantumCircuit(1, 1)
    if bit:
        qc.x(0)
    for _ in range(rounds):
        qc.id(0)
    qc.measure(0, 0)
    return qc


def bits(key, rounds):
    b = [int(c) for c in reversed(key)]
    syn = [(b[2 * r], b[2 * r + 1]) for r in range(rounds)]
    d = b[2 * rounds: 2 * rounds + 3]
    return syn, d


def detection_events(key, rounds):
    syn, d = bits(key, rounds)
    syn = syn + [(d[0] ^ d[1], d[1] ^ d[2])]
    prev, ev = (0, 0), []
    for s in syn:
        ev.append((s[0] ^ prev[0], s[1] ^ prev[1]))
        prev = s
    return ev


if EXPERIMENT == "repetition":
    R, S = 3, 4000
    counts = run([memory_circuit(R, 0), memory_circuit(R, 1), bare_circuit(R, 0), bare_circuit(R, 1)], S)
    for bit, c in ((0, counts[0]), (1, counts[1])):
        wrong = sum(v for k, v in c.items() if (1 if sum(bits(k, R)[1]) >= 2 else 0) != bit)
        f = wrong / S
        print(f"stored {bit}: logical error {f:.4f} +- {math.sqrt(f * (1 - f) / S):.4f}")
    for bit, c in ((0, counts[2]), (1, counts[3])):
        f = c.get(str(1 - bit), 0) / S
        print(f"bare qubit, stored {bit}: error {f:.4f} +- {math.sqrt(f * (1 - f) / S):.4f}")
    rate = [sum(v for k, v in counts[0].items() if detection_events(k, R)[i] != (0, 0)) / S for i in range(R + 1)]
    print("detection-event rates (stored 0), rounds 1 to 3 and final data:", [round(r, 4) for r in rate])
    kept = {k: v for k, v in counts[0].items() if all(e == (0, 0) for e in detection_events(k, R))}
    nk = sum(kept.values())
    nw = sum(v for k, v in kept.items() if sum(bits(k, R)[1]) >= 2)
    print(f"post-selection: kept {nk} of {S} shots, {nw} of them wrong")
    print("course noise model (stored 0):", COURSE["repetition"])"""),

md("""## Step 3b: mitigation study

⟨X X X⟩ of the three-qubit GHZ state at folding scales 1, 3 and 5. A **barrier** separates U, U⁻¹ and U: without it the transpiler would notice that U⁻¹ U does nothing and remove it, and there would be no extra noise to extrapolate. The readout is mitigated with the computer's own error rates, measured by two calibration circuits (all qubits 0, all qubits 1) on the same qubits, because the real readout errors are not the course's."""),
code("""def ghz_circuit(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    for i in range(n):
        qc.h(i)
    return qc


def fold_global(qc, scale):
    out = qc.copy()
    for _ in range((scale - 1) // 2):
        out.barrier()
        out.compose(qc.inverse(), inplace=True)
        out.barrier()
        out.compose(qc, inplace=True)
    return out


def measured(qc):
    out = QuantumCircuit(qc.num_qubits, qc.num_qubits)
    out.compose(qc, inplace=True)
    out.measure(range(qc.num_qubits), range(qc.num_qubits))
    return out


def parity(probs):
    return sum((-1) ** k.count("1") * v for k, v in probs.items()) / sum(probs.values())


def mitigate(probs, e0, e1):
    \"\"\"Each qubit's readout inverted with its own measured error rates.\"\"\"
    n = len(next(iter(probs)))
    v = np.zeros(2 ** n)
    for k, p in probs.items():
        v[int(k, 2)] = p
    v /= v.sum()
    for j in range(n):
        inv = np.linalg.inv(np.array([[1 - e0[j], e0[j]], [e1[j], 1 - e1[j]]]).T)
        v = np.einsum("ij,ajb->aib", inv, v.reshape(2 ** (n - 1 - j), 2, 2 ** j)).reshape(-1)
    return {format(i, f"0{n}b"): float(x) for i, x in enumerate(v)}


if EXPERIMENT == "mitigation":
    n, S, SCALES = 3, 4000, [1, 3, 5]
    counts = run([measured(fold_global(ghz_circuit(n), s)) for s in SCALES] + readout_calibration(n), S)
    e0, e1 = readout_errors(counts[3], counts[4])
    print("measured readout errors, 0 read as 1:", np.round(e0, 4), " 1 read as 0:", np.round(e1, 4))
    est = lambda c: parity(mitigate(c, e0, e1))
    raw = [parity(counts[i]) for i in range(3)]
    ro = [est(counts[i]) for i in range(3)]
    sig = [bootstrap_sigma(counts[i], est) for i in range(3)]
    lin = list(np.linalg.pinv(np.vander(np.array(SCALES, float), 2, increasing=True))[0])
    zne = sum(c * v for c, v in zip(lin, ro))
    zsig = math.sqrt(sum((c * s) ** 2 for c, s in zip(lin, sig)))
    for s, a, b, sg in zip(SCALES, raw, ro, sig):
        print(f"scale {s}: raw {a:.4f}   readout-mitigated {b:.4f} +- {sg:.4f}")
    print(f"readout + linear ZNE: {zne:.4f} +- {zsig:.4f}   (ideal 1)")
    print("course noise model:", COURSE["mitigation"])"""),

md("""## Step 3c: spin chain

M(T = 2) of the three-spin chain with 8 Trotter steps. Each RZZ is folded locally, with **barriers** between the folded gates so that the transpiler cannot cancel RZZ(−θ) RZZ(θ). The readout correction uses the computer's own error rates from two calibration circuits, applied to each qubit's ⟨Z⟩, and the bootstrap gives each σ."""),
code("""def trotter_circuit(N, h, T, steps):
    qc = QuantumCircuit(N)
    dt = T / steps
    for _ in range(steps):
        for i in range(N - 1):
            qc.rzz(-2 * dt, i, i + 1)
        for i in range(N):
            qc.rx(-2 * h * dt, i)
    return qc


def fold_rzz(qc, scale):
    out = qc.copy_empty_like()
    for inst in qc.data:
        if inst.operation.name == "rzz":
            th = float(inst.operation.params[0])
            a, b = inst.qubits
            out.rzz(th, a, b)
            for _ in range((scale - 1) // 2):
                out.barrier(a, b); out.rzz(-th, a, b); out.barrier(a, b); out.rzz(th, a, b)
        else:
            out.append(inst)
    return out


def z_values(counts):
    n = len(next(iter(counts)))
    s = sum(counts.values())
    return [sum(v * (1 if k[n - 1 - j] == "0" else -1) for k, v in counts.items()) / s for j in range(n)]


if EXPERIMENT == "spin":
    N, H, T, STEPS, S, SCALES = 3, 0.7, 2.0, 8, 4000, [1, 3, 5]
    base = trotter_circuit(N, H, T, STEPS)
    circs = []
    for s in SCALES:
        c = QuantumCircuit(N, N)
        c.compose(fold_rzz(base, s), inplace=True)
        c.measure(range(N), range(N))
        circs.append(c)
    counts = run(circs + readout_calibration(N), S)
    e0, e1 = readout_errors(counts[3], counts[4])
    print("measured readout errors, 0 read as 1:", np.round(e0, 4), " 1 read as 0:", np.round(e1, 4))
    corrected = lambda c: float(np.mean([(zj - (e1[j] - e0[j])) / (1 - e0[j] - e1[j]) for j, zj in enumerate(z_values(c))]))
    raw = [float(np.mean(z_values(c))) for c in counts[:3]]
    corr = [corrected(c) for c in counts[:3]]
    sig = [bootstrap_sigma(c, corrected) for c in counts[:3]]
    lin = [13 / 12, 1 / 3, -5 / 12]
    mit = sum(c * v for c, v in zip(lin, corr))
    msig = math.sqrt(sum((c * sg) ** 2 for c, sg in zip(lin, sig)))
    for s, a, b, sg in zip(SCALES, raw, corr, sig):
        print(f"scale {s}: raw M {a:.4f}   readout-corrected {b:.4f} +- {sg:.4f}")
    print(f"mitigated M(2.0) = {mit:.4f} +- {msig:.4f};  exact {COURSE['spin']['exact']}")
    print("course noise model:", COURSE["spin"])"""),

md("""## Step 3d: kernel classifier

The 8 training and 4 test points of your capstone (written out below), 28 + 32 overlap circuits of 1,000 shots in one job, and the support vector classifier on the measured kernel."""),
code(f"""from sklearn.svm import SVC

DATA = {DATA!r}


def feature_map(x, reps=2):
    qc = QuantumCircuit(2)
    for _ in range(reps):
        qc.h([0, 1])
        qc.p(2 * x[0], 0); qc.p(2 * x[1], 1)
        qc.cx(0, 1); qc.p(2 * (np.pi - x[0]) * (np.pi - x[1]), 1); qc.cx(0, 1)
    return qc


def overlap_circuit(x1, x2):
    qc = QuantumCircuit(2, 2)
    qc.compose(feature_map(x1), inplace=True)
    qc.compose(feature_map(x2).inverse(), inplace=True)
    qc.measure([0, 1], [0, 1])
    return qc


if EXPERIMENT == "kernel":
    Xtr, ytr = np.array(DATA["X_train"]), np.array(DATA["y_train"])
    Xte, yte = np.array(DATA["X_test"]), np.array(DATA["y_test"])
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    tests = [(a, b) for a in range(4) for b in range(8)]
    circs = [overlap_circuit(Xtr[i], Xtr[j]) for i, j in pairs] + [overlap_circuit(Xte[a], Xtr[b]) for a, b in tests]
    counts = run(circs, 1000, optimization_level=3)
    k = [c.get("00", 0) / 1000 for c in counts]
    Ktr = np.eye(8)
    for (i, j), v in zip(pairs, k[:28]):
        Ktr[i, j] = Ktr[j, i] = v
    Kte = np.array(k[28:]).reshape(4, 8)
    model = SVC(kernel="precomputed", C=1).fit(Ktr, ytr)
    print("training kernel (measured):\\n", np.round(Ktr, 3))
    print("test accuracy (4 points):", model.score(Kte, yte), "  decision values:", np.round(model.decision_function(Kte), 3))
    print(f"K(x0, x1) measured {{Ktr[0, 1]:.4f}} +- {{math.sqrt(Ktr[0, 1] * (1 - Ktr[0, 1]) / 1000):.4f}};  "
          f"ideal {KC.exact_entry(Xtr[0], Xtr[1], 0, 0):.4f}, course noise model {KC.exact_entry(Xtr[0], Xtr[1], P2, RO):.4f}")"""),

md("""## What to look for

- **σ comes from your shots.** Compare each measured value with the course model's prediction: a difference of more than 3σ means the course model does not describe this computer. IBM's current computers are usually **less** noisy than the course model (their two-qubit errors are a few times 0.001), so expect results closer to the ideal than the course predicts, and sometimes worse ones on a busy or poorly calibrated device.
- **Readout** errors on real devices differ from qubit to qubit and are not the course's 0.02 and 0.04; that is why the mitigation and spin cells measure them.
- **Folding on hardware** only amplifies the noise if the transpiler keeps the extra gates: the barriers do that. Real noise is not exactly depolarizing, so the extrapolated value can miss the ideal by more than its σ.
- A second run, an hour or a day later, can differ from the first by more than its σ: the computer itself changes, and that uncertainty is not in σ.

If you ran it, mention the result under **Limits** in your capstone summary: it shows how far the course noise model is from a real computer."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl3m6h{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / NAME)
print("wrote", OUT / NAME)
