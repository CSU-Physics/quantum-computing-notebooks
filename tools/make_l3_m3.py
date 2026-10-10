"""Build the Level 3 Module 3 browser notebook (error suppression and mitigation: one expectation value, <XX> of a
Bell state after an idle period, raw, with dynamical decoupling, with readout mitigation and with zero-noise
extrapolation; the cost in shots; saved device data).
The check code is checks/l3_m3_check.py, published next to the notebook by l3_hidden.
Device data: content/level3/l3_m3_device_run.json (tools/make_l3_m3_data.py)."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level3"
from l3_hidden import hidden_check  # noqa: E402
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}
NAME = "QC-L3-M3-lab-error-mitigation.ipynb"
VERSION = "2026-10-10"


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md(f"""# Module 3 lab: error suppression and mitigation

**Quantum Computing Advanced · Module 3 · about 95 minutes** · notebook version {VERSION}

Error correction (Module 2) needs many qubits per logical qubit. Today's computers run circuits without it, and two cheaper tools make their results better. **Suppression** changes the circuit so that less noise happens. **Mitigation** runs extra circuits and corrects the *average* result afterwards. In this lab you take one expectation value, ⟨XX⟩ of a Bell state that waits for 10 µs, and improve it step by step:

1. build the experiment and see the ideal value, 1;
2. add four kinds of noise and see what each one does;
3. write `idle_block(qc, n, dd)`: **dynamical decoupling**, an X echo that cancels a coherent error while the qubits wait;
4. write `mitigate_readout(probs, A)`: undo readout errors with a calibration matrix;
5. write `fold_cx(qc, scale)` and `extrapolate_zero(scales, values)`: **zero-noise extrapolation**;
6. measure the price of mitigation in shots;
7. apply your functions to a saved run with a real device's noise;
8. compare with what IBM's Estimator does for you;
9. get your verification value.

The **Module 3 lab check** in Canvas asks for results from this notebook, so keep it open in a second tab. Run each code cell with **Shift + Enter**, in order."""),

md("""## Step 0: start Python

Run the cell below. The first time, Python can take up to a minute to start in your browser. It also loads the saved device run used in Step 7."""),
code("""import json
import numpy as np
import matplotlib.pyplot as plt
import qsim
from qsim import (QuantumCircuit, NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error,
                  coherent_unitary_error, simulate_density_matrix)

run = json.load(open("l3_m3_device_run.json"))
KEYS = ("00", "01", "10", "11")          # results as Qiskit prints them: qubit 1 on the left, qubit 0 on the right
N_IDLE = 10                              # idle steps of 1 us each
print("Ready. qsim", qsim.__version__, "and NumPy", np.__version__)
print("Device run:", run["backend"], "-", run["kind"])"""),

md("""## Step 1: the experiment

One H gate and one CNOT make the Bell state (|00⟩ + |11⟩)/√2. The two qubits then **wait** for 10 steps of 1 µs, as qubits do in a real circuit while other qubits are busy. Finally an H gate on each qubit and a measurement give **⟨XX⟩**: the H gates turn an X measurement into an ordinary Z measurement, so ⟨XX⟩ is the average of (−1)^(number of 1s in the result). For the Bell state ⟨XX⟩ = 1.

The cell defines four helpers that every later step uses:

- `idle_block(qc, n, dd=False)`: for now, n waiting steps (an `id` gate on each qubit per step). You rewrite it in Step 3.
- `experiment(dd=False)`: the whole circuit.
- `probabilities(qc, nm=None, p_ro=0.0)`: the **exact** probabilities of the four results, from the density matrix, including a readout error p_ro. Exact numbers have no shot noise, so every number you get matches the lab check.
- `parity_expectation(probs)`: ⟨XX⟩ from the probabilities."""),
code("""def idle_block(qc, n, dd=False):
    \"\"\"n idle steps on qubits 0 and 1 (dynamical decoupling comes in Step 3).\"\"\"
    if dd:
        raise NotImplementedError("You write the echo (dd=True) in Step 3.")
    for _ in range(n):
        qc.id(0)
        qc.id(1)


def experiment(dd=False, n_idle=N_IDLE):
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)                          # Bell state (|00> + |11>)/sqrt(2)
    idle_block(qc, n_idle, dd)           # the qubits wait
    qc.h(0)
    qc.h(1)                              # measure X on both qubits
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc


def readout_matrix(p_ro):
    \"\"\"Each qubit: 0 is misread as 1 with probability p_ro, 1 as 0 with probability 2 p_ro.\"\"\"
    return np.array([[1 - p_ro, p_ro], [2 * p_ro, 1 - 2 * p_ro]])


def probabilities(qc, nm=None, p_ro=0.0):
    \"\"\"Exact probabilities of '00', '01', '10', '11' for a circuit that measures both qubits at the end.\"\"\"
    p = np.real(np.diag(simulate_density_matrix(qc, nm).data)).clip(min=0)
    p = p / p.sum()
    if p_ro:
        R = readout_matrix(p_ro)
        p = np.kron(R, R).T @ p          # the readout error of each qubit
    return {k: float(p[i]) for i, k in enumerate(KEYS)}


def parity_expectation(probs):
    return float(sum(v * (-1) ** k.count("1") for k, v in probs.items()))


qc = experiment()
print(qc.draw())
p = probabilities(qc)
print("ideal probabilities:", {k: round(v, 4) for k, v in p.items()}, "  <XX> =", parity_expectation(p))"""),
md("""**What to notice.** Without noise only '00' and '11' appear, each half the time, so ⟨XX⟩ = 1. Every later step starts from this circuit, so any value below 1 is caused by noise."""),

md("""## Step 2: four kinds of noise

The noise model below has four parts, chosen to look like a real device:

| noise | where | fixed by |
|---|---|---|
| **relaxation and dephasing** (T1 = 200 µs, T2 = 150 µs) | every idle step, 1 µs | nothing in this lab |
| **detuning**: an unwanted Z rotation by `delta` per step, a qubit frequency slightly off | every idle step | suppression (Step 3) |
| **readout error**: 0 read as 1 with probability `p_ro`, 1 read as 0 with 2 `p_ro` | measurement | mitigation (Step 4) |
| **CNOT error**: two-qubit depolarizing error `p_cx` (and 0.1% on H and X gates) | gates | mitigation (Step 5) |

The lab uses DELTA = 0.05, P_RO = 0.02 and P_CX = 0.02. The cell switches the parts on one at a time, then all together. *Predict first:* which part costs the most?"""),
code("""T1, T2, T_STEP = 200.0, 150.0, 1.0      # microseconds


def rz_matrix(delta):
    return np.diag([np.exp(-1j * delta / 2), np.exp(1j * delta / 2)])


def noise_model(delta, p_ro, p_cx, relax=True):
    nm = NoiseModel()
    idle = thermal_relaxation_error(T1, T2, T_STEP) if relax else None
    if delta:
        coh = coherent_unitary_error(rz_matrix(delta))
        idle = coh if idle is None else idle.compose(coh)
    if idle is not None:
        nm.add_all_qubit_quantum_error(idle, ["id"])
    nm.add_all_qubit_quantum_error(depolarizing_error(0.001, 1), ["h", "x"])
    if p_cx:
        nm.add_all_qubit_quantum_error(depolarizing_error(p_cx, 2), ["cx"])
    if p_ro:
        nm.add_all_qubit_readout_error(ReadoutError(readout_matrix(p_ro)))
    return nm


DELTA, P_RO, P_CX = 0.05, 0.02, 0.02
nm = noise_model(DELTA, P_RO, P_CX)
qc = experiment()
cases = [("relaxation and dephasing only", noise_model(0, 0, 0), 0.0),
         ("detuning only", noise_model(DELTA, 0, 0, relax=False), 0.0),
         ("readout error only", noise_model(0, P_RO, 0, relax=False), P_RO),
         ("CNOT error only", noise_model(0, 0, P_CX, relax=False), 0.0),
         ("all four together", nm, P_RO)]
for name, model, pro in cases:
    print(f"{name:32s} <XX> = {parity_expectation(probabilities(qc, model, pro)):.4f}")
print(f"\\ncos(2 x {N_IDLE} x DELTA) = {np.cos(2 * N_IDLE * DELTA):.4f};  exp(-2 x {N_IDLE} us / T2) = {np.exp(-2 * N_IDLE / T2):.4f}")
RAW = parity_expectation(probabilities(qc, nm, P_RO))
print(f"raw <XX> with all the noise: {RAW:.3f}")"""),
md("""**What to notice.** The detuning costs the most: each qubit turns by 10 × 0.05 = 0.5 rad around Z, the Bell state's two terms get a relative phase of 1 rad, and ⟨XX⟩ falls to about cos(1) ≈ 0.54. Dephasing alone gives about 0.87 (close to e^(−20/150)), the readout errors about 0.88, the CNOT error about 0.98. Together they bring ⟨XX⟩ from 1 down to **0.409**. The lab check asks for this raw value.

Only one of these errors is **coherent** (the same rotation every time): the detuning. The other three, relaxation and dephasing, the readout errors and the CNOT error, are **incoherent** (random). That difference decides which tool can fix them."""),

md("""## Step 3: suppression with dynamical decoupling

A coherent Z rotation can be undone by a **spin echo**. Wait half the time, so each qubit turns by φ/2; apply X to both qubits; wait the other half. After an X, a Z rotation turns the state the *other* way, so the second half cancels the first. A final X on both qubits restores the original state. On the Bell state, X on both qubits changes nothing anyway (|00⟩ + |11⟩ → |11⟩ + |00⟩), so only the rotation is affected.

This is the simplest **dynamical decoupling** (DD) sequence. Real devices use longer ones (XY4, for example) that also cancel errors around other axes. DD is **suppression**: it changes the circuit so less error happens, and it needs no extra circuits or shots.

**Your task:** write `idle_block(qc, n, dd)`. With `dd=False` it waits n steps as before. With `dd=True` it also adds an X on both qubits after `n // 2` steps and again after the last step. Keep exactly n `id` gates on each qubit."""),
code("""def idle_block(qc, n, dd=False):
    \"\"\"n idle steps on qubits 0 and 1 (an id gate on each per step). With dd=True, also an X on both
    qubits after n // 2 steps and after the last step.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete idle_block() first.")"""),
code("""DD = parity_expectation(probabilities(experiment(dd=True), nm, P_RO))
print(f"raw: {RAW:.3f}    with dynamical decoupling: {DD:.3f}")

deltas = np.linspace(0, 0.2, 21)
raw_curve = [parity_expectation(probabilities(experiment(False), noise_model(d, P_RO, P_CX), P_RO)) for d in deltas]
dd_curve = [parity_expectation(probabilities(experiment(True), noise_model(d, P_RO, P_CX), P_RO)) for d in deltas]
plt.figure(figsize=(6, 3.3))
plt.plot(deltas, raw_curve, "o-", ms=3, label="raw")
plt.plot(deltas, dd_curve, "s-", ms=3, label="dynamical decoupling")
plt.axvline(DELTA, color="gray", lw=0.8, ls=":")
plt.xlabel("detuning per step, delta (rad)"); plt.ylabel("<XX>"); plt.legend(fontsize=8); plt.tight_layout(); plt.show()

calm = noise_model(0, P_RO, P_CX)                  # no detuning: only incoherent errors left
print(f"without detuning: raw {parity_expectation(probabilities(experiment(False), calm, P_RO)):.4f}, "
      f"with DD {parity_expectation(probabilities(experiment(True), calm, P_RO)):.4f}")"""),
md("""**What to notice.** With the echo ⟨XX⟩ rises from 0.409 to **0.753**, the same for every detuning in the plot: the coherent error is gone (the lab check asks for this value). Without detuning, DD changes nothing, and even costs a little (two X gates per qubit with 0.1% error each): random dephasing does not reverse, so the echo cannot undo it. Suppression works on errors that are coherent, or that change slowly compared with the echo."""),

md("""## Step 4: readout mitigation

A readout error changes the recorded result, not the quantum state. If we know how often each result is misread, we can correct the *probabilities*. Prepare each of the four basis states with X gates, measure many times, and fill in the **calibration matrix** A: column j holds the probabilities of the four results when the bit string with index j was prepared ('00' → 0, '01' → 1, '10' → 2, '11' → 3). Then

p_measured = A p_true, so p_true = A⁻¹ p_measured.

This is **mitigation**: it runs extra circuits (the calibration) and corrects the average afterwards. The corrected "probabilities" can come out slightly negative; they are estimates, not a real distribution, and ⟨XX⟩ computed from them is still a good estimate.

**Your task:** write `mitigate_readout(probs, A)`. Put the four measured probabilities into a vector in the order '00', '01', '10', '11', solve A x = p (use `np.linalg.solve`), and return a dictionary with the same four keys."""),
code("""def calibration_matrix(nm, p_ro):
    A = np.zeros((4, 4))
    for j, key in enumerate(KEYS):
        cal = QuantumCircuit(2, 2)
        if key[1] == "1":
            cal.x(0)
        if key[0] == "1":
            cal.x(1)
        cal.measure(0, 0)
        cal.measure(1, 1)
        p = probabilities(cal, nm, p_ro)
        for i, k in enumerate(KEYS):
            A[i, j] = p[k]
    return A


A = calibration_matrix(nm, P_RO)
print("calibration matrix A (columns: prepared 00, 01, 10, 11; rows: measured):")
print(np.round(A, 4))"""),
code("""def mitigate_readout(probs, A):
    \"\"\"Return the readout-corrected probabilities: the solution x of A x = p_measured, as a dict with
    the keys '00', '01', '10', '11'.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete mitigate_readout() first.")"""),
code("""p_dd = probabilities(experiment(dd=True), nm, P_RO)
p_fix = mitigate_readout(p_dd, A)
print("measured :", {k: round(v, 4) for k, v in p_dd.items()})
print("corrected:", {k: round(v, 4) for k, v in p_fix.items()})
DD_RO = parity_expectation(p_fix)
print(f"<XX>: raw {RAW:.3f}, DD {DD:.3f}, DD and readout mitigation {DD_RO:.3f}")"""),
md("""**What to notice.** The calibration matrix is close to the identity but not equal to it: its columns show that a 1 is misread twice as often as a 0. Mitigation moves ⟨XX⟩ from 0.753 to **0.853**. The correction is only as good as the calibration: here the calibration circuits are exact, while on a device they are measured with shots and drift over time."""),

md("""## Step 5: zero-noise extrapolation

The CNOT error is random, so no echo removes it, and it happens inside the circuit, so no calibration matrix describes it. **Zero-noise extrapolation** (ZNE) uses a different idea: run the circuit with *more* noise on purpose, see how the result changes, and extrapolate back to *zero* noise.

To add noise without changing what the circuit does, **fold** the gate: a CNOT is its own inverse, so CNOT · CNOT · CNOT = CNOT. Three CNOTs in a row do the same as one, with about three times the error (scale 3); five give scale 5. Measure ⟨XX⟩ at scales 1, 3 and 5, fit a line (or a parabola), and read the fit at scale 0.

**Your tasks:**
- `fold_cx(qc, scale)`: return a **new** circuit (start from `qc.copy_empty_like()`) with every instruction of `qc` in order, except that each CNOT appears `scale` times in a row. Leave `qc` itself unchanged. (`qc.data` is the list of instructions; `inst.name` is the gate's name; `new.append(inst)` adds an instruction.)
- `extrapolate_zero(scales, values, degree=1)`: fit a polynomial of that degree with `np.polyfit(scales, values, degree)` and return its value at 0, which is the **last** coefficient."""),
code("""def fold_cx(qc, scale):
    \"\"\"A new circuit equal to qc, with every CNOT repeated `scale` times (scale odd).\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete fold_cx() first.")


def extrapolate_zero(scales, values, degree=1):
    \"\"\"Fit a polynomial of this degree to (scales, values) and return its value at scale 0.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete extrapolate_zero() first.")"""),
code("""SCALES = [1, 3, 5]
base = experiment(dd=True)
print("CNOTs at scale 1, 3, 5:", [fold_cx(base, s).count_ops().get("cx", 0) for s in SCALES])
points = [parity_expectation(mitigate_readout(probabilities(fold_cx(base, s), nm, P_RO), A)) for s in SCALES]
LINEAR = extrapolate_zero(SCALES, points, 1)
QUAD = extrapolate_zero(SCALES, points, 2)
target = parity_expectation(mitigate_readout(probabilities(base, noise_model(DELTA, P_RO, 0.0), P_RO),
                                             calibration_matrix(noise_model(DELTA, P_RO, 0.0), P_RO)))
for s, v in zip(SCALES, points):
    print(f"scale {s}: <XX> = {v:.4f}")
print(f"zero-noise estimate: linear {LINEAR:.3f}, quadratic {QUAD:.3f};  the same circuit with no CNOT error: {target:.3f}")

xs = np.linspace(0, 5.5, 50)
plt.figure(figsize=(6, 3.3))
plt.plot(SCALES, points, "o", label="DD and readout mitigation, measured")
plt.plot(xs, np.polyval(np.polyfit(SCALES, points, 1), xs), "--", lw=1, label="linear fit")
plt.plot(xs, np.polyval(np.polyfit(SCALES, points, 2), xs), ":", lw=1, label="quadratic fit")
plt.plot([0], [target], "k*", ms=9, label="no CNOT error")
plt.xlabel("noise scale (number of CNOTs)"); plt.ylabel("<XX>"); plt.legend(fontsize=8); plt.tight_layout(); plt.show()"""),
md("""**What to notice.** Each extra pair of CNOTs lowers ⟨XX⟩ by about 0.03. Extrapolating to scale 0 gives **0.869** with a line (the lab check asks for it) and 0.870 with a parabola, against 0.870 for the same circuit without any CNOT error: ZNE removed almost all of the CNOT's effect. The line is a little low because depolarizing noise shrinks ⟨XX⟩ by a constant *factor* per CNOT, a curve that bends upward towards scale 0.

All three steps together raised ⟨XX⟩ from 0.409 to about 0.87, but not to 1. What is left is the dephasing while the qubits wait, which is random, not folded and not calibrated, so none of the three tools touches it. That needs shorter circuits, better qubits, or error correction."""),

md("""## Step 6: the price of mitigation

So far every number was exact. A real computer gives **shots**, and every estimate has a statistical error. Mitigation can bring the average much closer to the ideal value (it reduces the bias, as far as its assumptions hold), but it makes the scatter larger: inverting A amplifies small fluctuations, and extrapolation from three points amplifies them again (the fitted value at 0 lies outside the measured range).

The cell repeats the whole experiment 300 times with 4,000 shots per circuit (sampled from the exact probabilities; the calibration matrix is measured with shots too) and compares the spread of three estimates of the same circuit with DD: no mitigation, readout mitigation, and readout mitigation with ZNE. *Predict first:* which estimate scatters the most?"""),
code("""rng = np.random.default_rng(2026)
SHOTS, REPEATS = 4000, 300
exact = {"raw": probabilities(experiment(False), nm, P_RO)}
exact.update({s: probabilities(fold_cx(base, s), nm, P_RO) for s in SCALES})
cal_exact = [A[:, j] for j in range(4)]           # the exact calibration columns of Step 4


def sample(p):
    n = rng.multinomial(SHOTS, [p[k] for k in KEYS])
    return {k: n[i] / SHOTS for i, k in enumerate(KEYS)}


est = {"DD": [], "DD + readout": [], "DD + readout + ZNE": []}
for _ in range(REPEATS):
    A_shots = np.column_stack([rng.multinomial(SHOTS, col / col.sum()) / SHOTS for col in cal_exact])
    samples = {s: sample(exact[s]) for s in SCALES}
    est["DD"].append(parity_expectation(samples[1]))
    pts = [parity_expectation(mitigate_readout(samples[s], A_shots)) for s in SCALES]
    est["DD + readout"].append(pts[0])
    est["DD + readout + ZNE"].append(extrapolate_zero(SCALES, pts, 1))
for name, vals in est.items():
    print(f"{name:20s} mean {np.mean(vals):.3f}   standard deviation {np.std(vals):.4f}")
ratio = np.std(est["DD + readout + ZNE"]) / np.std(est["DD"])
print(f"\\nthe mitigated estimate scatters {ratio:.1f} times as much as the unmitigated one: "
      f"{ratio**2:.1f} times the shots per circuit for the same precision, and three circuits instead of one")"""),
md("""**What to notice.** The means agree with the exact values of Steps 3 to 5 (0.753, 0.853 and 0.869) within their scatter, but the scatter grows. Readout mitigation scatters about 1.3 times as much as no mitigation, and ZNE about 1.5 times as much, so ZNE needs more than twice the shots per circuit for the same precision, on three circuits instead of one, plus the calibration. Here the noise is mild; the stronger the noise that mitigation removes, the larger this factor, and for ZNE on large circuits it grows quickly. This is the general rule: **mitigation trades bias for variance**. It is why mitigation works for small circuits and moderate noise, and why large computations will need error correction."""),

md("""## Step 7: a device's noise

The file loaded in Step 0 holds the same experiment run on two coupled qubits of **FakePittsburgh** (ibm_pittsburgh's calibration of 17 April 2026, qubits 97 and 107), 40,000 shots per circuit, simulated with Qiskit Aer. The idle period is a real 20 µs delay, and the X echo comes after 10 µs. The saved circuits are: raw, with DD, with DD and the CNOT folded 3 and 5 times, and the four calibration circuits.

The cell uses **your** `mitigate_readout` and `extrapolate_zero` on these counts."""),
code("""counts, shots = run["counts"], run["shots"]
probs = {name: {k: n / shots for k, n in c.items()} for name, c in counts.items()}
A_dev = np.array([[probs[f"cal_{j}"][i] for j in KEYS] for i in KEYS])
print("qubits", run["physical_qubits"], {q: (p["t1_us"], p["t2_us"], p["readout_error"]) for q, p in run["qubit_properties"].items()},
      " CZ error", run["cz_error"])
print("calibration matrix:")
print(np.round(A_dev, 4))

dev_raw = parity_expectation(probs["raw"])
dev_dd = parity_expectation(probs["dd"])
dev_pts = [parity_expectation(mitigate_readout(probs[n], A_dev)) for n in ("dd", "dd_fold3", "dd_fold5")]
dev_zne = extrapolate_zero(SCALES, dev_pts, 1)
se = 1 / np.sqrt(shots)
print(f"\\nraw {dev_raw:.3f}   DD {dev_dd:.3f}   DD + readout {dev_pts[0]:.3f}   DD + readout + ZNE {dev_zne:.3f}")
print(f"(each raw value has a statistical error of about {se:.3f})")
t2 = [run["qubit_properties"][q]["t2_us"] for q in map(str, run["physical_qubits"])]
print(f"dephasing during {run['idle_us']} us alone: exp(-t/T2a - t/T2b) = {np.exp(-run['idle_us'] / t2[0] - run['idle_us'] / t2[1]):.3f}")"""),
md("""**What to notice.** This noise model has relaxation, depolarizing gate errors and readout errors, but **no coherent idle errors**, so DD changes the result only within the statistical error (0.884 → 0.887). Readout mitigation helps most, to **0.900** (the lab check asks for this value), and ZNE adds little because this device's CZ error is only about 0.2%. What remains is again the dephasing during the 20 µs delay, close to exp(−t/T2) for the two qubits.

On real hardware the picture is often different: low-frequency noise and crosstalk between neighbouring qubits act like the detuning in Step 2, and DD helps clearly. The course team can replace this file with a real run of the same circuits; the analysis stays the same."""),

md("""## Step 8: what IBM's Estimator does for you

With Qiskit on IBM hardware you rarely write these functions yourself. The **Estimator** primitive computes expectation values and has options for exactly the tools of this lab. In qiskit-ibm-runtime 0.50 it is `qiskit_ibm_runtime.executor_estimator.Estimator`, which prepares the extra circuits and does the mitigation on your own computer (the older `EstimatorV2` takes the same options but is deprecated):

- `options.dynamical_decoupling.enable = True` inserts DD sequences (`sequence_type` "XX", "XpXm" or "XY4") into every idle period;
- `options.resilience.measure_mitigation = True` corrects readout errors; IBM's method, **TREX** (twirled readout error extinction), randomizes the measurement so that no full calibration matrix is needed;
- `options.resilience.zne_mitigation = True` runs ZNE, with `zne.noise_factors` such as (1, 3, 5) and an `extrapolator` such as "linear" or "exponential"; IBM amplifies noise by gate folding or by learning the noise first (probabilistic error amplification, **PEA**);
- `options.resilience_level` sets them in bulk: 0 is none, 1 (the default) is readout mitigation, 2 adds ZNE.

Each option costs shots or time, as in Step 6. Probabilistic error cancellation (**PEC**) goes further: if the noise model it learns is exact, it removes the bias completely, at a cost that grows exponentially with circuit size.

The optional Colab notebook on the Module 3 page runs this experiment with the Estimator and these options, on a real IBM computer if you have an account, or on a small simulated device."""),

md("""## Step 9: your personal check

Open the **Module 3 lab check** in Canvas. Question 1 shows your own DELTA (0.02 to 0.15), P_RO (0.01 to 0.06) and P_CX (0.01 to 0.08). Type them below and run the next two cells. The check cell tests `idle_block()`, `mitigate_readout()`, `fold_cx()` and `extrapolate_zero()`. Only if every test passes does it print your **verification value**: 1,000 times the improvement in ⟨XX⟩ from all three tools together (DD, readout mitigation and linear ZNE from scales 1, 3 and 5) over the raw value, for your noise, rounded to a whole number. For the lab's numbers (0.05, 0.02, 0.02) it is 1,000 × (0.869 − 0.409) = 460."""),
code("""DELTA = 0.0    # your number from Canvas, for example 0.05
P_RO = 0.0     # for example 0.02
P_CX = 0.0     # for example 0.02"""),
code(hidden_check(['l3_m3_check'], 'check_l3_module3', '''

passed, messages, value = check_l3_module3(idle_block, mitigate_readout, fold_cx, extrapolate_zero, DELTA, P_RO, P_CX)
for m in messages:
    print(m)
if passed:
    print("\\nAll tests passed. Your verification value is", value)
else:
    print("\\nNot yet: fix the item above and run this cell again.")''')),

md("""## What you should notice

- **Suppression** changes the circuit so less error happens. Dynamical decoupling cancels coherent errors on waiting qubits, at almost no cost, but cannot undo random dephasing.
- **Mitigation** corrects the average afterwards. Readout mitigation inverts a calibration matrix; zero-noise extrapolation amplifies the noise on purpose and extrapolates to zero. Both cost extra circuits, and both increase the statistical error.
- Neither removes every error: here the dephasing during the wait remains. **Error correction** (Module 2) is the only route to arbitrarily long computations, but it needs many more qubits.
- On IBM hardware the Estimator's options apply these tools for you; knowing what each one does tells you what to switch on and what it will cost.

**Next in Canvas:** the lab check, the quiz and the time log."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl3m3c{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / NAME)
print("wrote", OUT / NAME)
