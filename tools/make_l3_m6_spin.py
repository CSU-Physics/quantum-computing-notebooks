"""Build the Level 3 Module 6 capstone notebook "Spin-chain simulation with mitigation".
Numbers quoted in the text are computed here from checks/l3_m6_spin_check.py (exact, course noise model)."""
import math
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "level3"), str(ROOT / "checks"), str(ROOT / "tools")]
import l3_m6_spin_check as P  # noqa: E402
from l3_m6_common import exact_probabilities  # noqa: E402
from l3_hidden import hidden_check  # noqa: E402
from l3_m6_text import META, check_cell, code, ending, intro, md, personal_step, step0, step1_statistics, step2_noise  # noqa: E402

NAME = "QC-L3-M6-capstone-spin-chain.ipynb"
P2, RO, SHOTS = 0.01, 0.02, 4000
STEP_LIST = [2, 4, 6, 8, 10, 12]
LIN = [13 / 12, 1 / 3, -5 / 12]


def msig(probs, shots=SHOTS):
    n = len(next(iter(probs)))
    m = lambda k: sum(1 - 2 * int(b) for b in k) / n  # noqa: E731
    M = sum(v * m(k) for k, v in probs.items())
    return math.sqrt((sum(v * m(k) ** 2 for k, v in probs.items()) - M * M) / shots)


EXACT = P.exact_magnetization(3, 0.7, 2.0)
ST = {s: P.study(s, P2, RO) for s in STEP_LIST}
SIG = {}
for s in STEP_LIST:
    base = P.reference_trotter_circuit(3, 0.7, 2.0, s)
    sg = [msig(exact_probabilities(P.measured(P.reference_fold_rzz(base, k)), P2, RO)) / (1 - 3 * RO) for k in (1, 3, 5)]
    SIG[s] = (sg[0], math.sqrt(sum((c * x) ** 2 for c, x in zip(LIN, sg))))
best_raw = min(STEP_LIST, key=lambda s: abs(ST[s]["raw"][1] - EXACT))
best_mit = min(STEP_LIST, key=lambda s: abs(ST[s]["mitigated"] - EXACT))
print("exact", round(EXACT, 4), "best raw", best_raw, "best mitigated", best_mit,
      {s: (round(ST[s]["ideal"], 4), round(ST[s]["raw"][1], 4), round(ST[s]["corrected"][1], 4), round(ST[s]["mitigated"], 4),
           round(SIG[s][0], 4), round(SIG[s][1], 4)) for s in STEP_LIST})
assert best_mit == 6
assert max(ST[s]["mitigated"] for s in (4, 6, 8, 10)) - min(ST[s]["mitigated"] for s in (4, 6, 8, 10)) < SIG[8][1]
TS = [(0.25 * k, P.exact_magnetization(3, 0.7, 0.25 * k), P.study(k, P2, RO, T=0.25 * k)["mitigated"]) for k in range(1, 13)]
gaps = [e - m for _, e, m in TS]
print("time-series gaps", [round(g, 4) for g in gaps])
s8 = ST[8]

cells = [
intro("Spin-chain simulation with mitigation",
      "How well can a noisy quantum computer follow the magnetization of a three-spin chain in time, how many Trotter steps "
      "should it use when every step adds noise, and how much do readout correction and zero-noise extrapolation recover?",
      """1. write the statistics functions every capstone uses (Step 1) and meet the capstone noise model (Step 2);
2. build the **Trotter circuit** of the transverse-field Ising chain and measure the **magnetization** (Step 3);
3. find the **trade-off** between Trotter error and gate noise as the number of steps grows (Step 4);
4. correct the **readout** error of the magnetization (Step 5);
5. fold the two-qubit gates and **extrapolate** to zero noise (Step 6);
6. follow M(t) from t = 0.25 to 3 with error bars (Step 7);
7. get your verification value (Step 8).""",
      "Question 1 gives you three personal numbers: STEPS, P2 and READOUT."),
*step0(),
*step1_statistics(),
*step2_noise(),

md(f"""## Step 3: the spin chain and its magnetization

The model is the one of Module 4 Track B: an open chain of N = 3 spins with the Hamiltonian

H = −J Σ Zᵢ Zᵢ₊₁ − h Σ Xᵢ, with J = 1 and h = 0.7.

All spins start up, |000⟩, and the quantity you follow is the **magnetization** M(t) = (1/N) Σ ⟨Zᵢ⟩, which starts at 1. Its exact value at T = 2 is **{EXACT:.4f}** (from the matrix exponential of H, computed in the next cells).

A quantum computer cannot apply e^(−iHT) in one gate. The **first-order Trotter** circuit splits T into `steps` steps of length dt = T / steps, and in each step applies the two parts of H one after the other:

- e^(+i dt Zᵢ Zᵢ₊₁) on every neighbouring pair: **RZZ(−2 dt)**, because RZZ(θ) = e^(−iθ Z Z / 2);
- e^(+i h dt Xᵢ) on every spin: **RX(−2 h dt)**.

The two parts do not commute, so the circuit is only approximately right: the **Trotter error** shrinks as dt shrinks.

From shots, each qubit's ⟨Zᵢ⟩ is (fraction of 0s) − (fraction of 1s). Qubit i is character N − 1 − i of a key, but M averages over all qubits, so the order does not matter here.

**Your task:**

- `trotter_circuit(N, h, T, steps)`: the circuit on N qubits; in every step, all the RZZ gates first (`qc.rzz(theta, i, i + 1)`), then the RX gates (`qc.rx(theta, i)`). **No measurements**: Step 6 folds the circuit.
- `magnetization(probs)`: (1/N) Σᵢ ⟨Zᵢ⟩ from counts or probabilities; divide by the total weight.

The prepared `magnetization_sigma(probs, shots)` gives the σ of M from shots. M is not a ±1 average (each shot gives one of 1, 1/3, −1/3, −1), so `expectation_sigma` does not apply: σ = √((⟨m²⟩ − M²) / shots), where m is the value of one shot."""),
code("""def trotter_circuit(N, h, T, steps):
    \"\"\"`steps` first-order Trotter steps of dt = T / steps: RZZ(-2 dt) on each pair, then RX(-2 h dt) on each qubit.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete trotter_circuit() first.")


def magnetization(probs):
    \"\"\"(1/N) sum_i <Z_i> from counts or probabilities.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete magnetization() first.")"""),
code("""N, H_FIELD, T = 3, 0.7, 2.0


def measured(qc):
    \"\"\"Prepared: a copy of qc with every qubit measured (qubit i into classical bit i).\"\"\"
    out = QuantumCircuit(qc.num_qubits, qc.num_qubits)
    out.compose(qc, inplace=True)
    for i in range(qc.num_qubits):
        out.measure(i, i)
    return out


def magnetization_sigma(probs, shots):
    \"\"\"Prepared: the standard deviation of M measured with `shots` shots, from the probabilities of each result.\"\"\"
    total = sum(probs.values())
    n = len(next(iter(probs)))
    m = {k: sum(1 - 2 * int(b) for b in k) / n for k in probs}
    M = sum(v * m[k] for k, v in probs.items()) / total
    m2 = sum(v * m[k] ** 2 for k, v in probs.items()) / total
    return math.sqrt(max(m2 - M * M, 0.0) / shots)


def exact_magnetization(N, h, t):
    \"\"\"Prepared: the exact M(t) from the matrix exponential of H (no Trotter error, no noise).\"\"\"
    X, Z, I2 = np.array([[0, 1], [1, 0]]), np.diag([1.0, -1.0]), np.eye(2)

    def op(single, i):
        m = np.array([[1.0]])
        for q in reversed(range(N)):
            m = np.kron(m, single if q == i else I2)
        return m
    H = -sum(op(Z, i) @ op(Z, i + 1) for i in range(N - 1)) - h * sum(op(X, i) for i in range(N))
    w, v = np.linalg.eigh(H)
    psi0 = np.zeros(2 ** N); psi0[0] = 1
    psi = v @ (np.exp(-1j * w * t) * (v.conj().T @ psi0))
    return float(np.real(np.vdot(psi, (sum(op(Z, i) for i in range(N)) / N) @ psi)))


print(trotter_circuit(N, H_FIELD, T, 2).draw())
M_EXACT = exact_magnetization(N, H_FIELD, T)
print("Exact M(2.0) =", round(M_EXACT, 4))
for steps in [2, 4, 6, 8, 10, 12]:
    ideal = magnetization(exact_probabilities(measured(trotter_circuit(N, H_FIELD, T, steps)), 0, 0))
    print(f"   {steps:2d} Trotter steps, no noise: M = {ideal:.4f}   Trotter error {ideal - M_EXACT:+.4f}")"""),
md(f"""**What to notice.** With 2 steps (dt = 1) the Trotter circuit gives {ST[2]['ideal']:.4f}, far from {EXACT:.4f}; with 8 steps it gives {ST[8]['ideal']:.4f}, and the error keeps shrinking roughly as dt. Without noise, more steps is always better."""),

md("""## Step 4: more steps, more noise

Every step adds N − 1 = 2 RZZ gates (each a two-qubit gate with error P2) and N = 3 RX gates. So more steps means a smaller Trotter error but more gate noise. Run the cell: for 2 to 12 steps it gives the noiseless Trotter value, the exact noisy value, and a run of 4,000 shots with its σ."""),
code("""print(f"{'steps':>6s}{'RZZ':>5s}{'no noise':>10s}{'noisy, exact':>14s}{'your run':>18s}{'error of noisy':>16s}")
raw_exact, probs_by_steps = {}, {}
for steps in [2, 4, 6, 8, 10, 12]:
    qc = trotter_circuit(N, H_FIELD, T, steps)
    ideal = magnetization(exact_probabilities(measured(qc), 0, 0))
    probs_by_steps[steps] = exact_probabilities(measured(qc))
    raw_exact[steps] = magnetization(probs_by_steps[steps])
    run = magnetization(sample_counts(probs_by_steps[steps], 4000, seed=steps))
    sg = magnetization_sigma(probs_by_steps[steps], 4000)
    print(f"{steps:6d}{2 * steps:5d}{ideal:10.4f}{raw_exact[steps]:14.4f}{run:11.4f} +- {sg:.4f}{raw_exact[steps] - M_EXACT:+16.4f}")
print("Exact M(2.0) =", round(M_EXACT, 4))"""),
md(f"""**What to notice.** The noisy value first gets closer to the exact {EXACT:.4f} as the Trotter error shrinks, then falls away as the gate noise builds up: the raw value is best at about **{best_raw} steps** ({ST[best_raw]['raw'][1]:.4f}), and at 8 steps it is **{s8['raw'][1]:.4f}**. This trade-off decides the number of steps on every real device: the best dt is set by the noise, not by the mathematics. Noise pulls every ⟨Zᵢ⟩ toward 0, so it pulls M toward 0, here downward. With σ ≈ {SIG[8][0] * (1 - 3 * RO):.3f} for 4,000 shots, neighbouring step numbers are hard to tell apart from a single run."""),

md("""## Step 5: correct the readout

For one qubit, the capstone readout records a 0 as 1 with probability r₀ = READOUT and a 1 as 0 with r₁ = 2 · READOUT. If p₀ is the true probability of 0 and z = p₀ − p₁ the true ⟨Z⟩, the measured value is

z_measured = (1 − r₀ − r₁) · z + (r₁ − r₀).

The factor (1 − r₀ − r₁) shrinks ⟨Z⟩; the offset (r₁ − r₀) appears because 1s are misread more often than 0s, which pushes ⟨Z⟩ up. M is an average of ⟨Zᵢ⟩, so the same relation holds for M, and you can **invert it**:

z = (z_measured − (r₁ − r₀)) / (1 − r₀ − r₁).

This is the readout mitigation of the mitigation capstone, done on the expectation value instead of on the probabilities. It is exact for this readout model because M needs only one qubit at a time. It also scales the uncertainty: σ_corrected = σ / (1 − r₀ − r₁).

**Your task:** `correct_readout_z(z, readout)`: the corrected value."""),
code("""def correct_readout_z(z, readout):
    \"\"\"<Z> (or M) corrected for readout errors r0 = readout (0 read as 1) and r1 = 2 readout (1 read as 0).\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete correct_readout_z() first.")"""),
code("""for steps in [2, 8, 12]:
    print(f"{steps:2d} steps: raw {raw_exact[steps]:.4f}  ->  readout-corrected {correct_readout_z(raw_exact[steps], READOUT_COURSE):.4f}")
print("Fixed point: correct_readout_z(1/3, 0.02) =", round(correct_readout_z(1 / 3, 0.02), 4))
print("At t = 0 (M = 1, no gates): measured", round(magnetization(exact_probabilities(measured(QuantumCircuit(3)))), 4),
      " corrected", round(correct_readout_z(magnetization(exact_probabilities(measured(QuantumCircuit(3)))), 0.02), 4))"""),
md(f"""**What to notice.** Here the readout correction changes almost nothing (8 steps: {s8['raw'][1]:.4f} to {s8['corrected'][1]:.4f}). That is not a mistake. The readout maps z to r₀ + (1 − 3r₀) z, which leaves z = 1/3 unchanged, and M(2.0) happens to be close to 1/3. At t = 0, where M = 1, the readout error is large ({1 - (1 - 3 * RO) - RO:.2f}) and the correction removes it completely. A correction that does nothing at one point can still matter elsewhere: Step 7 shows it over the whole time range. The remaining error at T = 2 comes from the gates."""),

md("""## Step 6: fold the two-qubit gates and extrapolate

The gate noise here comes mostly from the RZZ gates (error P2, against P2/10 for each RX). **Local folding** amplifies only them: every RZZ(θ) becomes

RZZ(θ), then (scale − 1)/2 pairs RZZ(−θ), RZZ(θ).

Ideally RZZ(−θ) RZZ(θ) does nothing, so the circuit is unchanged, but each RZZ makes its error, so scale 3 has three times the RZZ noise. Then **extrapolate**: fit a straight line through (scale, corrected M) at scales 1, 3, 5 by least squares and read it at scale 0.

**Your tasks:**

- `fold_rzz(qc, scale)`: a new circuit (`out = qc.copy_empty_like()`), going through `qc.data`: for an instruction with `inst.name == "rzz"`, take `th = float(inst.params[0])` and the qubits `a, b = inst.qubits`, add `out.rzz(th, a, b)` and the pairs; any other instruction is appended unchanged (`out.data.append(inst)`).
- `extrapolate_linear(scales, values)`: the intercept of the least-squares line, for example `np.polyfit(scales, values, 1)[1]`."""),
code("""def fold_rzz(qc, scale):
    \"\"\"Every RZZ(theta) followed by (scale - 1) / 2 pairs RZZ(-theta), RZZ(theta); other gates unchanged.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete fold_rzz() first.")


def extrapolate_linear(scales, values):
    \"\"\"The least-squares straight line through (scale, value), read at scale 0.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete extrapolate_linear() first.")"""),
code("""SCALES, LIN = [1, 3, 5], [13 / 12, 1 / 3, -5 / 12]     # the linear-extrapolation weights of the mitigation capstone


def mitigated_run(steps, t, shots=4000, seed=0):
    \"\"\"Corrected and folded: the exact mitigated M, and one run of `shots` shots per scale with its sigma.\"\"\"
    base = trotter_circuit(N, H_FIELD, t, steps)
    ex = {s: exact_probabilities(measured(fold_rzz(base, s))) for s in SCALES}
    corr = {s: correct_readout_z(magnetization(ex[s]), READOUT_COURSE) for s in SCALES}
    run = {s: correct_readout_z(magnetization(sample_counts(ex[s], shots, seed + s)), READOUT_COURSE) for s in SCALES}
    sig = [magnetization_sigma(ex[s], shots) / (1 - 3 * READOUT_COURSE) for s in SCALES]
    return (extrapolate_linear(SCALES, [corr[s] for s in SCALES]), extrapolate_linear(SCALES, [run[s] for s in SCALES]),
            combined_sigma(LIN, sig), corr)


print(f"{'steps':>6s}{'corrected, scales 1, 3, 5':>28s}{'mitigated, exact':>18s}{'your run':>18s}{'error':>9s}")
mit = {}
for steps in [2, 4, 6, 8, 10, 12]:
    m_exact, m_run, sg, corr = mitigated_run(steps, T, seed=100 * steps)
    mit[steps] = m_exact
    vals = ", ".join(f"{corr[s]:.3f}" for s in SCALES)
    print(f"{steps:6d}{vals:>28s}{m_exact:18.4f}{m_run:11.4f} +- {sg:.4f}{m_exact - M_EXACT:+9.4f}")
print("Exact M(2.0) =", round(M_EXACT, 4), "  best number of steps after mitigation:", min(mit, key=lambda s: abs(mit[s] - M_EXACT)))"""),
md(f"""**What to notice.** Mitigation moves the 8-step value from {s8['corrected'][1]:.4f} to **{s8['mitigated']:.4f}**, and the best number of steps rises to **{best_mit}** ({ST[best_mit]['mitigated']:.4f}): once the noise is partly removed, more steps can be afforded. What remains (about {EXACT - ST[best_mit]['mitigated']:.3f} at {best_mit} steps) is the Trotter error, the RX noise that was not folded, and the curvature that a straight line misses. The price is in σ: about {SIG[8][1]:.3f} for the mitigated value against {SIG[8][0] * (1 - 3 * RO):.3f} for the raw one, with three times the shots. Compare the error column with σ: from 4 to 10 steps the exact mitigated values differ by less than one σ, so a single run could not pick the best one; the exact noise model can."""),

md("""## Step 7: the magnetization in time

A simulation is usually wanted at many times. Run the cell: with a fixed dt = 0.25 (so t / 0.25 steps), it computes M(t) for t = 0.25 to 3: exact, raw (4,000 shots), readout-corrected, and fully mitigated (4,000 shots per scale), with ±σ error bars."""),
code("""times = [0.25 * k for k in range(1, 13)]
rows = []
for k, t in enumerate(times, start=1):
    qc = trotter_circuit(N, H_FIELD, t, k)
    ex1 = exact_probabilities(measured(qc))
    raw_run = magnetization(sample_counts(ex1, 4000, seed=500 + k))
    raw_sig = magnetization_sigma(ex1, 4000)
    m_exact, m_run, m_sig, _ = mitigated_run(k, t, seed=600 + 10 * k)
    rows.append((t, exact_magnetization(N, H_FIELD, t), raw_run, raw_sig, correct_readout_z(raw_run, READOUT_COURSE), m_run, m_sig))
    print(f"t = {t:4.2f}: exact {rows[-1][1]:.4f}   raw {raw_run:.4f} +- {raw_sig:.4f}   corrected {rows[-1][4]:.4f}"
          f"   mitigated {m_run:.4f} +- {m_sig:.4f}")

fine = np.linspace(0, 3, 121)
plt.figure(figsize=(6.5, 3.8))
plt.plot(fine, [exact_magnetization(N, H_FIELD, t) for t in fine], "k-", lw=1, label="exact")
r = np.array(rows)
plt.errorbar(r[:, 0], r[:, 2], yerr=r[:, 3], fmt="s", ms=4, capsize=2, label="raw")
plt.errorbar(r[:, 0] + 0.03, r[:, 5], yerr=r[:, 6], fmt="o", ms=4, capsize=2, label="corrected + ZNE")
plt.xlabel("t"); plt.ylabel("M(t)"); plt.legend(); plt.title("3 spins, h = 0.7, dt = 0.25, 4,000 shots per circuit")
plt.tight_layout(); plt.show()

within = sum(abs(row[5] - row[1]) <= 3 * row[6] for row in rows)
print(f"Mitigated points within 3 sigma of the exact curve: {within} of {len(rows)}")"""),
md(f"""**What to notice.** The raw points sit below the exact curve at every time, and the gap grows with t, because longer times need more steps and more gates. At early times much of the gap is readout (M is close to 1), and the readout correction removes it; later the gate noise dominates. The mitigated points are much closer to the curve, with error bars about 1.3 times the raw ones, but not on it: even without shot noise, the mitigated values lie {min(gaps):.3f} to {max(gaps):.3f} below the exact curve, up to about two σ of 4,000 shots. So in a run, a few mitigated points may miss the curve by more than 3σ. A bias of one or two σ shows clearly only when many points miss in the same direction, as they do here, or when you take more shots.

The dip of M around t ≈ 1.6 and the rise after it are the physics: the transverse field turns the spins away from Z, and the coupling brings them partly back. A noisy device that only showed the raw points would get the shape right and the size wrong."""),

*personal_step(8, ["STEPS", "P2", "READOUT"], "STEPS (2 to 12), P2 and READOUT (0.005 to 0.030)",
               "STEPS, P2, READOUT = 8, 0.01, 0.02 gives " + str(P.personal_value(8, 0.01, 0.02)), None, None,
               "10,000 times the exact mitigated magnetization at T = 2 (N = 3, h = 0.7) for your number of Trotter steps and "
               "your noise model: readout-corrected, RZZ gates folded at scales 1, 3 and 5, extrapolated linearly, rounded to a "
               "whole number."),
check_cell(hidden_check, "l3_m6_spin_check", "check_l3_m6_spin",
           "fraction_sigma, expectation_sigma, combined_sigma, trotter_circuit, magnetization,\n"
           "                                           correct_readout_z, fold_rzz, extrapolate_linear, STEPS, P2, READOUT"),
*ending("""- **Three spins.** A three-spin chain is easy for a classical computer, which is why you can compare with the exact answer. A useful quantum simulation needs a chain too large for that, and then the remaining bias can no longer be seen directly.
- **Only the RZZ gates were folded.** The RX gates and the readout are not amplified, so their part of the error is not extrapolated away. On hardware, RZZ is itself made of two CX gates (or one native two-qubit gate) and single-qubit gates, and the compiler may merge or cancel folded gates.
- **The noise model is simple.** Depolarizing errors and independent readout errors. Real devices also have coherent errors, which can add up step after step in a Trotter circuit, and crosstalk between neighbouring qubits.
- **First-order Trotter.** A second-order formula (half a step of one part, a full step of the other, half a step of the first) has a much smaller Trotter error for the same number of two-qubit gates, so it would favour fewer steps.
- **One quantity.** M averages three ⟨Zᵢ⟩. Correlations such as ⟨Zᵢ Zⱼ⟩ are more sensitive to noise and need more shots for the same σ.""",
       "the **Capstone Quiz — Spin Chain**"),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl3m6s{i:02d}"
out = ROOT / "content" / "level3" / NAME
nbf.write(nb, out)
print("wrote", out)
