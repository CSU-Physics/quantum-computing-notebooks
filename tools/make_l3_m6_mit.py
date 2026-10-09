"""Build the Level 3 Module 6 capstone notebook "Mitigation study: one expectation value".
Numbers quoted in the text are computed here from checks/l3_m6_mit_check.py (exact, course noise model)."""
import math
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "level3"), str(ROOT / "checks"), str(ROOT / "tools")]
import l3_m6_mit_check as M  # noqa: E402
from l3_m6_common import exact_probabilities  # noqa: E402
from l3_hidden import hidden_check  # noqa: E402
from l3_m6_text import META, check_cell, code, ending, intro, md, personal_step, step0, step1_statistics, step2_noise  # noqa: E402

NAME = "QC-L3-M6-capstone-mitigation.ipynb"
P2, RO, SHOTS = 0.01, 0.02, 4000


# ---------------------------------------------------------------- numbers for the text (exact)
def exact_sigma(n, scale, mitigate, shots=SHOTS):
    """The exact standard deviation of the (readout-mitigated) parity from `shots` shots: the estimator is linear in
    the measured fractions, E = sum w_k f_k, so var = (sum w_k^2 p_k - E^2) / shots."""
    p = exact_probabilities(M.measured(M.reference_fold_global(M.reference_ghz_circuit(n), scale)), P2, RO)
    keys = [format(i, f"0{n}b") for i in range(2 ** n)]
    w = []
    for k in keys:
        unit = {x: (1.0 if x == k else 0.0) for x in keys}
        w.append(M.reference_parity_expectation(M.reference_mitigate_readout(unit, RO) if mitigate else unit))
    pv = np.array([p.get(k, 0.0) for k in keys])
    E = float(np.dot(w, pv))
    return math.sqrt((float(np.dot(np.square(w), pv)) - E * E) / shots)


S = {n: M.study(n, P2, RO) for n in range(2, 6)}
s3 = S[3]
LIN = M.reference_zne_coefficients([1, 3, 5], 1)
QUAD = M.reference_zne_coefficients([1, 3, 5], 2)
sig_raw = exact_sigma(3, 1, False)
sig_ro = {s: exact_sigma(3, s, True) for s in (1, 3, 5)}
sig_lin = math.sqrt(sum((c * sig_ro[s]) ** 2 for c, s in zip(LIN, (1, 3, 5))))
sig_quad = math.sqrt(sum((c * sig_ro[s]) ** 2 for c, s in zip(QUAD, (1, 3, 5))))
bias = {"raw": 1 - s3["raw"][1], "ro": 1 - s3["ro"][1], "lin": 1 - s3["zne_lin"], "quad": 1 - s3["zne_quad"]}
# per-shot variances (sigma^2 * shots), for a budget B of total shots: ZNE spends B / 3 per scale
v1 = {"raw": sig_raw ** 2 * SHOTS, "ro": sig_ro[1] ** 2 * SHOTS, "lin": sig_lin ** 2 * 3 * SHOTS, "quad": sig_quad ** 2 * 3 * SHOTS}
cross_ro_lin = (v1["lin"] - v1["ro"]) / (bias["ro"] ** 2 - bias["lin"] ** 2)
cross_lin_quad = (v1["quad"] - v1["lin"]) / (bias["lin"] ** 2 - bias["quad"] ** 2)
NUM = dict(raw=s3["raw"], ro=s3["ro"], lin=s3["zne_lin"], quad=s3["zne_quad"], lin_raw=s3["zne_lin_raw"],
           sig_raw=sig_raw, sig_ro=sig_ro, sig_lin=sig_lin, sig_quad=sig_quad, cross1=cross_ro_lin, cross2=cross_lin_quad)
print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in NUM.items()})
print("n = 2..5:", {n: (round(S[n]["raw"][1], 4), round(S[n]["ro"][1], 4), round(S[n]["zne_lin"], 4)) for n in S})


def r2(x):
    """Round to 2 significant figures as a whole number, for the shot budgets in the text."""
    e = int(math.floor(math.log10(x))) - 1
    return int(round(x, -e))


cells = [
intro("Mitigation study: one expectation value",
      "How close to the ideal value can you bring one noisy expectation value, ⟨X X X⟩ of a three-qubit GHZ state, with "
      "readout mitigation and zero-noise extrapolation, how certain is each mitigated result, and when is mitigation worth "
      "its cost in shots?",
      """1. write the statistics functions every capstone uses (Step 1) and meet the capstone noise model (Step 2);
2. build the GHZ circuit and measure **⟨X X X⟩** from shots (Step 3);
3. remove the **readout error** by inverting the readout confusion (Step 4);
4. amplify the gate noise by **global folding** (Step 5);
5. extrapolate to zero noise with **weights** you compute (Step 6);
6. measure the uncertainty of every result with the **bootstrap** and decide which method is best for a given **shot budget** (Step 7);
7. repeat the study for 2 to 5 qubits (Step 8);
8. get your verification value (Step 9).""",
      "Question 1 gives you three personal numbers: NQ, P2 and READOUT."),
*step0(),
*step1_statistics(),
*step2_noise(),

md("""## Step 3: the expectation value

The **GHZ state** (|000⟩ + |111⟩)/√2 of Level 2 is made by H on qubit 0 and a chain of CX gates. Its ⟨X X X⟩ is exactly **1**: the state is unchanged when all three qubits are flipped. Noise brings the measured value below 1, and this capstone tries to get it back.

A quantum computer measures in the Z basis. To measure X on every qubit, add an H gate on every qubit before the measurement (H turns X into Z). Then each shot gives a key of n bits, and the product of the X results is +1 for an **even** number of 1s and −1 for an **odd** number. So ⟨X X X⟩ is the **parity**:

⟨X X X⟩ = Σ over results of (−1)^(number of 1s) · (fraction of shots with that result).

**Your task:**

- `ghz_circuit(n)`: a QuantumCircuit on n qubits: H on qubit 0, CX(0, 1), CX(1, 2), ..., CX(n − 2, n − 1), then H on every qubit. **No measurements**: Step 5 folds this circuit (runs it, its inverse and it again), and a measurement cannot be inverted. The prepared `measured(qc)` adds them (qubit i into classical bit i).
- `parity_expectation(probs)`: the parity above, for a dictionary of counts or of probabilities. Divide by the total, so both work."""),
code("""def ghz_circuit(n):
    \"\"\"H on qubit 0, the CX chain, then H on every qubit (to measure X); no measurements.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete ghz_circuit() first.")


def parity_expectation(probs):
    \"\"\"<X...X>: sum of (-1)^(number of 1s) * weight, divided by the total weight.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete parity_expectation() first.")"""),
code("""def measured(qc):
    \"\"\"Prepared: a copy of qc with every qubit measured (qubit i into classical bit i).\"\"\"
    out = QuantumCircuit(qc.num_qubits, qc.num_qubits)
    out.compose(qc, inplace=True)
    for i in range(qc.num_qubits):
        out.measure(i, i)
    return out


NQ_DEMO, SHOTS = 3, 4000
print(measured(ghz_circuit(NQ_DEMO)).draw())
ideal = exact_probabilities(measured(ghz_circuit(NQ_DEMO)), 0, 0)
print("Without noise:", {k: round(v, 3) for k, v in ideal.items()}, " <XXX> =", round(parity_expectation(ideal), 4), "  (expected 1.0)")

exact1 = exact_probabilities(measured(ghz_circuit(NQ_DEMO)))
counts1 = sample_counts(exact1, SHOTS, seed=1)
E_exact, E_run = parity_expectation(exact1), parity_expectation(counts1)
s = expectation_sigma(E_exact, SHOTS)
print(f"With noise: exact <XXX> = {E_exact:.4f};  your run of {SHOTS} shots: {E_run:.4f} +- {s:.4f} ({(E_run - E_exact) / s:+.1f} sigma)")"""),
md(f"""**What to notice.** Without noise every shot has an even number of 1s and ⟨X X X⟩ = 1. With the course noise model the exact value is **{NUM['raw'][1]:.4f}**: about {1 - NUM['raw'][1]:.0%} below the ideal, far more than the shot noise σ ≈ {NUM['sig_raw']:.4f} of 4,000 shots. Shot noise is random and shrinks with more shots; this **bias** comes from the noise and stays however many shots you take. Mitigation attacks the bias."""),

md("""## Step 4: readout mitigation

Part of the bias comes from reading out. In the capstone model, each bit is recorded wrongly with the confusion matrix C (rows: the true bit, columns: the recorded bit):

C = [[1 − r, r], [2r, 1 − 2r]] with r = READOUT.

For one bit, the measured probabilities are m[recorded] = Σ over true of C[true][recorded] · t[true], that is **m = Cᵀ t**, so the true probabilities are **t = (Cᵀ)⁻¹ m**. The errors of different bits are independent, so for n bits you apply the 2 × 2 inverse to each bit in turn (a tensor product of inverses, the method of Module 3). The prepared `apply_readout` does the forward direction (it applies Cᵀ bit by bit); your function undoes it.

Bit j of a key is the character n − 1 − j (qubit 0 is the rightmost). One way: put the probabilities in a vector v indexed by int(key, 2), and for each bit j reshape it to (2^(n−1−j), 2, 2^j) and apply the 2 × 2 matrix to the middle axis with `np.einsum("ij,ajb->aib", inverse, t)`, exactly as `apply_readout` does with C.

**Your task:** `mitigate_readout(probs, readout)`: a dictionary {key: corrected probability} with every key of n bits. Normalize the input first (it may be counts). The corrected values may be slightly negative when they come from shots: that is the statistical noise of an inverse, and the parity still works."""),
code("""def mitigate_readout(probs, readout):
    \"\"\"Probabilities corrected for readout errors: (C^T)^-1 applied to every bit.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete mitigate_readout() first.")"""),
code("""print("Readout-mitigated, exact:   <XXX> =", round(parity_expectation(mitigate_readout(exact1, READOUT_COURSE)), 4))
print("Readout-mitigated, your run: <XXX> =", round(parity_expectation(mitigate_readout(counts1, READOUT_COURSE)), 4))
check = mitigate_readout({k: float(v) for k, v in zip(["00", "01", "10", "11"], apply_readout([0.1, 0.2, 0.3, 0.4], 0.02))}, 0.02)
print("Undoing apply_readout:", {k: round(v, 6) for k, v in check.items()}, "  (expected 0.1, 0.2, 0.3, 0.4)")"""),
md(f"""**What to notice.** Readout mitigation moves the exact value from {NUM['raw'][1]:.4f} to **{NUM['ro'][1]:.4f}**: most of the bias came from reading out. What is left, about {1 - NUM['ro'][1]:.3f}, comes from the gates, and no readout correction can remove it. The price: the corrected value is less certain (Step 7 measures by how much), because the inverse amplifies the shot noise of every bit."""),

md("""## Step 5: amplify the noise by folding

**Zero-noise extrapolation** (ZNE, Module 3) measures the value at several noise levels and extrapolates back to zero noise. To raise the gate noise without changing what the circuit does, **global folding** runs the circuit U, then U⁻¹ U, (scale − 1)/2 times:

- scale 1: U;
- scale 3: U U⁻¹ U;
- scale 5: U U⁻¹ U U⁻¹ U.

U⁻¹ U does nothing ideally, but its gates make errors, so scale 3 has about three times the gate noise of scale 1. Folding the whole circuit (global) is the simplest choice; folding single gates (local) gives finer scales, as the spin-chain capstone does.

**Your task:** `fold_global(qc, scale)` for an odd scale: a **new** circuit (start from `qc.copy()`, do not change `qc`), with `qc.inverse()` and `qc` composed (scale − 1)/2 times. `out.compose(other, inplace=True)` appends a circuit."""),
code("""def fold_global(qc, scale):
    \"\"\"qc followed by (scale - 1) / 2 pairs of (qc inverse, qc); scale odd. qc itself is not changed.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete fold_global() first.")"""),
code("""SCALES = [1, 3, 5]
base = ghz_circuit(NQ_DEMO)
exact_s, raw_s, ro_s = {}, {}, {}
print(f"{'scale':>6s}{'gates':>7s}{'raw <XXX>':>11s}{'readout-mitigated':>19s}")
for s in SCALES:
    qc = fold_global(base, s)
    exact_s[s] = exact_probabilities(measured(qc))
    raw_s[s] = parity_expectation(exact_s[s])
    ro_s[s] = parity_expectation(mitigate_readout(exact_s[s], READOUT_COURSE))
    print(f"{s:6d}{len(qc.data):7d}{raw_s[s]:11.4f}{ro_s[s]:19.4f}")"""),
md(f"""**What to notice.** The readout-mitigated values fall with the scale ({NUM['ro'][1]:.4f}, {NUM['ro'][3]:.4f}, {NUM['ro'][5]:.4f}): folding added gate noise and nothing else, as intended. The raw values fall too, but they also carry the readout error, which folding does **not** amplify: the circuit is measured once at every scale."""),

md("""## Step 6: extrapolate to zero noise

Fit a polynomial of degree d to the points (scale, value) by least squares, and read it at scale 0. The answer is a **weighted sum** of the measured values, Ê = Σ cᵢ Eᵢ, and the weights depend only on the scales:

- write the fit as V a ≈ E, where V is the **Vandermonde** matrix with rows (1, sᵢ, sᵢ², ..., sᵢ^d);
- the least-squares coefficients are a = V⁺ E, with V⁺ the pseudo-inverse (`np.linalg.pinv`);
- the value at scale 0 is a₀, the first coefficient, so the weights are **the first row of V⁺**.

With three scales, degree 1 is a least-squares line and degree 2 passes through all three points (**Richardson extrapolation**). Knowing the weights is what lets you compute the uncertainty of the result with `combined_sigma`.

**Your task:** `zne_coefficients(scales, degree)`: the list of weights, one per scale. `np.vander(np.asarray(scales, dtype=float), degree + 1, increasing=True)` builds V."""),
code("""def zne_coefficients(scales, degree):
    \"\"\"Weights c_i with value(0) = sum c_i E_i for a least-squares polynomial of this degree.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete zne_coefficients() first.")"""),
code("""lin, quad = zne_coefficients(SCALES, 1), zne_coefficients(SCALES, 2)
print("linear weights:   ", [round(float(c), 4) for c in lin], "  (expected [1.0833, 0.3333, -0.4167])")
print("quadratic weights:", [round(float(c), 4) for c in quad], "  (expected [1.875, -1.25, 0.375])")
zne = lambda c, vals: sum(ci * vals[s] for ci, s in zip(c, SCALES))
print(f"\\nExact values (ideal 1):")
print(f"   raw, scale 1:                    {raw_s[1]:.4f}")
print(f"   readout-mitigated, scale 1:      {ro_s[1]:.4f}")
print(f"   ZNE linear, without readout fix: {zne(lin, raw_s):.4f}")
print(f"   readout + ZNE linear:            {zne(lin, ro_s):.4f}")
print(f"   readout + ZNE quadratic:         {zne(quad, ro_s):.4f}")"""),
md(f"""**What to notice.** Each weight list adds up to 1 (a value that does not change with the scale is returned unchanged). Combined with readout mitigation, the linear extrapolation gives **{NUM['lin']:.4f}** and the quadratic one {NUM['quad']:.4f}: almost all of the bias is gone. ZNE **without** the readout fix gives only **{NUM['lin_raw']:.4f}**, because folding does not amplify the readout error, so extrapolation cannot see it. The two methods remove different errors and work best together.

The weights also show the price. The quadratic weights are larger (1.875, −1.25, 0.375) than the linear ones, and a larger weight multiplies the shot noise of its measurement."""),

md("""## Step 7: how certain is each result, and is it worth the shots?

**The bootstrap.** For the readout-mitigated value, a formula for σ is messy. The **bootstrap** measures it from the data: pretend the measured fractions are the truth, draw many new runs of the same number of shots from them (`rng.multinomial(shots, fractions)`), apply the same estimator to each, and take the standard deviation of the results. It works for any estimator, however complicated.

**Your task:** `bootstrap_sigma(counts, estimator, n_boot=200, seed=0)`:

1. `rng = np.random.default_rng(seed)`; the keys, their counts, the total number of shots, the fractions;
2. n_boot times: draw new counts with `rng.multinomial(shots, fractions)`, turn them into a dictionary, apply `estimator`;
3. return the standard deviation of the n_boot values, `np.std(values, ddof=1)`.

Do not change `counts`; build a new dictionary for every resample.

The cell after it takes a run of 4,000 shots at each scale (12,000 shots in all), gives every result with its σ, and compares each with the ideal value 1."""),
code("""def bootstrap_sigma(counts, estimator, n_boot=200, seed=0):
    \"\"\"The standard deviation of estimator(resampled counts) over n_boot multinomial resamplings.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete bootstrap_sigma() first.")"""),
code("""ro_est = lambda c: parity_expectation(mitigate_readout(c, READOUT_COURSE))
runs = {s: sample_counts(exact_s[s], SHOTS, seed=s) for s in SCALES}
E_raw = {s: parity_expectation(runs[s]) for s in SCALES}
E_ro = {s: ro_est(runs[s]) for s in SCALES}
sig_raw = {s: bootstrap_sigma(runs[s], parity_expectation) for s in SCALES}
sig_ro = {s: bootstrap_sigma(runs[s], ro_est) for s in SCALES}
print("Raw, scale 1: bootstrap sigma", round(sig_raw[1], 4), " formula expectation_sigma:", round(expectation_sigma(E_raw[1], SHOTS), 4))

results = {
    "raw (4,000 shots)": (E_raw[1], sig_raw[1]),
    "readout-mitigated (4,000 shots)": (E_ro[1], sig_ro[1]),
    "readout + ZNE linear (12,000 shots)": (zne(lin, E_ro), combined_sigma(lin, [sig_ro[s] for s in SCALES])),
    "readout + ZNE quadratic (12,000 shots)": (zne(quad, E_ro), combined_sigma(quad, [sig_ro[s] for s in SCALES])),
}
print(f"\\n{'method':>40s}{'value':>9s}{'sigma':>9s}{'(1 - value) / sigma':>21s}")
for name, (v, sg) in results.items():
    print(f"{name:>40s}{v:9.4f}{sg:9.4f}{(1 - v) / sg:21.1f}")"""),
md(f"""**What to notice.** The bootstrap σ of the raw value agrees with the formula `expectation_sigma`, a good test of both. Each step of mitigation **removes bias and adds uncertainty**: with these runs the exact standard deviations are about {NUM['sig_raw']:.4f} (raw), {NUM['sig_ro'][1]:.4f} (readout-mitigated), {NUM['sig_lin']:.4f} (with linear ZNE) and {NUM['sig_quad']:.4f} (with quadratic ZNE). The raw value is many σ below 1: its bias is significant in any run. The exact readout-mitigated value is about {(1 - NUM['ro'][1]) / NUM['sig_ro'][1]:.1f}σ below 1 (bias {1 - NUM['ro'][1]:.3f}, σ {NUM['sig_ro'][1]:.3f}), so a single run of 4,000 shots may or may not show the remaining gate error: look at the last column of your run. Both ZNE results agree with 1 within their σ, the quadratic one with a much larger σ.

**The shot budget.** Which method is best depends on how many shots you can spend. A good measure is the **root-mean-square error**, RMS = √(bias² + σ²): the typical distance from the true value, counting both kinds of error. σ shrinks as 1/√shots; the bias does not. Run the cell: for a total budget of shots (split equally between the three scales for ZNE), it gives the RMS error of each method from the exact biases and σ."""),
code("""bias = {"raw": 1 - raw_s[1], "readout": 1 - ro_s[1], "readout + ZNE linear": 1 - zne(lin, ro_s),
        "readout + ZNE quadratic": 1 - zne(quad, ro_s)}
per_shot = {"raw": sig_raw[1] ** 2 * SHOTS, "readout": sig_ro[1] ** 2 * SHOTS,      # sigma^2 x shots does not depend on shots
            "readout + ZNE linear": combined_sigma(lin, [sig_ro[s] for s in SCALES]) ** 2 * 3 * SHOTS,
            "readout + ZNE quadratic": combined_sigma(quad, [sig_ro[s] for s in SCALES]) ** 2 * 3 * SHOTS}
budgets = [300, 1200, 3000, 12000, 48000, 300000, 3000000]
print(f"{'total shots':>12s}" + "".join(f"{m:>25s}" for m in bias))
for B in budgets:
    rms = {m: math.sqrt(bias[m] ** 2 + per_shot[m] / B) for m in bias}
    best = min(rms, key=rms.get)
    print(f"{B:12,d}" + "".join(f"{rms[m]:25.4f}" for m in bias) + "   best: " + best)"""),
md(f"""**What to notice.** With a few hundred shots, readout mitigation alone gives the smallest error: ZNE's larger σ costs more than the small bias it removes. From about **{r2(NUM['cross1']):,} shots** in all, linear ZNE wins, because its σ falls below the remaining bias of readout mitigation. Quadratic ZNE removes the last bit of bias, but its σ is about twice the linear one, so it pays only when the total budget reaches roughly **{r2(NUM['cross2']):,} shots**. (Your bootstrap σ values come from one run each, so your crossings may differ a little from these exact ones.) Mitigation is a trade: the more bias you remove, the more shots you need to keep the uncertainty below it."""),

md("""## Step 8: more qubits

A larger GHZ state has more gates and more bits to misread. Run the cell: for 2 to 5 qubits it gives the exact raw, readout-mitigated and fully mitigated values, and the σ of each from a run of 4,000 shots per scale."""),
code("""print(f"{'qubits':>7s}{'raw':>16s}{'readout':>18s}{'readout + ZNE lin':>21s}")
table = {}
for n in range(2, 6):
    ex = {s: exact_probabilities(measured(fold_global(ghz_circuit(n), s))) for s in SCALES}
    rn = {s: sample_counts(ex[s], SHOTS, seed=10 * n + s) for s in SCALES}
    sr = {s: bootstrap_sigma(rn[s], ro_est) for s in SCALES}
    raw_n = parity_expectation(ex[1])
    ro_n = {s: ro_est(ex[s]) for s in SCALES}
    table[n] = (raw_n, ro_n[1], zne(lin, ro_n))
    print(f"{n:7d}{raw_n:9.4f} +- {expectation_sigma(raw_n, SHOTS):.3f}{ro_n[1]:11.4f} +- {sr[1]:.3f}"
          f"{zne(lin, ro_n):14.4f} +- {combined_sigma(lin, [sr[s] for s in SCALES]):.3f}")"""),
md(f"""**What to notice.** The raw value falls quickly with the number of qubits ({S[2]['raw'][1]:.3f} for 2 qubits, {S[5]['raw'][1]:.3f} for 5), and so does the readout-mitigated one ({S[2]['ro'][1]:.3f} to {S[5]['ro'][1]:.3f}). Full mitigation stays close to 1 ({S[2]['zne_lin']:.4f} to {S[5]['zne_lin']:.4f}), but its σ grows with n: more bits to correct means a larger amplification of the shot noise. For a much larger circuit the mitigated value would need many more shots to be as certain, which is the main limit of these methods."""),

*personal_step(9, ["NQ", "P2", "READOUT"], "NQ (2 to 5 qubits), P2 and READOUT (0.005 to 0.030)",
               "NQ, P2, READOUT = 3, 0.01, 0.02 gives " + str(M.personal_value(3, 0.01, 0.02)), None, None,
               "10,000 times the gain of full mitigation for your noise model, (readout + linear ZNE at scales 1, 3, 5) minus "
               "the raw value, both exact, for an NQ-qubit GHZ state, rounded to a whole number."),
check_cell(hidden_check, "l3_m6_mit_check", "check_l3_m6_mit",
           "fraction_sigma, expectation_sigma, combined_sigma, ghz_circuit, parity_expectation,\n"
           "                                          mitigate_readout, fold_global, zne_coefficients, bootstrap_sigma,\n"
           "                                          NQ, P2, READOUT"),
*ending("""- **One expectation value, one state.** ⟨X X X⟩ of a GHZ state is a single number whose ideal value you know. In a real application the ideal value is unknown, so you cannot see the remaining bias; you can only check that methods agree within their σ.
- **The noise model is known exactly.** Readout mitigation used the exact READOUT. On a device the confusion matrix is itself measured with shots (calibration circuits) and drifts over time, which adds uncertainty the bootstrap here does not include.
- **Folding assumes the noise scales.** Global folding triples the gate noise only if U⁻¹ U makes the same errors as U. On hardware, the compiler may cancel U⁻¹ U, coherent errors can cancel or add, and the noise is not exactly depolarizing, so the extrapolation can be biased.
- **The extrapolation model is a guess.** A line or a parabola is a choice. The values here fall almost exactly as a smooth curve, which is why both work; real data may not.
- **Larger circuits.** σ grows quickly with the number of qubits and the depth. These methods suit small circuits and a modest number of measured values.""",
       "the **Capstone Quiz — Mitigation Study**"),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl3m6m{i:02d}"
out = ROOT / "content" / "level3" / NAME
nbf.write(nb, out)
print("wrote", out)
