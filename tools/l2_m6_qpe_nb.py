"""Cells of the Module 6 project notebook "Phase estimation of 1/3 with noise"."""
from l2_m6_text import md, code, common_intro, step0_browser, step1_statistics, step2_noise, limits_and_summary, check_source


def cells():
    c = [common_intro(
        "phase estimation of 1/3 with noise", 180,
        "Phase estimation reads a phase as an m-bit binary fraction, so more counting qubits give a finer answer. "
        "But the circuit grows with m: m controlled phases, an inverse QFT with m(m − 1)/2 more, and m readouts. "
        "**On a noisy computer, how many counting qubits give the most accurate estimate of 1/3, and how does the answer "
        "depend on the noise?**",
        """1. Write three statistics functions (Step 1) and meet the course noise model (Step 2).
2. Write `qpe_circuit(m)` and `estimate_error(counts, m)` (Step 3).
3. Run it once on the noisy simulator and compare with the noise model's expected values (Step 4).
4. Measure the accuracy for 3 to 7 counting qubits, with uncertainties (Step 5), and test which differences are significant (Step 6).
5. Find the best number of counting qubits for different noise levels (Step 7).
6. Run your own computer's noise model and get your verification value (Step 8), then look at the limits.""",
        "Your project quiz gives you three numbers: COUNT (the number of counting qubits, 3 to 6), and P2 and READOUT for your computer.")]
    c += step0_browser()
    c += step1_statistics(0.559, ", and the check cell tests them")
    c += step2_noise()
    c += [
        md("""## Step 3: the phase estimation circuit

The phase gate P(2π/3) has the eigenvector |1⟩ with eigenvalue e^(2πi·θ), θ = 1/3. Phase estimation with m counting qubits (qubits 0 to m − 1) and the target (qubit m) works as in Module 1:

1. X on the target, so that it holds the eigenvector |1⟩;
2. H on every counting qubit;
3. for each counting qubit k, a controlled phase `cp(2 * pi * theta * 2**k, k, m)`: the phase gate applied 2ᵏ times, written as one gate with 2ᵏ times the angle;
4. the inverse QFT on the counting qubits (prepared, as in Module 1, with `h`, `cp` and `swap`);
5. measure counting qubit k into classical bit k. The target is not measured.

The result y, read as a binary number, gives the estimate θ ≈ y / 2ᵐ.

**Your task (a):** write `qpe_circuit(m)` for θ = 1/3. Use one `cp` per counting qubit: repeating the gate 2ᵏ times gives the same ideal result but many more gates, and so more noise.

**Your task (b):** write `estimate_error(counts, m)`: the **mean error** of the estimates over all shots, Σ count · error(y) / shots, where error(y) is the distance between y / 2ᵐ and 1/3 **around the circle**: a phase is only defined up to a whole turn, so with d = |y / 2ᵐ − 1/3|, the error is min(d, 1 − d). Read y with `int(key, 2)`."""),
        code("""THETA = 1 / 3


def inverse_qft(qc, qubits):
    \"\"\"Prepared: the inverse quantum Fourier transform on the given qubits (with the swaps), as in Module 1.\"\"\"
    qubits = list(qubits)
    n = len(qubits)
    for j in range(n // 2):
        qc.swap(qubits[j], qubits[n - 1 - j])
    for j in range(n):
        for k in range(j):
            qc.cp(-np.pi / 2 ** (j - k), qubits[k], qubits[j])
        qc.h(qubits[j])


def qpe_circuit(m):
    \"\"\"Phase estimation of THETA with m counting qubits (0 to m - 1) and the target qubit m.\"\"\"
    qc = QuantumCircuit(m + 1, m)
    # YOUR CODE HERE
    raise NotImplementedError("Complete qpe_circuit() first.")
    return qc


def estimate_error(counts, m):
    \"\"\"The mean distance, around the circle, between y / 2**m and THETA over all shots.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete estimate_error() first.")"""),
        code("""def as_counts(probs, shots=10 ** 6):
    \"\"\"Prepared: probabilities scaled to look like counts, so that estimate_error() can use them too.\"\"\"
    return {k: v * shots for k, v in probs.items()}


def best_estimate(m):
    \"\"\"Prepared: the m-bit result y whose y / 2**m is closest to THETA.\"\"\"
    return min(range(2 ** m), key=lambda y: min(abs(y / 2 ** m - THETA), 1 - abs(y / 2 ** m - THETA)))


qc = qpe_circuit(4)
ideal = expected_probabilities(qc, 0, 0)
best = format(best_estimate(4), "04b")
print("ideal probabilities with 4 counting qubits (largest first):")
for y, p in sorted(ideal.items(), key=lambda kv: -kv[1])[:5]:
    print(f"  {y} = {int(y, 2):2d}/16 = {int(y, 2) / 16:.4f}   probability {p:.4f}")
print("best estimate:", best, "=", int(best, 2) / 16, "  ideal P(best) =", round(ideal[best], 4), "  (expected 0.6849)")
print("ideal mean error:", round(estimate_error(as_counts(ideal), 4), 4), "  (expected 0.0461)")
print("gates:", qc.count_ops())"""),
        md("""**What to notice.** 1/3 = 0.010101... in binary never ends, so no m-bit fraction equals it. With 4 counting qubits the closest is 0101 = 5/16 = 0.3125, which comes out with probability 0.685; the rest spreads over nearby values (6/16 with 0.17, 4/16 with 0.04, 7/16 with 0.03, ...). This is not noise: it is how phase estimation behaves for a phase that is not an m-bit fraction, and the ideal probability of the best estimate stays near 0.68 for every m. The mean error still falls as m grows, because the estimates get finer."""),

        md("""## Step 4: one run on the noisy simulator

With the course noise model (P2 = 0.01, READOUT = 0.02) and 4 counting qubits, the cell computes the noise model's expected probability of the best estimate 0101 and its expected mean error, runs 4,000 shots, and compares.

The mean error is an average over shots of a quantity that varies from shot to shot, so its uncertainty is the **standard error of the mean**: SEM = s / √shots, where s is the standard deviation of the error over the shots. The prepared `error_sem(counts, m)` computes it.

`RUN_SEED` fixes the simulator's random numbers. The project quiz asks for your measured fraction of 0101 (any run with 4,000 shots) and for the expected mean error."""),
        code("""def error_sem(counts, m):
    \"\"\"Prepared: the standard error of the mean error: (standard deviation of the per-shot error) / sqrt(shots).\"\"\"
    errs = np.array([min(abs(int(y, 2) / 2 ** m - THETA), 1 - abs(int(y, 2) / 2 ** m - THETA)) for y in counts])
    n = np.array([counts[y] for y in counts], dtype=float)
    mean = np.sum(n * errs) / n.sum()
    return float(np.sqrt(np.sum(n * (errs - mean) ** 2) / (n.sum() - 1) / n.sum()))


RUN_SEED = 1                     # any whole number; change it to see another run
sim = AerSimulator(noise_model=course_noise_model(0.01, 0.02))

qc = qpe_circuit(4)
expected = expected_probabilities(qc, 0.01, 0.02)
p_expected = expected[best]
err_expected = estimate_error(as_counts(expected), 4)
counts = sim.run(qc, shots=4000, seed_simulator=RUN_SEED).result().get_counts()
k = counts.get(best, 0)
print(f"P(best = {best}): ideal {ideal[best]:.4f}, noise model expected {p_expected:.4f}, measured {k / 4000:.4f}")
print(f"  sigma = {shot_sigma(p_expected, 4000):.4f}; within 3 sigma: {within_3_sigma(k, 4000, p_expected)}")
print(f"mean error: ideal 0.0461, noise model expected {err_expected:.4f}, measured {estimate_error(counts, 4):.4f} ± {error_sem(counts, 4):.4f} (SEM)")

labels = [format(i, "04b") for i in range(16)]
x = np.arange(16)
plt.figure(figsize=(8, 3.2))
plt.bar(x - 0.2, [expected.get(s, 0) for s in labels], 0.4, label="noise model, expected")
plt.bar(x + 0.2, [counts.get(s, 0) / 4000 for s in labels], 0.4, label="4000 shots")
plt.xticks(x, labels, rotation=90); plt.ylabel("fraction of shots"); plt.legend(); plt.tight_layout(); plt.show()"""),
        md("""**What to notice.** The noise lowers P(0101) from 0.685 to 0.559 and raises the mean error from 0.046 to 0.079: the probability it takes away spreads over all 16 results, including ones far from 1/3. A completely random result would have a mean error of 0.25 (the average distance around the circle). Your measured fraction should be within about 0.024 (3σ) of 0.559."""),

        md("""## Step 5: accuracy against the number of counting qubits

The cell builds the circuit for m = 3 to 7 counting qubits and, for each, computes the ideal and noise-model mean errors and measures 4,000 shots. The plot shows the measured mean error with error bars of ±3 SEM. It takes up to a minute: with 7 counting qubits the simulator follows an 8-qubit density matrix."""),
        code("""ms = [3, 4, 5, 6, 7]
rows = []
for m in ms:
    qc = qpe_circuit(m)
    ideal_err = estimate_error(as_counts(expected_probabilities(qc, 0, 0)), m)
    exp_err = estimate_error(as_counts(expected_probabilities(qc, 0.01, 0.02)), m)
    cts = sim.run(qc, shots=4000, seed_simulator=RUN_SEED + 10 + m).result().get_counts()
    two_q = sum(v for g, v in qc.count_ops().items() if g in TWO_QUBIT_GATES)
    rows.append((m, two_q, ideal_err, exp_err, estimate_error(cts, m), error_sem(cts, m)))

print(" m   two-qubit gates   ideal error   expected (noise)   measured ± SEM")
for m, two_q, ie, ee, me, se in rows:
    print(f" {m}        {two_q:2d}           {ie:.4f}         {ee:.4f}          {me:.4f} ± {se:.4f}")
measured_err = {m: (me, se) for m, _, _, _, me, se in rows}

plt.figure(figsize=(7, 3.5))
plt.plot(ms, [r[2] for r in rows], "k--o", label="ideal")
plt.plot(ms, [r[3] for r in rows], "C0-s", label="noise model, expected")
plt.errorbar(ms, [r[4] for r in rows], yerr=[3 * r[5] for r in rows], fmt="C3o", capsize=4, label="4000 shots, ±3 SEM")
plt.xlabel("counting qubits m"); plt.ylabel("mean error of the estimate"); plt.legend(fontsize=8); plt.tight_layout(); plt.show()"""),
        md("""**What to notice.** Without noise the mean error roughly halves with each counting qubit (0.079, 0.046, 0.026, 0.015, 0.008). With noise it falls much more slowly (0.101, 0.079, 0.067, 0.063, 0.062): the finer resolution is paid for with more gates (7 two-qubit gates for m = 3, 31 for m = 7) and more readouts, and their errors add a background of random results that does not shrink. Under the course noise model, going from 6 to 7 counting qubits gains almost nothing."""),

        md("""## Step 6: which differences are significant?

For two mean errors with standard errors SEM₁ and SEM₂, the difference has σ_diff = √(SEM₁² + SEM₂²): the same rule as `difference_sigma`, which does this for fractions. The cell tests two pairs from Step 5: m = 4 against 5, and m = 6 against 7."""),
        code("""for a, b in ((4, 5), (6, 7)):
    (ea, sa), (eb, sb) = measured_err[a], measured_err[b]
    d, sd = eb - ea, math.sqrt(sa ** 2 + sb ** 2)
    print(f"m = {a}: {ea:.4f}   m = {b}: {eb:.4f}   difference {d:+.4f}, sigma_diff {sd:.4f} = {d / sd:+.1f} sigma_diff,",
          "significant" if abs(d) > 3 * sd else "not significant")"""),
        md("""**What to notice.** Going from 4 to 5 counting qubits lowers the expected mean error by 0.013, about 5 σ_diff: significant. From 6 to 7 the expected gain is 0.0008, well below σ_diff (about 0.003): 4,000 shots cannot say whether 7 counting qubits are better than 6 under this noise, and with a different RUN_SEED the measured order of the two can flip."""),

        md("""## Step 7: the best number of counting qubits depends on the noise

The cell computes the noise model's expected mean error for m = 3 to 7 at six noise levels (READOUT = 0.02 throughout; no shots) and prints the best m for each. The project quiz asks for the best m at P2 = 0.020. It takes up to a minute."""),
        code("""p2_levels = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]
best_m = {}
print("P2      " + "   ".join(f"m={m}  " for m in ms) + "  best m")
for p in p2_levels:
    errs = {m: estimate_error(as_counts(expected_probabilities(qpe_circuit(m), p, 0.02)), m) for m in ms}
    best_m[p] = min(errs, key=errs.get)
    print(f"{p:.3f}   " + "   ".join(f"{errs[m]:.4f}" for m in ms) + f"    {best_m[p]}")"""),
        md("""**What to notice.** At P2 = 0.005 the error still falls up to m = 7; at P2 = 0.015 the best is 6, and from P2 = 0.020 on it is 5. The noisier the computer, the fewer counting qubits are worth using: past the best m, each extra qubit adds more noise than precision. The best m also depends on READOUT, since every counting qubit is read out."""),

        md("""## Step 8: your computer, and your verification value

Open your **project quiz for this experiment** in Canvas. Question 1 shows your COUNT (3 to 6 counting qubits), your P2 and your READOUT. Type them below and run the cells.

The first cell runs your circuit on **your** noise model with 4,000 shots and keeps the counts as `my_counts`. The check cell tests your three statistics functions, `qpe_circuit()` and `estimate_error()`, and checks that `my_counts` agree with your noise model by the 3σ rule. Only if everything passes does it print your **verification value**: 10,000 times the noise model's exact probability of the best estimate with COUNT counting qubits on your computer, rounded to a whole number."""),
        code("""COUNT = 0        # your number from Canvas, for example 5
P2 = 0.0         # your number from Canvas, for example 0.017
READOUT = 0.0    # your number from Canvas, for example 0.015"""),
        code("""qc = qpe_circuit(int(COUNT))
my_expected = expected_probabilities(qc, P2, READOUT)
my_best = format(best_estimate(int(COUNT)), f"0{int(COUNT)}b")
my_counts = AerSimulator(noise_model=course_noise_model(P2, READOUT)).run(qc, shots=4000, seed_simulator=RUN_SEED + 100).result().get_counts()
kk = my_counts.get(my_best, 0)
print(f"best estimate {my_best}: expected {my_expected[my_best]:.4f}, measured {kk / 4000:.4f} ± {shot_sigma(my_expected[my_best], 4000):.4f} (1 sigma)")
print(f"mean error: expected {estimate_error(as_counts(my_expected), int(COUNT)):.4f}, "
      f"measured {estimate_error(my_counts, int(COUNT)):.4f} ± {error_sem(my_counts, int(COUNT)):.4f} (SEM)")"""),
        code(check_source("l2_m6_qpe_check.py", '''passed, messages, value = check_l2_m6_qpe(qpe_circuit, estimate_error, shot_sigma, within_3_sigma, difference_sigma,
                                          COUNT, P2, READOUT, my_counts)
for m in messages:
    print(m)
if passed:
    print("\\nAll tests passed. Your verification value is", value)
else:
    print("\\nNot yet: fix the item above and run this cell again.")''')),
    ]
    c += limits_and_summary(
        """- **A model, not a computer.** The course model has the same depolarizing error on every gate and qubit. A real computer has a different error on each qubit pair, drifts between calibrations, and has errors that repeat the same way every time (a phase that is slightly off), which add up instead of averaging out.
- **No connectivity.** The inverse QFT has a `cp` between every pair of counting qubits. On a real chip most pairs are not connected, so the transpiler adds swaps (Module 4), and the best m would be smaller still.
- **One phase.** 1/3 is a hard case: it is never an m-bit fraction. A phase such as 1/4 has an exact 2-bit answer, and its ideal probability is 1.
- **Shot noise.** Each mean error is uncertain by its SEM; the data cannot rank two values of m whose difference is below 3σ_diff, even when the model can.""",
        "the Module 6 project quiz for this experiment, phase estimation with noise,")
    return c
