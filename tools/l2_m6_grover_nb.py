"""Cells of the Module 6 project notebook "Grover's search with noise"."""
from l2_m6_text import md, code, common_intro, step0_browser, step1_statistics, step2_noise, limits_and_summary, check_source


def cells():
    c = [common_intro(
        "Grover's search with noise", 180,
        "Without noise, 2 Grover iterations find one marked string among 8 with probability 0.945. Each iteration "
        "adds two 3-qubit controlled-Z gates, 12 CNOTs in all, and each CNOT adds a little error. **On a noisy computer, "
        "is 2 still the best number of iterations, and how noisy can the computer be before 1 iteration wins?**",
        """1. Write three statistics functions (Step 1) and meet the course noise model (Step 2).
2. Write `grover_circuit(marked, iterations)`, your Module 2 search as one function (Step 3).
3. Run it once on the noisy simulator and compare the result with the noise model's expected value (Step 4).
4. Measure the success probability for 0 to 5 iterations, with uncertainties (Step 5), and test whether 2 iterations beat 1 (Step 6).
5. Find how noisy the computer can be before 1 iteration wins (Step 7).
6. Run your own computer's noise model and get your verification value (Step 8), then look at 2 qubits (Step 9) and at the limits.""",
        "Your project quiz gives you three numbers: MARKED (the string to search for, 0 to 7), and P2 and READOUT for your computer.")]
    c += step0_browser()
    c += step1_statistics(0.687, ", and the check cell tests them")
    c += step2_noise()
    c += [
        md("""## Step 3: your Grover circuit

**Your task:** write `grover_circuit(marked, iterations)`, as in Module 2: H on every qubit; then `iterations` rounds of

- the **oracle**: X on every qubit whose bit of `marked` is 0, the multi-controlled Z `mcz`, the same X gates;
- the **diffuser**: H, X, `mcz`, X, H on every qubit;

then measure qubit q into classical bit q. Qubit q is the bit `marked[n - 1 - q]` (qubit 0 is the rightmost bit), and `n = len(marked)` is 2 or 3.

The prepared `mcz` is `cz` for 2 qubits. For 3 qubits it is **not** the `h`, `mcx`, `h` of Module 2: a 3-qubit gate would get no error from the noise model, so `mcz` is written out with 6 CNOTs and 7 T or T† gates, as a real computer would run it. Use it once in each oracle and once in each diffuser.

The cell after it checks the ideal success probability for 101 with 2 iterations (no noise): 0.9453."""),
        code("""def mcz(qc, qubits):
    \"\"\"Prepared: a controlled Z on 2 or 3 qubits, written with one- and two-qubit gates only.\"\"\"
    qubits = list(qubits)
    if len(qubits) == 2:
        qc.cz(qubits[0], qubits[1])
        return
    a, b, c = qubits                       # the textbook CCZ: 6 CNOTs, T and T-dagger gates
    qc.cx(b, c); qc.tdg(c); qc.cx(a, c); qc.t(c); qc.cx(b, c); qc.tdg(c); qc.cx(a, c)
    qc.t(b); qc.t(c); qc.cx(a, b); qc.t(a); qc.tdg(b); qc.cx(a, b)


def grover_circuit(marked, iterations):
    \"\"\"Grover's search for one marked bit string, with measurements (your Module 2 code as one function).\"\"\"
    n = len(marked)
    qc = QuantumCircuit(n, n)
    # YOUR CODE HERE: H on every qubit; `iterations` rounds of the oracle and the diffuser; measure qubit q into bit q
    raise NotImplementedError("Complete grover_circuit() first.")
    return qc"""),
        code("""qc = grover_circuit("101", 2)
ideal = expected_probabilities(qc, p2=0, readout=0)
print("ideal P(101) with 2 iterations:", round(ideal.get("101", 0), 4), "  (expected 0.9453)")
print("gates:", qc.count_ops())"""),
        md("""**What to notice.** With 2 iterations the circuit has 4 `mcz` gates, so 24 `cx` gates, plus 16 `t` and 12 `tdg` gates and the H and X gates. `expected_probabilities(qc, p2=0, readout=0)` gives the ideal probabilities: a noise model with zero errors."""),

        md("""## Step 4: one run on the noisy simulator

Now use the course noise model (P2 = 0.01, READOUT = 0.02). The cell

1. computes the noise model's **expected** probability of 101, `p_expected`, exactly;
2. runs **4,000 shots** on `AerSimulator(noise_model=course_noise_model())`;
3. compares the two with your `within_3_sigma`, and plots the expected and measured fraction of every result.

`RUN_SEED` fixes the simulator's random numbers so that you can repeat a run. Change it and run the cell again: the measured count changes, the expected value does not. The project quiz asks for your measured fraction of 101 (any run you made with 4,000 shots) and for σ."""),
        code("""RUN_SEED = 1                     # any whole number; change it to see another run
noise = course_noise_model(0.01, 0.02)
sim = AerSimulator(noise_model=noise)

qc = grover_circuit("101", 2)
p_expected = expected_probabilities(qc, 0.01, 0.02)["101"]
counts = sim.run(qc, shots=4000, seed_simulator=RUN_SEED).result().get_counts()
k = counts.get("101", 0)
sigma = shot_sigma(p_expected, 4000)
print(f"ideal (no noise):               P(101) = {ideal['101']:.4f}")
print(f"noise model, expected:          P(101) = {p_expected:.4f}")
print(f"noise model, 4000 shots:        {k} found 101, a fraction {k / 4000:.4f}")
print(f"sigma = {sigma:.4f}, 3 sigma = {3 * sigma:.4f}; the measured fraction is {(k / 4000 - p_expected) / sigma:+.2f} sigma from expected")
print("within 3 sigma:", within_3_sigma(k, 4000, p_expected))

expected = expected_probabilities(qc, 0.01, 0.02)
labels = [format(i, "03b") for i in range(8)]
x = np.arange(8)
plt.figure(figsize=(7, 3.2))
plt.bar(x - 0.2, [expected.get(s, 0) for s in labels], 0.4, label="noise model, expected")
plt.bar(x + 0.2, [counts.get(s, 0) / 4000 for s in labels], 0.4, label="4000 shots")
plt.xticks(x, labels); plt.ylabel("fraction of shots"); plt.legend(); plt.tight_layout(); plt.show()"""),
        md("""**What to notice.** The noise takes P(101) from 0.945 down to 0.687. The lost probability is spread over the seven wrong strings: depolarizing errors push the state toward a completely random one, where every string has probability 1/8. Your measured fraction should lie within about 0.022 (3σ) of 0.687; compare it with the ideal 0.945 too, which is 35σ away: a noisy run cannot be judged against the ideal value."""),

        md("""## Step 5: success against the number of iterations

The cell computes, for 0 to 5 iterations, the ideal probability of 101, the noise model's expected probability, and a measured fraction from 4,000 shots with its σ. The plot shows the measured fractions as points with **error bars of ±3σ**. It takes a few seconds."""),
        code("""iterations = list(range(6))
ideal_p, noisy_p, measured, sig = [], [], [], []
for it in iterations:
    qc = grover_circuit("101", it)
    ideal_p.append(expected_probabilities(qc, 0, 0)["101"])
    noisy_p.append(expected_probabilities(qc, 0.01, 0.02)["101"])
    kk = sim.run(qc, shots=4000, seed_simulator=RUN_SEED + 10 + it).result().get_counts().get("101", 0)
    measured.append(kk / 4000)
    sig.append(shot_sigma(noisy_p[-1], 4000))

print("iterations   ideal    expected (noise)   measured    sigma    cx gates")
for it in iterations:
    print(f"    {it}       {ideal_p[it]:.4f}       {noisy_p[it]:.4f}         {measured[it]:.4f}    {sig[it]:.4f}      {12 * it}")

plt.figure(figsize=(7, 3.5))
plt.plot(iterations, ideal_p, "k--o", label="ideal")
plt.plot(iterations, noisy_p, "C0-s", label="noise model, expected")
plt.errorbar(iterations, measured, yerr=[3 * s for s in sig], fmt="C3o", capsize=4, label="4000 shots, ±3σ")
plt.axhline(1 / 8, color="gray", lw=1, ls=":", label="random guess (1/8)")
plt.xlabel("iterations"); plt.ylabel("P(101)"); plt.legend(fontsize=8); plt.tight_layout(); plt.show()"""),
        md("""**What to notice.**

- Noise lowers the peak (0.945 to 0.687 at 2 iterations) and every iteration costs more: each adds 12 `cx` gates.
- Where the ideal probability is **below** 1/8 (4 iterations: 0.012), the noise **raises** it (to 0.070). Noise does not simply subtract: it mixes the state toward the random one, which pulls every probability toward 1/8.
- 2 iterations still give the highest expected value under the course noise model. Is the measured difference from 1 iteration significant? That is Step 6."""),

        md("""## Step 6: is 2 iterations better than 1?

Use `difference_sigma` to test the difference between the measured fractions of 1 and 2 iterations from Step 5. The difference is **significant** if it is larger than 3σ_diff."""),
        code("""f1, f2 = measured[1], measured[2]
d = f2 - f1
sd = difference_sigma(f1, f2, 4000)
print(f"1 iteration: {f1:.4f}   2 iterations: {f2:.4f}   difference {d:+.4f}")
print(f"sigma_diff = {sd:.4f}; the difference is {d / sd:.1f} sigma_diff")
print("significant (more than 3 sigma_diff):", abs(d) > 3 * sd)
print(f"expected difference from the noise model: {noisy_p[2] - noisy_p[1]:+.4f}")"""),
        md("""**What to notice.** With the course noise model the expected difference is 0.053, about 5σ_diff for 4,000 shots each, so your measured difference will almost always be significant: under this noise, 2 iterations are better. The test uses only the two measured fractions; it would work the same way on a real computer, where no one knows the expected values."""),

        md("""## Step 7: how noisy can the computer be?

As P2 grows, the 24 `cx` gates of 2 iterations lose more than the 12 of 1 iteration. The cell scans P2 from 0 to 0.030 in steps of 0.001 (READOUT stays 0.02) and computes the noise model's expected success for 1 and 2 iterations; no shots are needed. It prints the **crossover**: the smallest P2 in the scan at which 1 iteration is better. The project quiz asks for it.

Then it asks how many shots would show the difference at P2 = 0.015. For a difference d between two fractions near p₁ and p₂, the condition d > 3σ_diff with S shots each gives **S > 9 (p₁(1 − p₁) + p₂(1 − p₂)) / d²**."""),
        code("""p2_values = np.round(np.arange(0, 0.0305, 0.001), 3)
one = [expected_probabilities(grover_circuit("101", 1), p, 0.02)["101"] for p in p2_values]
two = [expected_probabilities(grover_circuit("101", 2), p, 0.02)["101"] for p in p2_values]
crossover = next(p for p, a, b in zip(p2_values, one, two) if a > b)
print(f"crossover: from P2 = {crossover:.3f}, 1 iteration gives a higher expected success than 2")

i = list(p2_values).index(0.015)
a, b = one[i], two[i]
shots_needed = 9 * (a * (1 - a) + b * (1 - b)) / (b - a) ** 2
print(f"at P2 = 0.015: 1 iteration {a:.4f}, 2 iterations {b:.4f}, difference {b - a:+.4f}")
print(f"  with 4000 shots each, sigma_diff = {difference_sigma(a, b, 4000):.4f}; shots needed for 3 sigma_diff: about {shots_needed:,.0f} each")

plt.figure(figsize=(7, 3.3))
plt.plot(p2_values, one, label="1 iteration"); plt.plot(p2_values, two, label="2 iterations")
plt.axvline(crossover, color="gray", ls=":", lw=1)
plt.xlabel("P2 (two-qubit depolarizing error)"); plt.ylabel("expected P(101)"); plt.legend(); plt.tight_layout(); plt.show()"""),
        md("""**What to notice.** The two curves cross near P2 = 0.018 (an average two-qubit gate error of about 0.013): on a computer noisier than that, the shorter circuit wins although it is worse in theory. Near the crossover the two are almost equal, and telling them apart takes many shots: about 14,000 each at P2 = 0.015, where the difference is 0.017. Close to a crossover, a "which is better" question needs far more data than one far from it."""),

        md("""## Step 8: your computer, and your verification value

Open your **project quiz for this experiment** in Canvas. Question 1 shows your MARKED (0 to 7: the 3-bit string to search for, for example 5 is 101), your P2 and your READOUT. Type them below and run the cells.

The first cell runs 1 and 2 iterations on **your** noise model, 4,000 shots each, and keeps the 2-iteration counts as `my_counts`. The check cell then tests your three statistics functions and `grover_circuit()`, and checks that `my_counts` agree with your noise model by the 3σ rule. Only if everything passes does it print your **verification value**: 10,000 times the noise model's exact probability that 2 iterations find MARKED on your computer, rounded to a whole number."""),
        code("""MARKED = -1      # your number from Canvas, for example 5 (the string 101)
P2 = 0.0         # your number from Canvas, for example 0.017
READOUT = 0.0    # your number from Canvas, for example 0.015
my_counts = {}   # filled by the next cell"""),
        code("""my_marked = format(int(MARKED), "03b")
my_sim = AerSimulator(noise_model=course_noise_model(P2, READOUT))
my_runs = {}
for it in (1, 2):
    qc = grover_circuit(my_marked, it)
    p = expected_probabilities(qc, P2, READOUT)[my_marked]
    cts = my_sim.run(qc, shots=4000, seed_simulator=RUN_SEED + 100 + it).result().get_counts()
    my_runs[it] = cts
    kk = cts.get(my_marked, 0)
    print(f"{it} iteration(s): expected {p:.4f}, measured {kk / 4000:.4f} ± {shot_sigma(p, 4000):.4f} (1 sigma)")
my_counts = my_runs[2]
f1, f2 = my_runs[1].get(my_marked, 0) / 4000, my_runs[2].get(my_marked, 0) / 4000
print(f"difference (2 minus 1): {f2 - f1:+.4f} = {(f2 - f1) / difference_sigma(f1, f2, 4000):+.1f} sigma_diff")"""),
        code(check_source("l2_m6_grover_check.py", '''passed, messages, value = check_l2_m6_grover(grover_circuit, shot_sigma, within_3_sigma, difference_sigma,
                                             MARKED, P2, READOUT, my_counts)
for m in messages:
    print(m)
if passed:
    print("\\nAll tests passed. Your verification value is", value)
else:
    print("\\nNot yet: fix the item above and run this cell again.")''')),

        md("""## Step 9: two qubits

With 2 qubits, one iteration finds the marked string with probability exactly 1 in theory, and the circuit has only 2 two-qubit gates (the two `cz`). The cell compares the four possible marked strings under the course noise model."""),
        code("""for w in ("00", "01", "10", "11"):
    qc = grover_circuit(w, 1)
    p = expected_probabilities(qc, 0.01, 0.02)[w]
    p_gates = expected_probabilities(qc, 0.01, 0.0)[w]
    print(f"marked {w}: expected {p:.4f}   (gate errors only, no readout error: {p_gates:.4f})")"""),
        md("""**What to notice.** With only 2 two-qubit gates, readout errors dominate: without them the success would be about 0.98. The marked string 11 does worst and 00 best, because a 1 is misread twice as often as a 0 in this model. Real readout errors are asymmetric in the same way, which is why IBM's software offers **readout error mitigation**: it measures the confusion matrix and corrects the counts for it."""),
    ]
    c += limits_and_summary(
        """- **A model, not a computer.** The course model has the same depolarizing error on every gate and qubit, and no errors that depend on time. A real computer has a different error on each qubit pair (Module 4), drifts between calibrations, and has errors that are not random at all (a gate that over-rotates by the same angle every time).
- **No connectivity.** Here any two qubits can share a `cx`. On a real chip, qubits not connected need swaps, which add more two-qubit gates (Module 4); the crossover would come at a lower P2.
- **One marked string, 3 qubits.** With more qubits the best number of iterations grows (about (π/4)√N), the circuit grows with it, and the noise matters even more.
- **Shot noise.** Each measured fraction is uncertain by σ; differences smaller than 3σ_diff are not shown by the data, even when the model says they are there.""",
        "the Module 6 project quiz for this experiment, Grover's search with noise,")
    return c
