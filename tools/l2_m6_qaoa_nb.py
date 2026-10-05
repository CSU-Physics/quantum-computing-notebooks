"""Cells of the Module 6 project notebook "QAOA trained on a simulator, evaluated with noise"."""
from l2_m6_text import md, code, common_intro, step0_browser, step1_statistics, step2_noise, limits_and_summary, check_source


def cells():
    c = [common_intro(
        "QAOA trained on a simulator, evaluated with noise", 180,
        "In Module 5, a second QAOA layer gave much better cuts. Each layer adds one `rzz` gate per edge, and on a noisy "
        "computer every gate costs a little. A common way to use QAOA is to find good angles on a simulator without "
        "noise, then run the circuit on the computer. **When angles trained without noise are run with noise, how much "
        "of each layer's gain survives, and how noisy can the computer be before a third layer stops helping?**",
        """1. Write three statistics functions (Step 1) and meet the course noise model (Step 2).
2. Write `qaoa_circuit(gammas, betas, edges, n)` (your Module 5 code) and `mean_cut(counts, edges)` for a 5-node graph (Step 3).
3. Train 1, 2 and 3 layers on the ideal simulator (Step 4).
4. Run the trained circuit with noise and compare with the noise model's expected values (Step 5).
5. Compare 1, 2 and 3 layers with noise, with uncertainties (Step 6), and find how much noise a third layer can take (Step 7).
6. Run your own computer's noise model and get your verification value (Step 8), then look at the limits.""",
        "Your project quiz gives you three numbers: LAYERS (1, 2 or 3), and P2 and READOUT for your computer.")]
    c += step0_browser(extra="\nfrom scipy.optimize import minimize")
    c += step1_statistics(0.587, ", and the check cell tests them")
    c += step2_noise()
    c += [
        md("""## Step 3: the project graph, the circuit and the measured cut

The project graph has 5 nodes: a ring 0-1-2-3-4-0 and one chord, (0, 2). It has 6 edges. The cell finds the maximum cut by brute force over all 2⁵ = 32 splits: bit q of a split (counted from the right) is node q's group.

**Your task (a):** write `qaoa_circuit(gammas, betas, edges, n)` as in Module 5: H on every qubit; then, for each layer k, `rzz(2 * gammas[k], i, j)` on every edge and `rx(2 * betas[k], q)` on every qubit. No measurements: the training uses the state itself, and the prepared `measured(qc)` adds the measurements for the runs with shots.

**Your task (b):** write `mean_cut(counts, edges)`: the average cut size over all shots. For each result key, node q's group is `key[-1 - q]` (qubit 0 is the rightmost character); an edge (i, j) is cut when the two groups differ. Weight each key by its count and divide by the total number of shots."""),
        code("""EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]
N = 5
CUTS = np.array([sum(((s >> i) & 1) != ((s >> j) & 1) for i, j in EDGES) for s in range(2 ** N)])   # cut of each split
MAX_CUT = CUTS.max()
print("maximum cut:", MAX_CUT, "for", [format(s, "05b") for s in range(2 ** N) if CUTS[s] == MAX_CUT])
print("average cut of a random split:", CUTS.mean())


def qaoa_circuit(gammas, betas, edges, n):
    \"\"\"QAOA: H on every qubit, then for each layer rzz(2*gamma) on every edge and rx(2*beta) on every qubit.\"\"\"
    qc = QuantumCircuit(n)
    # YOUR CODE HERE
    raise NotImplementedError("Complete qaoa_circuit() first.")
    return qc


def mean_cut(counts, edges):
    \"\"\"The average cut size over all shots in counts (node q is the bit key[-1 - q]).\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete mean_cut() first.")


def measured(qc):
    \"\"\"Prepared: a copy of qc that measures every qubit (qubit q into classical bit q).\"\"\"
    out = QuantumCircuit(qc.num_qubits, qc.num_qubits)
    out.compose(qc, inplace=True)
    out.measure(list(range(qc.num_qubits)), list(range(qc.num_qubits)))
    return out"""),
        code("""print("mean_cut({'01001': 3, '00000': 1}, EDGES) =", mean_cut({"01001": 3, "00000": 1}, EDGES), "  (expected 3.75)")
qc = qaoa_circuit([0.4], [0.3], EDGES, N)
print("gates in one layer:", qc.count_ops())"""),
        md("""**What to notice.** The maximum cut is 5 of the 6 edges: the chord makes a triangle (0, 1, 2), and a triangle can never have all three edges cut. Four splits reach it. A random split cuts 3 edges on average, so any useful QAOA result must beat 3. One layer has 6 `rzz` gates (one per edge) and 5 `rx` gates."""),

        md("""## Step 4: train on the ideal simulator

The cell trains the angles without noise, as in Module 5, with the **expected cut** computed from the state vector:

- **1 layer:** a grid scan of 33 × 17 angles (γ from 0 to π, β from 0 to π/2), then COBYLA from the best grid point;
- **2 layers:** COBYLA started from the 1-layer angles, repeated (γ₁, γ₁, β₁, β₁): a **warm start**;
- **3 layers:** COBYLA started from the 2-layer angles, with the second layer repeated.

It takes about half a minute. COBYLA's path depends on the SciPy version, so your angles can differ a little from the course's **reference angles**, printed below; the expected cuts agree to about 3 decimals. From Step 5 on, everyone uses the reference angles, so that the numbers in the project quiz are the same for all."""),
        code("""def ideal_cut(x):
    \"\"\"The expected cut without noise, for x = [gamma_1, ..., gamma_p, beta_1, ..., beta_p].\"\"\"
    p = len(x) // 2
    probs = Statevector(qaoa_circuit(list(x[:p]), list(x[p:]), EDGES, N)).probabilities()
    return float(probs @ CUTS)


gammas_grid, betas_grid = np.linspace(0, np.pi, 33), np.linspace(0, np.pi / 2, 17)
grid = np.array([[ideal_cut([g, b]) for g in gammas_grid] for b in betas_grid])
k = np.unravel_index(np.argmax(np.round(grid, 9)), grid.shape)
r1 = minimize(lambda x: -ideal_cut(x), [gammas_grid[k[1]], betas_grid[k[0]]], method="COBYLA", options={"maxiter": 300})
r2 = minimize(lambda x: -ideal_cut(x), [r1.x[0], r1.x[0], r1.x[1], r1.x[1]], method="COBYLA", options={"maxiter": 800})
r3 = minimize(lambda x: -ideal_cut(x), [r2.x[0], r2.x[1], r2.x[1], r2.x[2], r2.x[3], r2.x[3]], method="COBYLA",
              options={"maxiter": 1500})

REFERENCE_ANGLES = {1: ([2.817], [0.354]),
                    2: ([2.849, 2.627], [0.517, 0.309]),
                    3: ([2.905, 2.669, 2.619], [0.535, 0.401, 0.204])}
for p, r in ((1, r1), (2, r2), (3, r3)):
    g, b = REFERENCE_ANGLES[p]
    print(f"{p} layer(s): your training {-r.fun:.4f} at angles {np.round(r.x, 3)} | reference angles {ideal_cut(g + b):.4f}")"""),
        md("""**What to notice.** Without noise each layer helps: the expected cut rises from 4.110 (1 layer) to 4.623 (2) and 4.826 (3), out of a maximum of 5. The warm start matters: COBYLA started for 2 layers from all angles 0.5 stops at about 4.16, on a lower hill of the landscape (Module 5)."""),

        md("""## Step 5: the trained circuit with noise

Now run the 2-layer circuit with the reference angles on the course noise model. Two numbers describe each run:

- the **mean cut** over the shots, compared with the noise model's expected cut; its uncertainty is the standard error of the mean, SEM = s / √shots, where s is the standard deviation of the cut over the shots (prepared: `cut_sem`);
- the fraction of shots that gave a **maximum cut**, a binomial fraction with σ from your `shot_sigma`, tested with `within_3_sigma`.

The project quiz asks for your measured mean cut (any run with 4,000 shots) and for the expected fraction of maximum cuts."""),
        code("""def cut_sem(counts, edges):
    \"\"\"Prepared: the standard error of the mean cut: (standard deviation of the per-shot cut) / sqrt(shots).\"\"\"
    cuts = np.array([sum(k[-1 - i] != k[-1 - j] for i, j in edges) for k in counts], dtype=float)
    n = np.array([counts[k] for k in counts], dtype=float)
    mean = np.sum(n * cuts) / n.sum()
    return float(np.sqrt(np.sum(n * (cuts - mean) ** 2) / (n.sum() - 1) / n.sum()))


def expected_values(layers, p2, readout):
    \"\"\"Prepared: the noise model's expected cut and probability of a maximum cut for the reference angles.\"\"\"
    g, b = REFERENCE_ANGLES[layers]
    probs = expected_probabilities(measured(qaoa_circuit(g, b, EDGES, N)), p2, readout)
    cut = sum(p * CUTS[int(s, 2)] for s, p in probs.items())
    pmax = sum(p for s, p in probs.items() if CUTS[int(s, 2)] == MAX_CUT)
    return cut, pmax


RUN_SEED = 1                     # any whole number; change it to see another run
sim = AerSimulator(noise_model=course_noise_model(0.01, 0.02))
g, b = REFERENCE_ANGLES[2]
qc = measured(qaoa_circuit(g, b, EDGES, N))
cut_exp, pmax_exp = expected_values(2, 0.01, 0.02)
counts = sim.run(qc, shots=4000, seed_simulator=RUN_SEED).result().get_counts()
kmax = sum(v for s, v in counts.items() if CUTS[int(s, 2)] == MAX_CUT)
print(f"mean cut: ideal {ideal_cut(g + b):.4f}, noise model expected {cut_exp:.4f}, measured {mean_cut(counts, EDGES):.4f} ± {cut_sem(counts, EDGES):.4f} (SEM)")
print(f"maximum cut: ideal 0.7070, expected {pmax_exp:.4f}, measured {kmax / 4000:.4f}, sigma {shot_sigma(pmax_exp, 4000):.4f}, "
      f"within 3 sigma: {within_3_sigma(kmax, 4000, pmax_exp)}")"""),
        md("""**What to notice.** With noise the 2-layer circuit gives an expected cut of 4.333 instead of 4.623: it keeps about 82% of its gain over a random split (1.333 of 1.623). The maximum cut now comes out 59% of the time instead of 71%. The noise pushes the distribution toward random splits, whose average cut is 3."""),

        md("""## Step 6: 1, 2 or 3 layers with noise?

The cell runs the three trained circuits on the course noise model, 4,000 shots each, prints the expected and measured mean cuts, and tests the measured differences between neighbouring layer counts with σ_diff = √(SEM₁² + SEM₂²)."""),
        code("""rows = {}
for p in (1, 2, 3):
    g, b = REFERENCE_ANGLES[p]
    cts = sim.run(measured(qaoa_circuit(g, b, EDGES, N)), shots=4000, seed_simulator=RUN_SEED + 10 + p).result().get_counts()
    cut_exp, pmax_exp = expected_values(p, 0.01, 0.02)
    rows[p] = (ideal_cut(g + b), cut_exp, mean_cut(cts, EDGES), cut_sem(cts, EDGES))
print("layers  rzz gates   ideal    expected (noise)   measured ± SEM")
for p, (ic, ec, mc, se) in rows.items():
    print(f"  {p}        {6 * p:2d}      {ic:.4f}       {ec:.4f}         {mc:.4f} ± {se:.4f}")
for a, b2 in ((1, 2), (2, 3)):
    d, sd = rows[b2][2] - rows[a][2], math.sqrt(rows[a][3] ** 2 + rows[b2][3] ** 2)
    print(f"{a} -> {b2} layers: measured gain {d:+.4f}, sigma_diff {sd:.4f} = {d / sd:+.1f} sigma_diff,",
          "significant" if abs(d) > 3 * sd else "not significant")

plt.figure(figsize=(6.5, 3.4))
plt.plot([1, 2, 3], [rows[p][0] for p in (1, 2, 3)], "k--o", label="ideal")
plt.plot([1, 2, 3], [rows[p][1] for p in (1, 2, 3)], "C0-s", label="noise model, expected")
plt.errorbar([1, 2, 3], [rows[p][2] for p in (1, 2, 3)], yerr=[3 * rows[p][3] for p in (1, 2, 3)], fmt="C3o", capsize=4, label="4000 shots, ±3 SEM")
plt.axhline(3, color="gray", ls=":", lw=1, label="random split")
plt.xticks([1, 2, 3]); plt.xlabel("layers"); plt.ylabel("mean cut"); plt.legend(fontsize=8); plt.tight_layout(); plt.show()"""),
        md("""**What to notice.** Under the course noise model the third layer still helps: the expected gain from 2 to 3 layers is 0.111 (ideal: 0.203), about 5σ_diff with 4,000 shots each, so your measured gain will almost always be significant. But noise takes a larger share of each new layer's gain: of the gain that each layer adds without noise, it keeps about 85% for the first layer (over a random split), 75% for the second and 55% for the third."""),

        md("""## Step 7: how much noise can a third layer take?

The cell scans P2 from 0 to 0.050 in steps of 0.001 (READOUT stays 0.02) and computes the expected cut for 2 and 3 layers; no shots. It prints the **crossover**: the smallest P2 in the scan at which 2 layers give a higher expected cut than 3. The project quiz asks for it. It takes about half a minute."""),
        code("""p2_values = np.round(np.arange(0, 0.0505, 0.001), 3)
two = [expected_values(2, p, 0.02)[0] for p in p2_values]
three = [expected_values(3, p, 0.02)[0] for p in p2_values]
crossover = next(p for p, a, b in zip(p2_values, two, three) if a > b)
print(f"crossover: from P2 = {crossover:.3f}, 2 layers give a higher expected cut than 3")
i = list(p2_values).index(0.03)
print(f"at P2 = 0.030: 2 layers {two[i]:.4f}, 3 layers {three[i]:.4f}, difference {three[i] - two[i]:+.4f}")

plt.figure(figsize=(7, 3.3))
plt.plot(p2_values, two, label="2 layers"); plt.plot(p2_values, three, label="3 layers")
plt.axvline(crossover, color="gray", ls=":", lw=1)
plt.xlabel("P2 (two-qubit depolarizing error)"); plt.ylabel("expected cut"); plt.legend(); plt.tight_layout(); plt.show()"""),
        md("""**What to notice.** The curves cross near P2 = 0.032. At P2 = 0.030, the highest value in the personal range, 3 layers are still better by 0.006 in expectation, but that is far below σ_diff (about 0.024 for 4,000 shots each): no affordable experiment would show it. In practice, on a computer near the crossover the extra layer costs gates, angles to train and quantum time, and gives nothing measurable."""),

        md("""## Step 8: your computer, and your verification value

Open your **project quiz for this experiment** in Canvas. Question 1 shows your LAYERS (1, 2 or 3), your P2 and your READOUT. Type them below and run the cells.

The first cell runs the circuit with the reference angles for LAYERS on **your** noise model with 4,000 shots and keeps the counts as `my_counts`. The check cell tests your three statistics functions, `qaoa_circuit()` and `mean_cut()`, and checks that `my_counts` agree with your noise model by the 3σ rule (for the mean cut, 3 SEM). Only if everything passes does it print your **verification value**: 1,000 times the noise model's exact expected cut on your computer, rounded to a whole number."""),
        code("""LAYERS = 0       # your number from Canvas: 1, 2 or 3
P2 = 0.0         # your number from Canvas, for example 0.017
READOUT = 0.0    # your number from Canvas, for example 0.015"""),
        code("""g, b = REFERENCE_ANGLES[int(LAYERS)]
my_counts = AerSimulator(noise_model=course_noise_model(P2, READOUT)).run(
    measured(qaoa_circuit(g, b, EDGES, N)), shots=4000, seed_simulator=RUN_SEED + 100).result().get_counts()
my_cut, my_pmax = expected_values(int(LAYERS), P2, READOUT)
print(f"{int(LAYERS)} layer(s) on your computer: expected cut {my_cut:.4f}, measured {mean_cut(my_counts, EDGES):.4f} ± {cut_sem(my_counts, EDGES):.4f} (SEM)")
print(f"expected fraction of maximum cuts: {my_pmax:.4f}")"""),
        code(check_source("l2_m6_qaoa_check.py", '''passed, messages, value = check_l2_m6_qaoa(qaoa_circuit, mean_cut, shot_sigma, within_3_sigma, difference_sigma,
                                           LAYERS, P2, READOUT, my_counts)
for m in messages:
    print(m)
if passed:
    print("\\nAll tests passed. Your verification value is", value)
else:
    print("\\nNot yet: fix the item above and run this cell again.")''')),
    ]
    c += limits_and_summary(
        """- **Gates as a real computer runs them.** Here `rzz` is one two-qubit gate with one error. IBM computers run it as `cz` gates and single-qubit rotations, and the triangle (0, 1, 2) cannot be placed on a heavy-hex chip without swaps (Module 4): each layer would cost more two-qubit gates, and the crossover would come at a lower P2.
- **Training without noise.** For depolarizing noise, the noise mostly shrinks the landscape toward the random-split value without moving its peak, so angles trained without noise stay near the best. Errors that rotate the state in the same direction every time do move the peak; training on the computer itself, or with a more detailed noise model, can then find better angles.
- **A model, not a computer.** The same error on every gate and qubit, no drift, no crosstalk between neighbouring qubits.
- **One small graph.** With 5 nodes, brute force is instant. QAOA is only interesting for graphs far too large for brute force, where no one knows the maximum cut to compare with.""",
        "the Module 6 project quiz for this experiment, QAOA with noise,")
    return c
