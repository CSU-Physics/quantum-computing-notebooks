"""Build the Level 3 Module 6 capstone notebook "Quantum-kernel classifier within hardware limits".
Numbers quoted in the text are computed here from checks/l3_m6_kernel_check.py (exact, course noise model)."""
import json
import math
import sys
from pathlib import Path

import nbformat as nbf
import numpy as np
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "level3"), str(ROOT / "checks"), str(ROOT / "tools")]
import l3_m6_kernel_check as K  # noqa: E402
from l3_hidden import hidden_check  # noqa: E402
from l3_m6_text import META, check_cell, code, ending, intro, md, personal_step, step0, step1_statistics, step2_noise  # noqa: E402

NAME = "QC-L3-M6-capstone-kernel-classifier.ipynb"
P2, RO = 0.01, 0.02

Xtr, ytr, Xte, yte = K.capstone_data(str(ROOT / "content" / "level3" / "l3_m4a_data.json"))
ex = lambda a, b: K.exact_entry(a, b, 0, 0)  # noqa: E731
no = lambda a, b: K.exact_entry(a, b, P2, RO)  # noqa: E731
Ke, Kn = K.reference_training_kernel(Xtr, ex), K.reference_training_kernel(Xtr, no)
iu = np.triu_indices(8, 1)
slope, icpt = np.polyfit(Ke[iu], Kn[iu], 1)
diag = float(np.mean([no(x, x) for x in Xtr]))
d = json.load(open(ROOT / "content" / "level3" / "l3_m4a_data.json"))["adhoc"]
XT, yT = np.array(d["X_test"]), np.array(d["y_test"])
acc20 = {n: SVC(kernel="precomputed", C=1).fit(k, ytr).score(K.reference_test_kernel(XT, Xtr, f), yT)
         for n, k, f in (("exact", Ke, ex), ("noisy", Kn, no))}
acc20["rbf"] = SVC(kernel="rbf", C=1).fit(Xtr, ytr).score(XT, yT)
acc4_rbf = SVC(kernel="rbf", C=1).fit(Xtr, ytr).score(Xte, yte)
NUM = dict(k01=Ke[0, 1], k01n=Kn[0, 1], k10n=no(Xtr[1], Xtr[0]), slope=slope, icpt=icpt, diag=diag, acc20=acc20, acc4_rbf=acc4_rbf)
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in NUM.items()})
assert acc4_rbf == 0.25
pq_exact = SVC(kernel="precomputed", C=1).fit(Ke, ytr).predict(K.reference_test_kernel(XT, Xtr, ex))
pq = SVC(kernel="precomputed", C=1).fit(Kn, ytr).predict(K.reference_test_kernel(XT, Xtr, no))  # the noisy classifier
assert (pq == pq_exact).all(), "noisy and exact kernels no longer give the same 20 predictions; revise the text"
pr = SVC(kernel="rbf", C=1).fit(Xtr, ytr).predict(XT)
ONLY_Q, ONLY_R = int(((pq == yT) & (pr != yT)).sum()), int(((pq != yT) & (pr == yT)).sum())
NDIS = ONLY_Q + ONLY_R
P_SIGN = min(1.0, 2 * sum(math.comb(NDIS, i) for i in range(min(ONLY_Q, ONLY_R) + 1)) / 2 ** NDIS)
assert (ONLY_Q, ONLY_R) == (6, 0) and 0.003 < P_SIGN < 0.05, (ONLY_Q, ONLY_R, P_SIGN)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


K20 = round(acc20["noisy"] * 20)
W4, W20 = wilson(4, 4), wilson(K20, 20)
assert K20 == 18 and abs(W4[0] - 0.510) < 0.001 and abs(W20[0] - 0.699) < 0.001 and abs(W20[1] - 0.972) < 0.001, (K20, W4, W20)

cells = [
intro("Quantum-kernel classifier within hardware limits",
      "Does the quantum-kernel classifier of Module 4 still work when every kernel entry comes from a noisy quantum computer "
      "with the limits of a real run (8 training points, 4 test points, 1,000 shots per entry), and what can 4 test points "
      "tell you about it?",
      """1. write the statistics functions every capstone uses (Step 1) and meet the capstone noise model (Step 2);
2. build the **overlap circuit** and estimate one **kernel entry** from 1,000 shots (Step 3);
3. build the **training and test kernels** with one circuit per entry (Step 4);
4. train the support vector classifier on exact, noisy and shot-based kernels, and compare with a classical kernel (Step 5);
5. find out **why the noise does so little harm**, and what 4 test points can and cannot show (Step 6);
6. get your verification value (Step 7).""",
      "Question 1 gives you three personal numbers: PAIR, P2 and READOUT."),
*step0("\nfrom sklearn.svm import SVC"),
*step1_statistics("kernel"),
*step2_noise(),

md(f"""## Step 3: one kernel entry

In Module 4 Track A, the **ZZ feature map** U(x) put a data point x = (x₁, x₂) into a two-qubit state |φ(x)⟩ = U(x)|00⟩, and the **quantum kernel** was the overlap

K(x, x′) = |⟨φ(x′)|φ(x)⟩|².

A quantum computer measures it with the **overlap circuit**: apply U(x), then U(x′)⁻¹, and measure. The probability of reading **00** is |⟨00|U(x′)⁻¹U(x)|00⟩|² = K(x, x′). So K is the fraction of shots that give 00, a fraction with σ = √(K(1 − K)/shots).

The prepared cell defines `feature_map(x)` (the Module 4 map, two repetitions) and loads the data: 8 training points (4 of each class) and 4 test points (2 of each) from the Module 4 data set.

**Your task:**

- `overlap_circuit(x1, x2)`: a QuantumCircuit with 2 qubits and 2 classical bits: `feature_map(x1)`, then `feature_map(x2).inverse()` (use `qc.compose(..., inplace=True)`), then measure qubit i into classical bit i.
- `kernel_from_counts(counts)`: the fraction of shots (or probability) with every bit 0. Divide by the total; a key that never appeared is missing from the counts."""),
code("""import json


def feature_map(x, reps=2):
    \"\"\"Prepared: the ZZ feature map of Module 4 (H, P(2 x_i), and CX-P-CX with angle 2 (pi - x_1)(pi - x_2)).\"\"\"
    x = [float(v) for v in x]
    n = len(x)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        for i in range(n):
            qc.h(i)
        for i in range(n):
            qc.p(2 * x[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                qc.cx(i, j)
                qc.p(2 * (np.pi - x[i]) * (np.pi - x[j]), j)
                qc.cx(i, j)
    return qc


def pick(X, y, k):
    \"\"\"Prepared: the first k points of each class.\"\"\"
    X, y = np.array(X), np.array(y)
    idx = [int(i) for c in sorted(set(y.tolist())) for i in np.where(y == c)[0][:k]]
    return X[idx], y[idx]


DATA = json.load(open("l3_m4a_data.json"))["adhoc"]
X_train, y_train = pick(DATA["X_train"], DATA["y_train"], 4)
X_test, y_test = pick(DATA["X_test"], DATA["y_test"], 2)
print("training points:", len(X_train), " classes", y_train.tolist())
print("test points:    ", len(X_test), " classes", y_test.tolist())"""),
code("""def overlap_circuit(x1, x2):
    \"\"\"feature_map(x1), then feature_map(x2).inverse(), then qubit i measured into classical bit i.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete overlap_circuit() first.")


def kernel_from_counts(counts):
    \"\"\"The fraction of shots (or the probability) with every bit 0.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete kernel_from_counts() first.")"""),
code("""SHOTS_K = 1000
x0, x1 = X_train[0], X_train[1]
print(overlap_circuit(x0, x1).draw())
k_ideal = kernel_from_counts(exact_probabilities(overlap_circuit(x0, x1), 0, 0))
k_noisy = kernel_from_counts(exact_probabilities(overlap_circuit(x0, x1)))
k_run = kernel_from_counts(sample_counts(exact_probabilities(overlap_circuit(x0, x1)), SHOTS_K, seed=1))
s = fraction_sigma(k_noisy, SHOTS_K)
print(f"K(x0, x1): ideal {k_ideal:.4f}   noisy, exact {k_noisy:.4f}   your run of {SHOTS_K} shots {k_run:.4f} +- {s:.4f}")
print(f"K(x1, x0): ideal {kernel_from_counts(exact_probabilities(overlap_circuit(x1, x0), 0, 0)):.4f}"
      f"   noisy, exact {kernel_from_counts(exact_probabilities(overlap_circuit(x1, x0))):.4f}")
print(f"K(x0, x0): ideal {kernel_from_counts(exact_probabilities(overlap_circuit(x0, x0), 0, 0)):.4f}"
      f"   noisy, exact {kernel_from_counts(exact_probabilities(overlap_circuit(x0, x0))):.4f}")"""),
md(f"""**What to notice.** The ideal K(x₀, x₁) is {NUM['k01']:.4f}; the noise lowers it to **{NUM['k01n']:.4f}**, a bias of about {NUM['k01'] - NUM['k01n']:.3f}, more than twice the shot noise of 1,000 shots. Three effects of the noise show here:

- **The diagonal is not 1.** K(x, x) is 1 in theory, but the noisy circuit gives about **{NUM['diag']:.3f}**: the circuit still has all its gates, and they make errors.
- **The kernel is not symmetric.** K(x₁, x₀) = {NUM['k10n']:.4f} differs from K(x₀, x₁) = {NUM['k01n']:.4f}. The two circuits are inverses of each other and meet the noise at different places; the readout error is not symmetric either.
- **Shot noise.** With 1,000 shots every entry has σ up to √(0.25 / 1000) ≈ 0.016."""),

md("""## Step 4: the kernel matrices

The classifier needs two matrices:

- the **training kernel**, 8 × 8, K[i][j] = K(xᵢ, xⱼ) for the training points;
- the **test kernel**, 4 × 8, one row per test point and one column per training point.

Every entry is a circuit, and on hardware every circuit costs time. Two savings are standard: the **diagonal is set to 1** (K(x, x) = 1 needs no circuit), and the training kernel is taken as **symmetric**, so only the pairs i < j are run (one circuit each) and each value is copied to [j][i]. That makes 8 · 7 / 2 = 28 training circuits and 4 · 8 = 32 test circuits: **60 circuits** of 1,000 shots.

**Your task:** write both functions for any estimator `estimate(x1, x2)` that returns one entry:

- `training_kernel(X, estimate)`: a len(X) × len(X) NumPy array with 1 on the diagonal and `estimate(X[i], X[j])` for i < j, copied to [j][i]: one call per pair, no more;
- `test_kernel(X_test, X_train, estimate)`: the len(X_test) × len(X_train) array of `estimate(X_test[a], X_train[b])`.

The cell after them builds three versions: **exact** (no noise), **noisy exact** (the noise model's exact probabilities) and **shots** (1,000 shots per circuit, the kernel a real run would give)."""),
code("""def training_kernel(X, estimate):
    \"\"\"len(X) x len(X): 1 on the diagonal, estimate(X[i], X[j]) for i < j, copied to [j][i].\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete training_kernel() first.")


def test_kernel(X_test, X_train, estimate):
    \"\"\"len(X_test) x len(X_train): estimate(X_test[a], X_train[b]).\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete test_kernel() first.")"""),
code("""calls = {"n": 0}


def est_exact(a, b):
    return kernel_from_counts(exact_probabilities(overlap_circuit(a, b), 0, 0))


def est_noisy(a, b):
    return kernel_from_counts(exact_probabilities(overlap_circuit(a, b)))


def est_shots(a, b):
    calls["n"] += 1                                  # one circuit of 1,000 shots, with its own seed
    return kernel_from_counts(sample_counts(exact_probabilities(overlap_circuit(a, b)), SHOTS_K, seed=calls["n"]))


kernels = {}
for name, est in (("exact", est_exact), ("noisy", est_noisy), ("shots", est_shots)):
    kernels[name] = (np.array(training_kernel(X_train, est)), np.array(test_kernel(X_test, X_train, est)))
print("circuits run for the shot-based kernels:", calls["n"], "  (expected 60)")

fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
for a, name in zip(ax, kernels):
    im = a.imshow(kernels[name][0], vmin=0, vmax=1, cmap="viridis")
    a.set_title(name); a.set_xticks([]); a.set_yticks([])
fig.colorbar(im, ax=ax, shrink=0.8); plt.show()
off = np.triu_indices(8, 1)
print("largest |noisy - exact| off the diagonal:", round(float(np.max(np.abs(kernels["noisy"][0] - kernels["exact"][0])[off])), 4))
print("largest |shots - noisy| off the diagonal:", round(float(np.max(np.abs(kernels["shots"][0] - kernels["noisy"][0])[off])), 4))"""),
md("""**What to notice.** The three matrices look almost the same: the same two blocks of similar points (the first four points are one class, the last four the other). The noise shifts entries by up to about 0.07, and the shot noise adds a few hundredths more. Whether that matters depends on the classifier, which is the next step."""),

md("""## Step 5: train and test

The **support vector classifier** (SVC) of Module 4 accepts a precomputed kernel: `SVC(kernel="precomputed", C=1).fit(K_train, y_train)` and then `.score(K_test, y_test)` gives the fraction of test points classified correctly. For comparison, a classical SVC with the **RBF kernel** works on the raw features. Run the cell."""),
code("""print(f"{'kernel':>22s}{'training accuracy':>19s}{'test accuracy (4 points)':>26s}")
models = {}
for name, (Ktr, Kte) in kernels.items():
    models[name] = SVC(kernel="precomputed", C=1).fit(Ktr, y_train)
    print(f"{'quantum, ' + name:>22s}{models[name].score(Ktr, y_train):19.2f}{models[name].score(Kte, y_test):26.2f}")
rbf = SVC(kernel="rbf", C=1).fit(X_train, y_train)
print(f"{'classical RBF':>22s}{rbf.score(X_train, y_train):19.2f}{rbf.score(X_test, y_test):26.2f}")
print("\\nDecision values on the test points (sign = class, size = margin):")
for name, (Ktr, Kte) in kernels.items():
    print(f"   {name:>6s}:", np.round(models[name].decision_function(Kte), 3), "  true classes", y_test.tolist())"""),
md(f"""**What to notice.** All three quantum kernels classify the 4 test points correctly, and the decision values are close to each other: the noise and the shots barely move them. The classical RBF kernel gets only {NUM['acc4_rbf']:.2f}. This data set was **made** from the ZZ feature map, so its classes are simple in the quantum feature space and twisted in the raw features: it is a test of the method, not evidence that quantum kernels beat classical ones on real data."""),

md(f"""## Step 6: why the noise does so little, and what 4 points can show

**The noise acts almost like a straight line.** Plot each noisy entry against the exact one: they lie close to a line, K_noisy ≈ a · K_exact + b. Depolarizing noise mixes the state with the completely mixed one, whose overlap is 1/4, so it shrinks every overlap toward a constant. An SVC is not affected by such a change:

- adding a constant b to every entry changes nothing, because the SVC's weights satisfy Σ αᵢ yᵢ = 0, so the constant cancels;
- multiplying by a > 0 is the same as changing C to C / a: the decision values are identical.

The one part that is not a straight line is the **diagonal**, set to 1 instead of the noisy ≈ {NUM['diag']:.3f}. That adds a small constant to the diagonal only, which acts like a little extra regularization.

**Four test points.** A test accuracy from 4 points is a fraction of 4, with σ = √(p (1 − p) / 4): for a true accuracy of 0.9 that is **0.15**. And a classifier that guesses at random gets all 4 right with probability (1/2)⁴ = **0.0625**. So 4 out of 4 is encouraging but weak evidence. Do not put the observed 4 of 4 into the formula: p = 1 gives σ = 0, a certainty that 4 points cannot give. Report \"4 of 4\" and say how few points that is. If you need an interval, use one made for small counts, such as the **Wilson interval**: for 4 of 4 its 95% range is {W4[0]:.2f} to {W4[1]:.2f}. The usual interval p ± 1.96 √(p (1 − p) / N), the normal approximation, behaves poorly for so few points. Run the cell: it fits the line, shows that the SVC on a K + b with C / a gives the same decision values, and then tests the classifiers on all 20 test points of the data set (exact kernels, no shots), which a real run would pay for with 160 more circuits."""),
code("""Ke, Kn = kernels["exact"][0], kernels["noisy"][0]
a, b = np.polyfit(Ke[off], Kn[off], 1)
print(f"noisy = {a:.3f} x exact + {b:.3f};  spread around the line {np.std(Kn[off] - (a * Ke[off] + b)):.4f}")
plt.figure(figsize=(4.2, 3.6))
plt.plot(Ke[off], Kn[off], "o", ms=4, label="training entries")
plt.plot([0, 1], [b, a + b], "k--", lw=1, label=f"{a:.3f} K + {b:.3f}")
plt.xlabel("exact K"); plt.ylabel("noisy K"); plt.legend(); plt.tight_layout(); plt.show()

m1 = SVC(kernel="precomputed", C=1).fit(Ke, y_train)
m2 = SVC(kernel="precomputed", C=1 / a).fit(a * Ke + b, y_train)
print("decision values, exact kernel:       ", np.round(m1.decision_function(kernels["exact"][1]), 4))
print("decision values, a K + b with C / a: ", np.round(m2.decision_function(a * kernels["exact"][1] + b), 4))

print("\\nWith 4 test points, sigma of an accuracy of 0.9:", round(fraction_sigma(0.9, 4), 3),
      "  chance of 4/4 by guessing:", 0.5 ** 4)


def wilson(k, n, z=1.96):
    \"\"\"95% Wilson interval for k successes in n trials: better than p +- 1.96 sigma when n is small.\"\"\"
    p = k / n
    d = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(float(max(0.0, c - h)), 2), round(float(min(1.0, c + h)), 2)


print("95% Wilson interval for 4 of 4:", wilson(4, 4))
X_all, y_all = np.array(DATA["X_test"]), np.array(DATA["y_test"])
for name, est in (("exact", est_exact), ("noisy", est_noisy)):
    acc = SVC(kernel="precomputed", C=1).fit(kernels[name][0], y_train).score(test_kernel(X_all, X_train, est), y_all)
    k = int(round(acc * len(y_all)))
    print(f"quantum, {name}: {k} of {len(y_all)} test points right, 95% Wilson interval {wilson(k, len(y_all))}")
k = int(round(rbf.score(X_all, y_all) * len(y_all)))
print(f"classical RBF:  {k} of {len(y_all)} test points right, 95% Wilson interval {wilson(k, len(y_all))}")
pred_exact = SVC(kernel="precomputed", C=1).fit(kernels["exact"][0], y_train).predict(test_kernel(X_all, X_train, est_exact))
pred_q = SVC(kernel="precomputed", C=1).fit(kernels["noisy"][0], y_train).predict(test_kernel(X_all, X_train, est_noisy))
print(f"noisy and exact kernels give the same {len(y_all)} predictions: {bool(np.all(pred_q == pred_exact))}")
pred_r = rbf.predict(X_all)
only_q = int(np.sum((pred_q == y_all) & (pred_r != y_all)))
only_r = int(np.sum((pred_q != y_all) & (pred_r == y_all)))
print(f"paired, same {len(y_all)} points: noisy quantum right and RBF wrong on {only_q}, RBF right and noisy quantum wrong on {only_r}")"""),
md(f"""**What to notice.** The noisy entries follow K_noisy ≈ {NUM['slope']:.3f} K_exact + {NUM['icpt']:.3f} closely, and the SVC trained on a K + b with C / a gives exactly the same decision values as the exact kernel: the noise that looks like a straight line is invisible to the classifier. On all 20 test points the quantum kernels reach {NUM['acc20']['exact']:.2f} (exact) and {NUM['acc20']['noisy']:.2f} (noisy), against {NUM['acc20']['rbf']:.2f} for the RBF kernel: the 4-point result of 1.00 was optimistic. These accuracies describe this test set. Their uncertainty comes from which 20 points happen to be in it, not from the shots, which here are not used at all: {K20} of 20 has a 95% Wilson interval of {W20[0]:.2f} to {W20[1]:.2f}.

**Comparing the two classifiers.** Both are tested on the **same** 20 points, so their results are paired, not independent, and combined_sigma does not apply to the difference of their accuracies. Only the points where they disagree carry information. The cell compares the RBF kernel with the quantum kernel built from the **noisy** entries, the classifier a run would use (here without shot noise), and first checks that its 20 predictions are identical to those of the exact kernel, so the result holds for both. The noisy quantum kernel is right on {ONLY_Q} points where the RBF kernel is wrong, and the RBF kernel on {ONLY_R} where the quantum kernel is wrong. If the two were equally good, a split of {NDIS} disagreements as uneven as this would happen by chance with probability about {P_SIGN:.2f} (a two-sided sign test). That is suggestive, but weaker than the course's 3σ standard (a chance of about 0.003), and it is one data set built from the same feature map (see Limits). A claim about one particular run with 1,000-shot entries would repeat this comparison with that run's kernel; Step 5 showed its decision values move only a little. Comparing classifiers needs many more test points than a hardware run of this size allows."""),

*personal_step(7, ["PAIR", "P2", "READOUT"], "PAIR (1 to 28), P2 and READOUT (0.005 to 0.030)",
               "PAIR, P2, READOUT = 1, 0.01, 0.02 gives " + str(K.personal_value(1, 0.01, 0.02)), None, None,
               "10,000 times the exact noisy kernel entry of one training pair with your noise model, rounded to a whole number. "
               "PAIR numbers the 28 pairs i < j in order: 1 is (0, 1), 2 is (0, 2), ..., 7 is (0, 7), 8 is (1, 2), ..., 28 is (6, 7)."),
check_cell(hidden_check, "l3_m6_kernel_check", "check_l3_m6_kernel",
           "fraction_sigma, expectation_sigma, combined_sigma, overlap_circuit, kernel_from_counts,\n"
           "                                             training_kernel, test_kernel, PAIR, P2, READOUT"),
*ending("""- **Very few points.** 8 training points and 4 test points are what a short run on a real computer allows. They show that the method works, not how well it works: the 20-point test is already more informative, and real data would need far more points and circuits (the training kernel grows as the square of the number of points).
- **A data set made for the method.** The labels come from the same feature map. On real data, a quantum kernel has no known advantage, and a classical kernel is the baseline to beat.
- **The noise model is simple.** Its depolarizing noise acts almost like a straight line on the kernel, which the SVC ignores. Coherent errors and drifting calibrations can distort the kernel in ways that are not a straight line, and the diagonal set to 1 hides how noisy the device is.
- **Two qubits.** With more features, the circuits get deeper and the kernel values of different points get very small (they **concentrate** near 0), so far more shots are needed to tell them apart.
- **No C search.** C = 1 was fixed. With so few points, cross-validation (Module 4) would be unreliable, which is itself a limit of small experiments.""",
       "the **Capstone Quiz — Kernel Classifier**", track="kernel"),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl3m6k{i:02d}"
out = ROOT / "content" / "level3" / NAME
nbf.write(nb, out)
print("wrote", out)
