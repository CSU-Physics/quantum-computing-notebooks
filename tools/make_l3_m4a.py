"""Build the Level 3 Module 4 Track A browser notebook (a quantum-kernel classifier against a fair classical
baseline: the ZZ feature map, kernel matrices, SVMs on two data sets, the cost in shots, and kernel concentration).
The check code is checks/l3_m4a_check.py, published next to the notebook by l3_hidden.
Data: content/level3/l3_m4a_data.json (tools/make_l3_m4a_data.py)."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level3"
from l3_hidden import hidden_check  # noqa: E402
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}
NAME = "QC-L3-M4A-lab-quantum-kernel.ipynb"
VERSION = "2026-10-09"


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md(f"""# Module 4, Track A lab: a quantum-kernel classifier

**Quantum Computing Advanced · Module 4 · Track A · about 110 minutes** · notebook version {VERSION}

A support vector machine (SVM) separates two classes of points using only a **kernel**: a number K(x, x′) that says how similar two points are. A **quantum kernel** encodes each point in a quantum state |φ(x)⟩ with a circuit, the **feature map**, and uses K(x, x′) = |⟨φ(x)|φ(x′)⟩|². In this lab you build one and test it honestly:

1. look at two data sets of 60 points with two features;
2. write `feature_map(x)`: the ZZ feature map, the same circuit as Qiskit's `zz_feature_map`;
3. write `kernel_matrix(XA, XB)`: every kernel value between two lists of points;
4. train an SVM with your quantum kernel and a classical SVM on the **same split**, tuned the **same way**, and compare their test accuracy;
5. repeat on a data set that was built for the quantum kernel;
6. write `kernel_entry_shots(x1, x2, shots, rng)`: estimate a kernel value from measurements, as a quantum computer must, and count the circuits;
7. see what happens to kernel values as the number of qubits grows;
8. write an honest report;
9. get your verification value.

The **Module 4 Track A quiz** in Canvas asks for results from this notebook, so keep it open in a second tab. Run each code cell with **Shift + Enter**, in order."""),

md("""## Step 0: start Python

Run the cell below. The first time, it also downloads scikit-learn, the classical machine-learning library used for the SVMs, so it can take a minute or two. Later runs are faster."""),
code("""import json
import numpy as np
import matplotlib.pyplot as plt
import qsim
from qsim import QuantumCircuit, Statevector
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

data = json.load(open("l3_m4a_data.json"))
print("Ready. qsim", qsim.__version__, "| NumPy", np.__version__, "| scikit-learn", sklearn.__version__)"""),

md("""## Step 1: two data sets

Each data set has 60 points with two features, already scaled to angles in radians, and two classes. Each is split once into **40 training points** and **20 test points**, 10 of each class in the test set. The test points are used only at the very end, to measure accuracy; nothing is tuned on them.

- **moons**: two interleaved half-moons, a standard test set from scikit-learn. Nothing quantum about it.
- **adhoc**: points labelled by the quantum feature map itself, as in Havlíček et al. (Nature, 2019). It was **built** so that the quantum kernel can separate it."""),
code("""def load(name):
    d = data[name]
    return (np.array(d["X_train"]), np.array(d["y_train"]), np.array(d["X_test"]), np.array(d["y_test"]))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, name in zip(axes, ("moons", "adhoc")):
    Xtr, ytr, Xte, yte = load(name)
    for cls, col in ((0, "#99004C"), (1, "#24313D")):
        ax.scatter(*Xtr[ytr == cls].T, c=col, s=28, label=f"class {cls}, training")
        ax.scatter(*Xte[yte == cls].T, facecolors="none", edgecolors=col, s=60, label=f"class {cls}, test")
    ax.set_title(name)
    ax.set_xlabel("feature 1 (rad)")
    ax.set_ylabel("feature 2 (rad)")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.show()
for name in ("moons", "adhoc"):
    Xtr, ytr, Xte, yte = load(name)
    print(f"{name}: {len(Xtr)} training points ({ytr.sum()} of class 1), {len(Xte)} test points ({yte.sum()} of class 1)")"""),
md("""**What to notice.** The moons are easy to see: two curved bands. The adhoc classes look like a checkerboard of irregular patches; their boundary comes from a quantum circuit, not from a simple shape."""),

md("""## Step 2: the feature map

The **ZZ feature map** encodes a point x = (x₀, x₁, …) in one qubit per feature. One repetition is:

1. an H gate on every qubit;
2. a phase gate P(2xᵢ) on qubit i;
3. for every pair of qubits i < j: CX(i, j), then P(2(π − xᵢ)(π − xⱼ)) on qubit j, then CX(i, j) again. The three gates together give the pair a phase that depends on the product of the two features: this is where the qubits become entangled.

The lab uses **two repetitions** (`reps=2`), as Qiskit's `zz_feature_map(n, reps=2)` does by default. Write `feature_map(x, reps=2)` for any number of features, and return the circuit (no measurements)."""),
code("""def feature_map(x, reps=2):
    \"\"\"The ZZ feature map: a QuantumCircuit with len(x) qubits and `reps` repetitions.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete feature_map() first.")


print(feature_map([0.4, 1.3]).draw())
print("gates:", dict(feature_map([0.4, 1.3]).count_ops()))"""),
md("""**What to notice.** For two features the circuit has 4 H gates, 4 single-qubit phases and 2 CX-phase-CX blocks: 14 gates. The amplitudes of the final state depend on the data in a way that is hard to write down by hand, which is the point of a quantum feature map."""),

md("""## Step 3: the kernel matrix

The kernel between two points is K(x, x′) = |⟨φ(x)|φ(x′)⟩|², a number between 0 and 1, and 1 when x = x′. Write `kernel_matrix(XA, XB)`, which returns the array with K[a, b] = K(XA[a], XB[b]).

Tip: compute the state of each point once, with `Statevector(feature_map(x)).data`, and then take inner products with `np.vdot(a, b)` (which conjugates its first argument). A training kernel needs 40 × 40 values, but only 40 states."""),
code("""def kernel_matrix(XA, XB):
    \"\"\"K[a, b] = |<phi(XA[a]) | phi(XB[b])>|^2 for the ZZ feature map.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete kernel_matrix() first.")


Xtr, ytr, Xte, yte = load("moons")
K_train = kernel_matrix(Xtr, Xtr)
print("training kernel:", K_train.shape, " diagonal all 1:", np.allclose(np.diag(K_train), 1),
      " symmetric:", np.allclose(K_train, K_train.T))
off = K_train[np.triu_indices(len(Xtr), 1)]
print(f"off-diagonal kernel values: mean {off.mean():.3f}, smallest {off.min():.3f}, largest {off.max():.3f}")
plt.figure(figsize=(4.6, 4))
plt.imshow(K_train, cmap="magma", vmin=0, vmax=1)
plt.colorbar(label="K(x, x')")
plt.title("moons: training kernel matrix")
plt.show()"""),
md("""**What to notice.** The matrix is symmetric with 1s on the diagonal. Most off-diagonal values lie between 0 and 1: some pairs of points look alike to the kernel, most do not."""),

md("""## Step 4: quantum kernel against a classical baseline

A comparison is only fair if both models get the **same data split** and the **same tuning effort**. Both SVMs below choose their settings by 4-fold cross-validation on the 40 training points only:

- **quantum kernel**: an SVM with your kernel matrix (`kernel="precomputed"`); it tunes the regularization C over 0.1, 1, 10, 100;
- **classical baseline**: an SVM with the radial basis function (RBF) kernel, K(x, x′) = exp(−γ |x − x′|²), the standard classical choice; it tunes C over the same values and γ over 0.1, 0.3, 1, 3, 10.

Then each model predicts the 20 test points once. Run the cell."""),
code("""CV = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
C_GRID = [0.1, 1, 10, 100]


def quantum_svm(Xtr, ytr, Xte, yte):
    search = GridSearchCV(SVC(kernel="precomputed"), {"C": C_GRID}, cv=CV)
    search.fit(kernel_matrix(Xtr, Xtr), ytr)
    return search.score(kernel_matrix(Xte, Xtr), yte), search.best_params_


def classical_svm(Xtr, ytr, Xte, yte):
    search = GridSearchCV(SVC(kernel="rbf"), {"C": C_GRID, "gamma": [0.1, 0.3, 1, 3, 10]}, cv=CV)
    search.fit(Xtr, ytr)
    return search.score(Xte, yte), search.best_params_


results = {}
Xtr, ytr, Xte, yte = load("moons")
results["moons"] = (quantum_svm(Xtr, ytr, Xte, yte), classical_svm(Xtr, ytr, Xte, yte))
(q_acc, q_par), (c_acc, c_par) = results["moons"]
print(f"moons, test accuracy: quantum kernel {q_acc:.2f} (C = {q_par['C']}),  classical RBF {c_acc:.2f} "
      f"(C = {c_par['C']}, gamma = {c_par['gamma']})")"""),
md("""**What to notice.** On the moons the quantum kernel reaches 0.70 and the classical RBF kernel 1.00 on the same 20 test points. With 20 test points each point is worth 0.05, so a difference of one or two points means little, but this gap is six points. The quantum kernel is not automatically better: its notion of similarity does not match the shape of these data."""),

md("""## Step 5: a data set built for the quantum kernel

Now the **adhoc** data. Its labels were made from the same ZZ feature map: a point is class 1 when a certain measurement on |φ(x)⟩ (after a fixed random two-qubit unitary) is above +0.3, and class 0 when it is below −0.3. Run the same two models."""),
code("""Xtr, ytr, Xte, yte = load("adhoc")
results["adhoc"] = (quantum_svm(Xtr, ytr, Xte, yte), classical_svm(Xtr, ytr, Xte, yte))
(q_acc, q_par), (c_acc, c_par) = results["adhoc"]
print(f"adhoc, test accuracy: quantum kernel {q_acc:.2f} (C = {q_par['C']}),  classical RBF {c_acc:.2f} "
      f"(C = {c_par['C']}, gamma = {c_par['gamma']})")"""),
md("""**What to notice.** Now the quantum kernel classifies every test point (1.00) and the RBF kernel gets 0.80. That is not evidence of a quantum advantage for real problems: the data were constructed from the quantum feature map, so the kernel "knows" the rule. Whether real data have such structure is the open question of quantum machine learning; Havlíček et al. used the same construction to show that a quantum kernel *can* help, not that it usually does."""),

md("""## Step 6: kernels from shots

A quantum computer cannot output |⟨φ(x)|φ(x′)⟩|² directly. It runs the **overlap circuit**: the feature map of x, then the feature map of x′ run backwards (`feature_map(x2).inverse()`). The probability that every qubit then reads 0 is exactly K(x, x′). With a finite number of shots you get an estimate: the fraction of shots that gave all zeros.

Write `kernel_entry_shots(x1, x2, shots, rng)`. Build the overlap circuit with `feature_map(x1).compose(feature_map(x2).inverse())`, take the probability of all zeros, p = |amplitude 0|² from `Statevector`, and draw the number of all-zero shots with `rng.binomial(shots, p)`. Return that number divided by `shots`."""),
code("""def kernel_entry_shots(x1, x2, shots, rng):
    \"\"\"Estimate K(x1, x2) from `shots` runs of the overlap circuit: the fraction of all-zero results.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete kernel_entry_shots() first.")


SHOTS = 1000
rng = np.random.default_rng(2026)
Xtr, ytr, Xte, yte = load("adhoc")
n_tr, n_te = len(Xtr), len(Xte)
K_tr = np.eye(n_tr)                                       # K(x, x) = 1 needs no circuit
for a in range(n_tr):
    for b in range(a + 1, n_tr):                          # K is symmetric: one circuit per pair
        K_tr[a, b] = K_tr[b, a] = kernel_entry_shots(Xtr[a], Xtr[b], SHOTS, rng)
K_te = np.array([[kernel_entry_shots(xa, xb, SHOTS, rng) for xb in Xtr] for xa in Xte])
circuits = n_tr * (n_tr - 1) // 2 + n_te * n_tr
exact = kernel_matrix(Xtr, Xtr)
print(f"circuits needed: {n_tr * (n_tr - 1) // 2} for training + {n_te * n_tr} for testing = {circuits}, "
      f"{circuits * SHOTS:,} shots in all")
print(f"average error of a kernel value with {SHOTS} shots: {np.abs(K_tr - exact)[np.triu_indices(n_tr, 1)].mean():.4f}")
search = GridSearchCV(SVC(kernel="precomputed"), {"C": C_GRID}, cv=CV).fit(K_tr, ytr)
print(f"adhoc test accuracy with the {SHOTS}-shot kernel: {search.score(K_te, yte):.2f}")"""),
md("""**What to notice.** A 40-point training set and 20 test points need 1,580 circuits; at 1,000 shots each that is 1.58 million shots, far beyond a free hardware budget. With 1,000 shots each kernel value is off by about 0.01 on average, and here the classifier still gets every test point right, because the classes are well separated. In the capstone the hardware limit is 8 training and 4 test points: 28 + 32 = 60 circuits."""),

md("""## Step 7: more qubits, smaller kernel values

More features mean more qubits. The cell below takes 50 random pairs of points for 1 to 8 qubits (features drawn uniformly from 0 to 2π) and computes their kernel values with your `feature_map`. It then asks: with 1,000 shots, is the statistical error of an estimate small compared with how much the kernel values differ from pair to pair?"""),
code("""rng = np.random.default_rng(7)
rows = []
for n in range(1, 9):
    vals = []
    for _ in range(50):
        a, b = rng.uniform(0, 2 * np.pi, n), rng.uniform(0, 2 * np.pi, n)
        vals.append(abs(np.vdot(Statevector(feature_map(a)).data, Statevector(feature_map(b)).data)) ** 2)
    vals = np.array(vals)
    shot_err = np.sqrt(vals.mean() * (1 - vals.mean()) / 1000)     # one standard deviation with 1,000 shots
    rows.append((n, vals.mean(), vals.std(), shot_err))
    print(f"{n} qubits: mean kernel {vals.mean():.4f}   spread (std) {vals.std():.4f}   1,000-shot error {shot_err:.4f}"
          f"   2^-n = {2.0 ** -n:.4f}")
r = np.array(rows)
plt.figure(figsize=(6, 3.8))
plt.semilogy(r[:, 0], r[:, 1], "o-", color="#99004C", label="mean kernel value")
plt.semilogy(r[:, 0], r[:, 2], "s-", color="#24313D", label="spread of kernel values")
plt.semilogy(r[:, 0], r[:, 3], "^--", color="#C9A227", label="error of a 1,000-shot estimate")
plt.semilogy(r[:, 0], 2.0 ** -r[:, 0], ":", color="grey", label="2^-n")
plt.xlabel("qubits")
plt.legend(fontsize=8)
plt.title("Kernel values concentrate as qubits are added")
plt.show()"""),
md("""**What to notice.** The mean kernel value falls like 2⁻ⁿ: with many qubits, almost every pair of points looks completely different to the kernel. The spread falls just as fast, while the shot error falls only like its square root. At 8 qubits the 1,000-shot error is already about half the spread; a few qubits more and the estimates are mostly noise unless the shots grow like 2ⁿ. This **exponential concentration** (Thanasilp et al., 2024) is why a quantum kernel needs a feature map designed for the data, not just more qubits."""),

md("""## Step 8: an honest report

A result in quantum machine learning is only meaningful next to a fair classical baseline. Run the cell to collect yours."""),
code("""print(f"{'data set':10s}{'quantum kernel':>16s}{'classical RBF':>16s}")
for name, ((q_acc, _), (c_acc, _)) in results.items():
    print(f"{name:10s}{q_acc:16.2f}{c_acc:16.2f}")
print("\\nSame 40/20 split, same 4-fold cross-validation on the training set, test set used once.")"""),
md("""Write two or three sentences for yourself before the quiz: on which data set did each kernel win, why, and what would you need to show before claiming that a quantum kernel helps on a real problem? (A larger test set, a baseline tuned as hard as the quantum model, several random splits, and data whose structure the feature map is designed to capture; and, on hardware, enough shots.)"""),

md("""## Step 9: your personal check

Open the **Module 4 Track A quiz** in Canvas. Question 1 shows your own numbers A, B and C (each 0.10 to 3.00). Type them below and run the next two cells. The check cell tests `feature_map()`, `kernel_matrix()` and `kernel_entry_shots()`. Only if every test passes does it print your **verification value**: 1,000 times the kernel K(x, x′) for x = (A, B) and x′ = (B, C), rounded to a whole number. For example, A, B, C = 1.25, 0.40, 2.10 gives 604."""),
code("""A = 0.0    # your number from Canvas, for example 1.25
B = 0.0    # for example 0.40
C = 0.0    # for example 2.10"""),
code(hidden_check(['l3_m4a_check'], 'check_l3_module4a', '''

passed, messages, value = check_l3_module4a(feature_map, kernel_matrix, kernel_entry_shots, A, B, C)
for m in messages:
    print(m)
if passed:
    print("\\nAll tests passed. Your verification value is", value)
else:
    print("\\nNot yet: fix the item above and run this cell again.")''')),

md("""## What you should notice

- A **quantum kernel** is a similarity measure computed from quantum states; an SVM uses it like any other kernel.
- A fair test needs the **same split**, the **same tuning effort** and a test set used once. On the moons the classical kernel won; on data built from the feature map the quantum kernel won. Neither result shows a quantum advantage for real problems.
- On hardware every kernel value costs a circuit and many shots, and the number of circuits grows with the square of the number of points.
- With many qubits, kernel values **concentrate** near 0 and shot noise swamps them, unless the feature map is designed for the data.

**Next in Canvas:** the Module 4 Track A quiz and the time log."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl3m4ac{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / NAME)
print("wrote", OUT / NAME)
