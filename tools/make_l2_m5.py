"""Build the Level 2 Module 5 browser notebook (variational algorithms: VQE for H2 and QAOA for Max-Cut)
from checks/l2_m5_check.py."""
import shutil
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level2"
SRC = (ROOT / "checks" / "l2_m5_check.py").read_text()
from l2_hidden import hidden_check  # noqa: E402
CHECK = SRC.split('"""', 2)[2].lstrip()
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}
NAME = "QC-L2-M5-lab-vqe-qaoa.ipynb"
VERSION = "2026-10-06"


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md(f"""# Module 5 lab: variational algorithms, VQE and QAOA

**Quantum Computing Intermediate · Module 5 · about 100 minutes** · notebook version {VERSION}

A **variational algorithm** splits the work between two computers. A quantum circuit with adjustable angles (the **ansatz**) prepares a trial state; the quantum computer estimates its energy, an **expectation value** ⟨ψ(θ)|H|ψ(θ)⟩; a classical **optimizer** changes the angles to lower that energy, and the loop repeats.

In this lab you run two of them:

1. write a Hamiltonian as a sum of **Pauli strings** (`SparsePauliOp`) and compute expectation values with the **Estimator**;
2. **VQE** for the hydrogen molecule: write a parameterized `ansatz(layers)`, minimize its energy with the COBYLA optimizer, then make one defined change, a second layer, and compare;
3. **QAOA** for Max-Cut on a 4-node graph: write `maxcut_hamiltonian(edges, n)` and `qaoa_circuit(gammas, betas, edges, n)`, find good angles, and check the answer by brute force;
4. get your verification value.

The **Module 5 lab check** in Canvas asks for results from this notebook, so keep it open in a second tab. Run each code cell with **Shift + Enter**, in order."""),

md("""## Step 0: start Python

Run the cell below. The first time, Python can take up to a minute to start in your browser, and SciPy and Matplotlib take a few more seconds to load."""),
code("""import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import qsim
from qsim import QuantumCircuit, Statevector, SparsePauliOp, StatevectorEstimator, Parameter, ParameterVector

estimator = StatevectorEstimator()
print("Ready. qsim", qsim.__version__, "and NumPy", np.__version__)"""),

md("""## Step 1: a Hamiltonian as a sum of Pauli strings

A quantum computer measures Pauli strings such as `ZZ` or `XX` easily, so Hamiltonians are written as sums of them with real coefficients. The cell below defines the Hamiltonian of the **hydrogen molecule** H₂ at a bond length of 0.735 Å, reduced to 2 qubits: the same operator that IBM's *Variational Algorithm Design* course uses. Its energies are in **hartrees** (1 hartree ≈ 27.2 eV).

On 2 qubits the matrix is only 4 × 4, so the cell also finds its exact lowest eigenvalue. That is the answer VQE should find. It is the **electronic** energy; adding the repulsion between the two nuclei, a constant 0.71997 hartree at this distance, gives the total energy of the molecule."""),
code("""H2 = SparsePauliOp.from_list([
    ("II", -1.052373245772859),
    ("IZ", 0.39793742484318045),
    ("ZI", -0.39793742484318045),
    ("ZZ", -0.01128010425623538),
    ("XX", 0.18093119978423156),
])
NUCLEAR_REPULSION = 0.7199689944    # hartree, for two protons 0.735 angstrom apart
print(H2)

E_EXACT = np.linalg.eigvalsh(H2.to_matrix())[0]
print(f"\\nexact lowest electronic energy: {E_EXACT:.6f} hartree")
print(f"total energy with the nuclear repulsion: {E_EXACT + NUCLEAR_REPULSION:.6f} hartree")"""),

md("""## Step 2: expectation values with the Estimator

The energy of a state is the expectation value ⟨ψ|H|ψ⟩ = Σₖ cₖ ⟨ψ|Pₖ|ψ⟩: one expectation value per Pauli string, weighted by its coefficient. The **Estimator** primitive computes them. You give it a list of **PUBs** (primitive unified blocs), each a tuple `(circuit, observable)` or `(circuit, observable, parameter_values)`, and read the result from `.data.evs`.

The cell prepares |01⟩ (an X on qubit 0; qubit 0 is the rightmost bit). For H₂ this is the Hartree-Fock state, the best state that a simple chemistry calculation without entanglement gives. The cell prints each Pauli string's expectation value and adds them up.

*Predict first:* what is ⟨01|XX|01⟩? (XX flips both qubits, so it turns |01⟩ into |10⟩.)"""),
code("""hf = QuantumCircuit(2)
hf.x(0)                                   # |01>

labels = [label for label, c in H2.to_list()]
terms = estimator.run([(hf, [SparsePauliOp(label) for label in labels])]).result()[0].data.evs
for (label, c), ev in zip(H2.to_list(), terms):
    print(f"{label}: coefficient {c.real:+.5f}   <P> = {ev:+.0f}")

energy = estimator.run([(hf, H2)]).result()[0].data.evs
print(f"\\nsum of coefficient x <P>: {sum(c.real * ev for (l, c), ev in zip(H2.to_list(), terms)):.6f}")
print(f"Estimator, whole H2:      {float(energy):.6f} hartree  (exact lowest: {E_EXACT:.6f})")"""),
md("""**What to notice.** ⟨XX⟩ = 0, because |10⟩ is orthogonal to |01⟩. The |01⟩ state misses the exact energy by about 0.020 hartree (20 millihartree). Chemists call an error below 1.6 millihartree (1 kcal/mol) **chemical accuracy**; to get there, the state needs a little of |10⟩ mixed in, which takes entanglement."""),

md("""## Step 3: a parameterized ansatz

In Qiskit (and in qsim) a circuit can hold **parameters** instead of numbers: `theta = ParameterVector("θ", 4)` makes θ[0] to θ[3], and `qc.ry(theta[0], 0)` uses one. The Estimator fills in the values for each run, so the same circuit serves every step of the optimization. `qc.assign_parameters(values)` gives a copy with numbers in place.

**Your task:** write `ansatz(layers)`, a 2-qubit circuit with `2 * layers` parameters:

- layer 1: `ry(θ[0])` on qubit 0 and `ry(θ[1])` on qubit 1;
- each further layer k (k = 1, 2, ...): `cx(0, 1)`, then `ry(θ[2k])` on qubit 0 and `ry(θ[2k+1])` on qubit 1.

With 1 layer the qubits stay unentangled; each further layer adds a CNOT. (This is the structure of Qiskit's `real_amplitudes(2, reps=layers - 1)`.) No measurements: the Estimator works with the state itself."""),
code("""def ansatz(layers):
    \"\"\"A 2-qubit parameterized circuit with 2 * layers parameters (see the description above).\"\"\"
    theta = ParameterVector("θ", 2 * layers)
    qc = QuantumCircuit(2)
    # YOUR CODE HERE
    raise NotImplementedError("Complete ansatz() first.")
    return qc"""),
code("""print(ansatz(2).draw())
print("parameters:", ansatz(2).parameters)
print("1 layer with theta = [pi, 0] gives", Statevector(ansatz(1).assign_parameters([np.pi, 0])), " (the state |01>)")"""),

md("""## Step 4: VQE with one layer

The prepared function `vqe(layers)` builds your ansatz, defines the cost function (the energy from the Estimator, one run per call) and hands it to SciPy's **COBYLA**, a gradient-free optimizer, starting from all angles 0 with at most 300 cost evaluations. It returns the optimizer's result and the energy at every evaluation.

Run it with **one layer**. The lab check asks for the lowest energy it finds, to 3 decimals."""),
code("""def vqe(layers, x0=None, maxiter=300):
    \"\"\"Prepared: minimize <psi(theta)|H2|psi(theta)> over the angles of ansatz(layers) with COBYLA.\"\"\"
    qc = ansatz(layers)
    history = []

    def cost(params):
        e = float(estimator.run([(qc, H2, params)]).result()[0].data.evs)
        history.append(e)
        return e

    x0 = np.zeros(qc.num_parameters) if x0 is None else np.asarray(x0, dtype=float)
    result = minimize(cost, x0, method="COBYLA", options={"maxiter": maxiter})
    return result, history


res1, hist1 = vqe(1)
print(f"1 layer: lowest energy {res1.fun:.6f} hartree after {len(hist1)} evaluations")
print(f"error: {1000 * (res1.fun - E_EXACT):.2f} millihartree   (chemical accuracy: below 1.6)")
print("optimal angles:", np.round(res1.x, 3))"""),
md("""**What to notice.** With one layer the two qubits never become entangled, so the best the ansatz can do is a product state. The optimizer finds θ ≈ (π, 0): the |01⟩ state of Step 2, with the same energy. More optimizer time will not help; the ansatz itself cannot reach the answer."""),

md("""## Step 5: the defined change, a second layer

Now make one change and nothing else: run VQE with **two layers** (one CNOT and two more angles). The cell prints the result and plots the energy at every evaluation for both runs. The lab check asks for the **total** energy with two layers (the electronic energy plus the nuclear repulsion), to 3 decimals."""),
code("""res2, hist2 = vqe(2)
print(f"2 layers: lowest energy {res2.fun:.6f} hartree after {len(hist2)} evaluations")
print(f"error: {1000 * (res2.fun - E_EXACT):.3f} millihartree")
print(f"total energy of H2 (with the nuclear repulsion): {res2.fun + NUCLEAR_REPULSION:.6f} hartree")

plt.figure(figsize=(7, 3.5))
plt.plot(hist1, label="1 layer")
plt.plot(hist2, label="2 layers")
plt.axhline(E_EXACT, color="k", ls="--", lw=1, label="exact")
plt.ylim(E_EXACT - 0.02, E_EXACT + 0.25)
plt.xlabel("cost function evaluation"); plt.ylabel("energy (hartree)"); plt.legend(); plt.tight_layout(); plt.show()"""),
md("""**What to notice.** The second layer adds a CNOT, the state can now be entangled, and VQE reaches the exact energy well within chemical accuracy. The optimizer needed more evaluations for 4 angles than for 2. COBYLA's path depends on its version, so your number of evaluations and your angles may differ a little from a classmate's; the lowest energy does not."""),

md("""## Step 6: Max-Cut as a Hamiltonian

**Max-Cut:** split the nodes of a graph into two groups so that as many edges as possible join nodes in different groups (are "cut"). The course graph is a square with one diagonal: nodes 0 to 3, edges (0,1), (1,2), (2,3), (3,0) and (0,2). The cell first finds the answer by **brute force**: it tries all 2⁴ = 16 ways to split the nodes (bit q of a string is node q's group, with node 0 on the right).

To give the problem to a quantum computer, write a **cost Hamiltonian**. For an edge (i, j), Z_i Z_j is +1 when nodes i and j are in the same group and −1 when the edge is cut, so

H_C = Σ over edges of Z_i Z_j,   and   cut size = (number of edges − ⟨H_C⟩) / 2.

The lowest energy of H_C belongs to the maximum cut.

**Your task:** write `maxcut_hamiltonian(edges, n)`: the `SparsePauliOp` sum of Z_i Z_j over the edges, coefficient 1 each, on n qubits. `SparsePauliOp.from_sparse_list([("ZZ", [i, j], 1.0), ...], num_qubits=n)` puts Z on qubits i and j and I elsewhere. The cell after it checks that the formula above gives the brute-force cut for all 16 strings."""),
code("""EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
N = 4


def cut_size(bits, edges):
    \"\"\"Number of edges whose two nodes have different bits (bits[-1 - q] is node q).\"\"\"
    return sum(bits[-1 - i] != bits[-1 - j] for i, j in edges)


cuts = {format(k, "04b"): cut_size(format(k, "04b"), EDGES) for k in range(2 ** N)}
best = max(cuts.values())
print("cut size of each split:", cuts)
print("maximum cut:", best, "for", [s for s, c in cuts.items() if c == best])"""),
code("""def maxcut_hamiltonian(edges, n):
    \"\"\"The cost Hamiltonian: the sum of Z_i Z_j over the edges, as a SparsePauliOp on n qubits.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete maxcut_hamiltonian() first.")"""),
code("""HC = maxcut_hamiltonian(EDGES, N)
print(HC)
ok = all(abs((len(EDGES) - Statevector.from_label(s).expectation_value(HC).real) / 2 - c) < 1e-9 for s, c in cuts.items())
print("\\n(edges - <H_C>) / 2 equals the brute-force cut for all 16 strings:", ok)"""),

md("""## Step 7: QAOA with one layer

QAOA starts in the equal superposition (H on every qubit) and applies p layers. Layer k applies the **cost** step exp(−iγₖH_C), which for each edge is `rzz(2 * gamma, i, j)`, then the **mixer** step exp(−iβₖΣX), which is `rx(2 * beta, q)` on every qubit. An optimizer then tunes the angles γ and β to maximize the expected cut.

**Your task:** write `qaoa_circuit(gammas, betas, edges, n)` for any number of layers (`gammas` and `betas` are lists of the same length). No measurements.

The cells after it: (a) run COBYLA from γ = β = 0.5; (b) scan a grid of 33 × 17 angles, γ from 0 to π and β from 0 to π/2, and plot the landscape; (c) run COBYLA again from the best grid point. The lab check asks for the expected cut that (c) finds, to 2 decimals."""),
code("""def qaoa_circuit(gammas, betas, edges, n):
    \"\"\"QAOA: H on every qubit, then for each layer rzz(2*gamma) on every edge and rx(2*beta) on every qubit.\"\"\"
    qc = QuantumCircuit(n)
    # YOUR CODE HERE
    raise NotImplementedError("Complete qaoa_circuit() first.")
    return qc"""),
code("""def expected_cut(x):
    \"\"\"Prepared: x = [gamma_1, ..., gamma_p, beta_1, ..., beta_p]; returns the expected cut size.\"\"\"
    p = len(x) // 2
    qc = qaoa_circuit(list(x[:p]), list(x[p:]), EDGES, N)
    e = float(estimator.run([(qc, HC)]).result()[0].data.evs)
    return (len(EDGES) - e) / 2


cost = lambda x: -expected_cut(x)          # COBYLA minimizes, so minimize minus the cut

run_a = minimize(cost, [0.5, 0.5], method="COBYLA", options={"maxiter": 300})
print(f"(a) COBYLA from (0.5, 0.5): expected cut {-run_a.fun:.4f} at gamma = {run_a.x[0]:.3f}, beta = {run_a.x[1]:.3f}")

gammas = np.linspace(0, np.pi, 33)
betas = np.linspace(0, np.pi / 2, 17)
grid = np.array([[expected_cut([g, b]) for g in gammas] for b in betas])
k = np.unravel_index(np.argmax(np.round(grid, 9)), grid.shape)    # the first of equal maxima
g0, b0 = gammas[k[1]], betas[k[0]]
print(f"(b) grid scan: best expected cut {grid[k]:.4f} at gamma = {g0:.3f}, beta = {b0:.3f}")

run_c = minimize(cost, [g0, b0], method="COBYLA", options={"maxiter": 300})
print(f"(c) COBYLA from the best grid point: expected cut {-run_c.fun:.4f} at gamma = {run_c.x[0]:.3f}, beta = {run_c.x[1]:.3f}")

plt.figure(figsize=(7, 3.6))
plt.imshow(grid, origin="lower", extent=[0, np.pi, 0, np.pi / 2], aspect="auto", cmap="viridis")
plt.colorbar(label="expected cut")
plt.plot(run_a.x[0] % np.pi, run_a.x[1] % (np.pi / 2), "wx", ms=10, label="(a) from (0.5, 0.5)")
plt.plot(run_c.x[0] % np.pi, run_c.x[1] % (np.pi / 2), "r*", ms=12, label="(c) from the grid")
plt.xlabel("gamma"); plt.ylabel("beta"); plt.legend(loc="upper right", fontsize=8); plt.tight_layout(); plt.show()"""),
code("""probs = Statevector(qaoa_circuit([run_c.x[0]], [run_c.x[1]], EDGES, N)).probabilities_dict()
top = sorted(probs.items(), key=lambda kv: -kv[1])[:5]
for s, pr in top:
    print(f"{s}: probability {pr:.3f}, cut size {cuts[s]}")
print(f"probability of a maximum cut (0101 or 1010): {probs['0101'] + probs['1010']:.3f}")"""),
md("""**What to notice.** From (0.5, 0.5), COBYLA climbs to the nearest hill of the landscape and stops there, at an expected cut of about 3.09: a **local** optimum. The grid shows a higher hill, and COBYLA started there reaches about 3.24. Local optimizers find the top of the hill they start on, so the starting point matters. With one layer, the most likely single outcomes are the two maximum cuts, but the state also gives worse cuts a good part of the time. (The landscape repeats outside the plotted range, so COBYLA's angles may be shifted by a period; the white cross is drawn back inside the plot.)"""),

md("""## Step 8: two layers

More layers give QAOA more freedom. The cell runs COBYLA for **two layers** (four angles), from all angles 0.5."""),
code("""run_d = minimize(cost, [0.5, 0.5, 0.5, 0.5], method="COBYLA", options={"maxiter": 500})
probs2 = Statevector(qaoa_circuit(list(run_d.x[:2]), list(run_d.x[2:]), EDGES, N)).probabilities_dict()
print(f"2 layers: expected cut {-run_d.fun:.4f} (1 layer: {-run_c.fun:.4f}; maximum: {best})")
print(f"probability of a maximum cut: {probs2.get('0101', 0) + probs2.get('1010', 0):.3f}")"""),
md("""**What to notice.** Two layers reach an expected cut of about 3.86 out of 4, and a maximum cut comes out more than 80% of the time. The price is a deeper circuit and twice as many angles to optimize; on real hardware (Module 4) every extra layer adds two-qubit gates and noise."""),

md("""## Step 9: your personal check

Open the **Module 5 lab check** in Canvas. Question 1 shows your own GAMMA and BETA (numbers from 0.10 to 1.50). Type them below and run the next two cells. The check cell tests `ansatz()`, `maxcut_hamiltonian()` and `qaoa_circuit()`. Only if every test passes does it print your **verification value**: 1,000 times the expected cut of one-layer QAOA on the course graph with your γ and β, rounded to a whole number."""),
code("""GAMMA = 0.0    # your number from Canvas, for example 0.37
BETA = 0.0     # your number from Canvas, for example 1.21"""),
code(hidden_check(['l2_m5_check'], 'check_l2_module5', '''

passed, messages, value = check_l2_module5(ansatz, maxcut_hamiltonian, qaoa_circuit, GAMMA, BETA)
for m in messages:
    print(m)
if passed:
    print("\\nAll tests passed. Your verification value is", value)
else:
    print("\\nNot yet: fix the item above and run this cell again.")''')),

md("""## What you should notice

- A Hamiltonian is a sum of Pauli strings; its expectation value is the weighted sum of the strings' expectation values, which is what the Estimator computes.
- A variational algorithm is a loop: a parameterized circuit prepares a trial state, the quantum computer estimates its energy, and a classical optimizer updates the angles. By the variational principle, the exact expectation value of the energy can never go below the true lowest energy; an estimate from a finite number of shots, or from a noisy computer, can fall a little below it.
- The ansatz decides what can be reached: one layer without a CNOT stopped 20 millihartree above the answer for H₂; one CNOT more reached it.
- QAOA turns Max-Cut into finding the lowest energy of H_C = Σ Z_i Z_j. Optimizers can stop at local optima, the starting point matters, and more layers give better cuts at the cost of deeper circuits.

**Next in Canvas:** the lab check, the quiz and the time log."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl2m5c{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / NAME)
shutil.copyfile(ROOT / "content" / "level1" / "qsim.py", OUT / "qsim.py")
print("wrote", OUT / NAME, "and synced qsim.py")
