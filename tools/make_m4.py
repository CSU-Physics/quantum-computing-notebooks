"""Build the Level 1 Module 4 browser notebook (JupyterLite, Pyodide kernel) from checks/m4_check.py."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level1"
CHECK = (ROOT / "checks" / "m4_check.py").read_text().split('"""', 2)[2].lstrip()
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}
NAME = "QC-L1-M4-lab-bell-states.ipynb"


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md("""# Module 4 lab: two qubits, CNOT and the Bell states

**Quantum Computing Foundations · Module 4 · about 85 minutes** · notebook version 2026-10-04

Two qubits have four amplitudes, one for each of |00⟩, |01⟩, |10⟩ and |11⟩. In this lab you write three functions:

1. `product_state(q0, q1)`: the two-qubit state when qubit 0 is in state q0 and qubit 1 in state q1;
2. `bell_circuit(name)`: a circuit that makes each of the four Bell states;
3. `correlation(counts)`: how strongly the two qubits' results agree, from +1 (always the same) to −1 (always different).

Then you measure the correlations of the four Bell states, see why one qubit's results alone carry no message, and use a prepared function to test which states are entangled. The **Module 4 lab check** asks for results from this notebook, so keep it open in a second tab. Run each code cell with **Shift + Enter**, in order."""),
md("""## Step 0: start Python

Run the cell below. The first time, Python can take up to a minute to start in your browser."""),
code("""import numpy as np
import matplotlib.pyplot as plt   # used for the plots below
from qsim import (QuantumCircuit, Statevector, Operator, AerSimulator, transpile,
                  plot_histogram, plot_bloch_multivector, bloch_vectors)

sim = AerSimulator(seed_simulator=7)
print("Ready. NumPy version", np.__version__)"""),
md("""## Step 1: four amplitudes, and Qiskit's bit order

A two-qubit state is a list of four amplitudes, for |00⟩, |01⟩, |10⟩ and |11⟩. In Qiskit (and qsim), **qubit 0 is the rightmost bit** of each label. So |01⟩ means qubit 0 is 1 and qubit 1 is 0. Results are written the same way: the count `'01'` means qubit 0 gave 1.

Run the cell: an X gate on qubit 0 alone gives the state |01⟩, the second amplitude."""),
code("""qc = QuantumCircuit(2)
qc.x(0)                                   # flip qubit 0 only
state = Statevector(qc)
print("amplitudes [|00>, |01>, |10>, |11>]:", np.round(state.data, 4) + 0)
print("as labels:", state.probabilities_dict())"""),
md("""## Step 2: product states

When the two qubits are prepared separately, qubit 0 in the state q0 = [a, b] and qubit 1 in q1 = [c, d], the two-qubit state is their **tensor product**:

[c·a, c·b, d·a, d·b] for |00⟩, |01⟩, |10⟩, |11⟩.

NumPy's `np.kron(q1, q0)` computes exactly this. Note the order: qubit 1 first, because it is the left bit of each label.

Complete `product_state(q0, q1)`, then run the cell. It compares your state with a circuit that prepares qubit 0 in [0.6, 0.8] (an Ry rotation) and qubit 1 in |+⟩. Canvas asks for the amplitude of |01⟩."""),
code("""def product_state(q0, q1):
    # YOUR CODE: replace the next line. Return the 4 amplitudes for |00>, |01>, |10>, |11>.
    raise NotImplementedError("Complete product_state() first.")


q0 = np.array([0.6, 0.8])
q1 = np.array([1, 1]) / np.sqrt(2)
mine = np.asarray(product_state(q0, q1))
print("product_state(q0, q1) =", np.round(mine, 4))

qc = QuantumCircuit(2)
qc.ry(2 * np.arccos(0.6), 0)              # qubit 0 -> [0.6, 0.8]
qc.h(1)                                   # qubit 1 -> |+>
print("circuit                =", np.round(Statevector(qc).data.real, 4))
print("amplitude of |01>:", round(float(np.real(mine[1])), 4))"""),
md("""Both lines should agree. The amplitude of |01⟩ is (amplitude of 0 for qubit 1) × (amplitude of 1 for qubit 0) = (1/√2)(0.8) = 0.566."""),
md("""## Step 3: the CNOT gate

The **CNOT** (controlled-NOT) gate `qc.cx(control, target)` flips the target qubit when the control qubit is 1, and does nothing when it is 0. Run the cell: it applies `cx(0, 1)` to each of the four basis states, then prints its matrix.

Then it runs the circuit **H on qubit 0, then CNOT from qubit 0 to qubit 1**. H puts qubit 0 in a superposition, and CNOT copies "0 or 1" into qubit 1 in both branches at once. The result is the Bell state (|00⟩ + |11⟩)/√2. This is the Module 4 Bell circuit that the final coding task extends to three qubits."""),
code("""for label in ["00", "01", "10", "11"]:
    qc = QuantumCircuit(2)
    if label[1] == "1":
        qc.x(0)                           # the right bit is qubit 0
    if label[0] == "1":
        qc.x(1)
    qc.cx(0, 1)
    out = Statevector(qc).probabilities_dict()
    print(f"cx(0, 1): |{label}>  ->  |{list(out)[0]}>")

qc = QuantumCircuit(2)
qc.cx(0, 1)
print("\\nmatrix of cx(0, 1) in the order |00>, |01>, |10>, |11>:")
print(np.round(Operator(qc).data.real, 0).astype(int))

bell = QuantumCircuit(2)
bell.h(0)
bell.cx(0, 1)
print("\\nH then CNOT gives", np.round(Statevector(bell).data.real, 4), "= (|00> + |11>)/sqrt(2)")"""),
md("""## Step 4: the four Bell states

There are four Bell states. All of them are made by the same H and CNOT, after X gates that choose the starting basis state:

| Name | State | Start from | Gates |
|---|---|---|---|
| `"phi+"` | (\\|00⟩ + \\|11⟩)/√2 | \\|00⟩ | H on qubit 0, cx(0, 1) |
| `"phi-"` | (\\|00⟩ − \\|11⟩)/√2 | \\|01⟩ (X on qubit 0) | H on qubit 0, cx(0, 1) |
| `"psi+"` | (\\|01⟩ + \\|10⟩)/√2 | \\|10⟩ (X on qubit 1) | H on qubit 0, cx(0, 1) |
| `"psi-"` | (\\|01⟩ − \\|10⟩)/√2 | \\|11⟩ (X on both) | H on qubit 0, cx(0, 1) |

Complete `bell_circuit(name)`. It returns a 2-qubit circuit **without** measurements. Then run the cell: it prints each state and measures each one 1,000 times."""),
code("""def bell_circuit(name):
    # YOUR CODE: replace the next line. name is "phi+", "phi-", "psi+" or "psi-".
    raise NotImplementedError("Complete bell_circuit() first.")


def measured(qc):
    \"\"\"A copy of a 2-qubit circuit with both qubits measured.\"\"\"
    m = QuantumCircuit(2, 2)
    for inst in qc.data:
        m.append(inst)
    m.measure([0, 1], [0, 1])
    return m


NAMES = ["phi+", "phi-", "psi+", "psi-"]
all_counts = []
for name in NAMES:
    state = Statevector(bell_circuit(name))
    counts = sim.run(transpile(measured(bell_circuit(name)), sim), shots=1000).result().get_counts()
    all_counts.append(counts)
    print(f"{name}:  state {np.round(state.data.real, 4) + 0}   counts {counts}")
plot_histogram(all_counts, title="phi+, phi-, psi+, psi- (1,000 shots each)")"""),
md("""Each Bell state gives only two results, each about half the time: phi± always give equal results (00 or 11), psi± always give different results (01 or 10). But phi+ and phi− give the same counts here, and so do psi+ and psi−. The difference is a phase, and measuring in another basis reveals it. That is the next step.

psi− may print as [0, −0.7071, 0.7071, 0]. That is −1 times (|01⟩ − |10⟩)/√2: a global phase, so it is the same state (Module 2)."""),
md("""## Step 5: correlations in two bases

The **correlation** of two results is

correlation = (number of equal results − number of different results) / total,

so +1 means the two qubits always agree, −1 means they always disagree, and 0 means no relation. Complete `correlation(counts)`. Use `counts.get("01", 0)` for a result that never appeared.

Then run the cell. It measures each Bell state in the Z basis (as in Step 4) and in the **X basis** (H on both qubits before the measurement, as in Module 2), and plots the two correlations. Canvas asks which state has −1 in both."""),
code("""def correlation(counts):
    # YOUR CODE: replace the next line. counts is a dict such as {"00": 489, "11": 511}.
    raise NotImplementedError("Complete correlation() first.")


def in_x_basis(qc):
    \"\"\"A copy of the circuit with H on both qubits, so a Z measurement after it is an X measurement.\"\"\"
    m = QuantumCircuit(2)
    for inst in qc.data:
        m.append(inst)
    m.h(0)
    m.h(1)
    return m


zz, xx = [], []
for name in NAMES:
    cz = sim.run(transpile(measured(bell_circuit(name)), sim), shots=1000).result().get_counts()
    cxb = sim.run(transpile(measured(in_x_basis(bell_circuit(name))), sim), shots=1000).result().get_counts()
    zz.append(correlation(cz))
    xx.append(correlation(cxb))
    print(f"{name}:  Z-basis correlation {zz[-1]:+.2f}   X-basis correlation {xx[-1]:+.2f}")

fig, ax = plt.subplots(figsize=(6.5, 3))
pos = np.arange(4)
ax.bar(pos - 0.18, zz, width=0.36, color="#24313D", label="Z basis")
ax.bar(pos + 0.18, xx, width=0.36, color="#99004C", label="X basis")
ax.axhline(0, color="#6A626B", linewidth=0.8)
ax.set_xticks(pos)
ax.set_xticklabels(NAMES)
ax.set_ylim(-1.15, 1.15)
ax.set_ylabel("correlation")
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
plt.close(fig)
display(fig)"""),
md("""Each Bell state has its own pair of signs, so the two correlations tell all four apart: phi+ is (+1, +1), phi− is (+1, −1), psi+ is (−1, +1) and psi− is (−1, −1). The correlations are perfect in **both** bases. No pair of separately prepared qubits can do that: this is entanglement."""),
md("""## Step 6: one qubit alone carries no message

Imagine qubit 0 is with Alice and qubit 1 with Bob, far apart, sharing phi+. Alice measures her qubit. Bob either measures his qubit directly, or first applies H. Does Bob's choice change what Alice sees?

Run the cell. It counts how often **qubit 0** (the right bit) gives 0 in both cases. Canvas asks how many shots gave `00` in the second case."""),
code("""def alice_zeros(counts):
    return counts.get("00", 0) + counts.get("10", 0)      # results where the right bit (qubit 0) is 0


plain = sim.run(transpile(measured(bell_circuit("phi+")), sim), shots=1000).result().get_counts()

bob_h = bell_circuit("phi+")
bob_h.h(1)                                                 # Bob turns his qubit before measuring
with_h = sim.run(transpile(measured(bob_h), sim), shots=1000).result().get_counts()

print("Bob measures directly:  ", plain, "  Alice's zeros:", alice_zeros(plain))
print("Bob applies H first:    ", with_h, "  Alice's zeros:", alice_zeros(with_h))
print("correlations:", round(correlation(plain), 2), "and", round(correlation(with_h), 2))"""),
md("""Alice gets 0 about half the time either way: nothing Bob does to his qubit changes Alice's own counts, so entanglement cannot send a message. The **correlation** does change (from +1 to about 0), but Alice and Bob can only see that by comparing their lists of results afterwards, over an ordinary channel."""),
md("""## Step 7 (guided): which states are entangled?

A two-qubit state is a **product state** if it can be written as `product_state(q0, q1)` for some q0 and q1. Otherwise it is **entangled**. The prepared function below tests this. Arrange the four amplitudes in a 2 × 2 table, [[a00, a01], [a10, a11]]. For a product state the table is c·[a, b] in the first row and d·[a, b] in the second, so a00·a11 − a01·a10 = 0. The function also returns a number between 0 (product) and 1 (as entangled as possible), the **concurrence** 2|a00·a11 − a01·a10|.

Run the cell, then compare the list with your expectations. Last, look at each qubit's own Bloch vector: for a product state both arrows have length 1, and for a Bell state both have length 0. Neither qubit has a state of its own."""),
code("""def is_product(state, tol=1e-9):
    \"\"\"Prepared for you: True if the 2-qubit state can be written as product_state(q0, q1).\"\"\"
    t = np.asarray(state, dtype=complex).reshape(2, 2)    # rows: qubit 1, columns: qubit 0
    return abs(t[0, 0] * t[1, 1] - t[0, 1] * t[1, 0]) < tol


def concurrence(state):
    t = np.asarray(state, dtype=complex).reshape(2, 2)
    return 2 * abs(t[0, 0] * t[1, 1] - t[0, 1] * t[1, 0])


examples = {
    "product of [0.6, 0.8] and |+>": product_state(np.array([0.6, 0.8]), np.array([1, 1]) / np.sqrt(2)),
    "(|00> + |01> + |10> + |11>)/2": np.array([0.5, 0.5, 0.5, 0.5]),
    "phi+": Statevector(bell_circuit("phi+")).data,
    "psi-": Statevector(bell_circuit("psi-")).data,
    "cos(30)|00> + sin(30)|11>": np.array([np.cos(np.pi / 6), 0, 0, np.sin(np.pi / 6)]),
    "(|00> + |01> + |10> - |11>)/2": np.array([0.5, 0.5, 0.5, -0.5]),
}
for label, st in examples.items():
    lengths = [round(float(np.linalg.norm(v)), 3) for v in bloch_vectors(Statevector(st))]
    print(f"{label:32s} product: {str(is_product(st)):5s}  concurrence {concurrence(st):.3f}  Bloch-vector lengths {lengths}")

display(plot_bloch_multivector(Statevector(bell_circuit("phi+")), title="phi+: each qubit alone"))"""),
md("""The state (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2 looks complicated but is |+⟩ ⊗ |+⟩, a product state. Changing one sign gives (|00⟩ + |01⟩ + |10⟩ − |11⟩)/2, which is as entangled as a Bell state. The state cos 30°|00⟩ + sin 30°|11⟩ is entangled, but only partly (concurrence 0.866), and its qubits' Bloch vectors are shorter than 1 but longer than 0."""),
md("""## Step 8: your personal circuit

Open the Canvas quiz **Module 4 lab check**. Question 1 gives you three numbers: **THETA_DEG**, **PHI_DEG** (both in degrees) and **SEED**. Type them below in place of `None` and run the cell. Your circuit is Ry(THETA) on qubit 0, CNOT from qubit 0 to qubit 1, Rz(PHI) on qubit 1, then H on both qubits and a measurement of both: a partly entangled state measured in the X basis."""),
code("""THETA_DEG = None   # for example: THETA_DEG = 73
PHI_DEG = None     # for example: PHI_DEG = 141
SEED = None        # for example: SEED = 512

if None in (THETA_DEG, PHI_DEG, SEED):
    print("Enter THETA_DEG, PHI_DEG and SEED from Canvas first.")
else:
    qc = QuantumCircuit(2)
    qc.ry(np.radians(THETA_DEG), 0)
    qc.cx(0, 1)
    st = Statevector(qc).data
    print("state after Ry and CNOT:", np.round(st, 4) + 0)
    print("concurrence:", round(concurrence(st), 3))"""),
md("""## Step 9: run the check cell

**Do not edit this cell.** Run it. It tests your three functions on cases you have not seen. If they all pass, it runs your personal circuit 1,000 times with your SEED and prints your **verification value**: the number of shots in which the two qubits gave the **same** result. Type it into Canvas question 1.

If a test does not pass, read its message, fix that function, run its cell again, then run this cell again."""),
code("# CHECK CELL: do not edit\n" + CHECK + '''

ok, messages, value = check_module4(product_state, bell_circuit, correlation, THETA_DEG, PHI_DEG, SEED)
for m in messages:
    print(m)
if ok:
    print(f"\\nAll tests passed. Your verification value for THETA_DEG = {THETA_DEG}, PHI_DEG = {PHI_DEG}, SEED = {SEED}: {value}")
    print("Type this number into Canvas question 1.")
else:
    print("\\nNot passed yet: no verification value.")
'''),
md("""## Finish

1. Answer the five questions of the **Module 4 lab check** in Canvas and submit.
2. Take the **Module 4 quiz**.
3. Fill in the short **Module 4 time log**.

Your notebook is saved in this browser as you work. It is not saved anywhere else, so to keep a copy choose **File > Download**. The final coding task starts from your Bell circuit, H on qubit 0 and CNOT from qubit 0 to qubit 1, and adds a third qubit. In Module 5 you run the Bell circuit on a simulator with noise and, when available, on a real quantum computer."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl1m4{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / NAME)
print("wrote", OUT / NAME)
