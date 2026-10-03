"""Build the Level 1 Module 3 browser notebook (JupyterLite, Pyodide kernel) from checks/m3_check.py."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level1"
CHECK = (ROOT / "checks" / "m3_check.py").read_text().split('"""', 2)[2].lstrip()
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}
NAME = "QC-L1-M3-lab-single-qubit-gates.ipynb"


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md("""# Module 3 lab: single-qubit gates

**Quantum Computing Foundations · Module 3 · about 85 minutes**

A gate is a 2 × 2 matrix, and running a circuit multiplies matrices. In this lab you write three functions:

1. `gate_matrix(name)`: the matrix of X, Y, Z, H, S or T;
2. `rotation(axis, theta)`: the rotation gates Rx, Ry and Rz;
3. `sequence_matrix(matrices)`: the single matrix of several gates applied one after another.

You check each one against `Operator`, which computes a circuit's matrix the way Qiskit does. Then you show that gates can be undone, watch interference build up as a phase changes, and fix a circuit that has one wrong gate. The **Module 3 lab check** asks for results from this notebook, so keep it open in a second tab. Run each code cell with **Shift + Enter**, in order."""),
md("""## Step 0: start Python

Run the cell below. The first time, Python can take up to a minute to start in your browser. `qsim` is the course's simulator; it uses the same names as Qiskit, and `Operator` is new in this module."""),
code("""import numpy as np
import matplotlib.pyplot as plt   # used for the plots below
from qsim import QuantumCircuit, Statevector, Operator, AerSimulator, plot_bloch_vector

print("Ready. NumPy version", np.__version__)"""),
md("""## Step 1: a gate is a matrix

A one-qubit state is a column of two amplitudes, and a one-qubit gate is a 2 × 2 matrix. Applying the gate is a matrix product: new state = matrix @ state. In NumPy, `@` multiplies matrices.

`Operator(qc)` gives the matrix of a whole circuit. Run the cell: the X gate swaps the two amplitudes, so it turns |0⟩ into |1⟩."""),
code("""X = np.array([[0, 1], [1, 0]], dtype=complex)
ket0 = np.array([1, 0], dtype=complex)
print("X @ |0> =", X @ ket0)

qc = QuantumCircuit(1)
qc.x(0)
print(Operator(qc))"""),
md("""## Step 2: the matrices of six gates

Complete `gate_matrix(name)` so it returns the matrix for each name below, as a NumPy array. Write the matrices yourself; do not use `Operator` inside the function.

| Name | Matrix | What it does to the Bloch sphere |
|---|---|---|
| X | [[0, 1], [1, 0]] | half turn about the x axis |
| Y | [[0, −i], [i, 0]] | half turn about the y axis |
| Z | [[1, 0], [0, −1]] | half turn about the z axis |
| H | [[1, 1], [1, −1]] / √2 | half turn about the axis halfway between x and z |
| S | [[1, 0], [0, i]] | quarter turn about the z axis |
| T | [[1, 0], [0, e^{iπ/4}]] | eighth turn about the z axis |

In Python, i is `1j`, √2 is `np.sqrt(2)` and e^{iπ/4} is `np.exp(1j * np.pi / 4)`. Then run the cell: it compares each of your matrices with `Operator` of a one-gate circuit."""),
code("""def gate_matrix(name):
    # YOUR CODE: replace the next line. name is "X", "Y", "Z", "H", "S" or "T".
    raise NotImplementedError("Complete gate_matrix() first.")


for name in ["X", "Y", "Z", "H", "S", "T"]:
    qc = QuantumCircuit(1)
    getattr(qc, name.lower())(0)          # for example qc.h(0) when name is "H"
    same = Operator(gate_matrix(name)) == Operator(qc)
    print(f"{name}: matches Operator: {same}")"""),
md("""All six lines should say `True`. If one says `False`, print `Operator(qc)` for that gate and compare it with your matrix entry by entry."""),
md("""## Step 3: every gate can be undone

A gate matrix U is **unitary**: U† U = I, where U† (U-dagger) is the conjugate transpose, `U.conj().T` in NumPy. So U† undoes U, and no information is lost. A measurement is different: it cannot be undone.

Some gates are their own inverse. Run the cell: it applies H twice to the state [0.6, 0.8i] and gets the same state back. Then it checks how the phase gates build on each other."""),
code("""H, S, T, Z = (np.asarray(gate_matrix(n), dtype=complex) for n in ["H", "S", "T", "Z"])
I = np.eye(2)

for name in ["X", "Y", "Z", "H", "S", "T"]:
    U = np.asarray(gate_matrix(name), dtype=complex)
    print(f"{name}: U-dagger U = I is {np.allclose(U.conj().T @ U, I)}")

psi = np.array([0.6, 0.8j])
print("\\npsi          =", psi)
print("H @ psi      =", np.round(H @ psi, 4))
print("H @ H @ psi  =", np.round(H @ H @ psi, 4), " <- back to the start")

print("\\nS @ S equals Z:", np.allclose(S @ S, Z))
print("T @ T equals S:", np.allclose(T @ T, S))
U, n = T.copy(), 1
while not np.allclose(U, I) and n < 16:      # stop after 16 so a wrong T matrix cannot loop for ever
    U, n = T @ U, n + 1
if np.allclose(U, I):
    print("T applied", n, "times in a row is the identity")
else:
    print("No power of T up to 16 is the identity. Check your T matrix in gate_matrix(): "
          "it should be [[1, 0], [0, exp(i pi/4)]].")"""),
md("""H takes |0⟩ to |+⟩ and |+⟩ back to |0⟩, so H twice does nothing. On the Bloch sphere, S is a quarter turn and T an eighth turn about the z axis, so two T gates make an S, two S gates make a Z, and eight T gates make a full turn. Canvas asks how many T gates in a row make the identity."""),
md("""## Step 4: rotation gates

The rotation gates turn the Bloch-sphere arrow by an angle θ about the x, y or z axis:

R(θ) = cos(θ/2) I − i sin(θ/2) P,

where I is the identity and P is the Pauli matrix X, Y or Z for the axis. The matrix uses **θ/2**, half the angle on the sphere, for the same reason as the cos(θ/2) in Module 2.

Complete `rotation(axis, theta)`. You can use your `gate_matrix()` for the Pauli matrix. Then run the cell: it compares your matrices with `Operator` of `qc.rx`, `qc.ry` and `qc.rz`, then compares Rz(π) with Z in two ways."""),
code("""def rotation(axis, theta):
    # YOUR CODE: replace the next line. axis is "X", "Y" or "Z"; theta is in radians.
    raise NotImplementedError("Complete rotation() first.")


for axis in ["X", "Y", "Z"]:
    qc = QuantumCircuit(1)
    getattr(qc, "r" + axis.lower())(0.8, 0)    # qc.rx(0.8, 0), qc.ry(0.8, 0), qc.rz(0.8, 0)
    print(f"R{axis.lower()}(0.8) matches Operator: {Operator(rotation(axis, 0.8)) == Operator(qc)}")

print("\\nRy(pi/2) |0> =", np.round(rotation("Y", np.pi / 2) @ ket0, 4), " <- the state |+>")

rz_pi = Operator(rotation("Z", np.pi))
print("\\nRz(pi) =")
print(np.round(rz_pi.data, 4) + 0)
print("Rz(pi) == Z:      ", rz_pi == Operator(gate_matrix("Z")))
print("Rz(pi).equiv(Z):  ", rz_pi.equiv(Operator(gate_matrix("Z"))))"""),
md("""`==` asks whether two matrices are exactly equal; `equiv` asks whether they are equal up to a **global phase**. Rz(π) is −i times Z: a different matrix, but the same gate as far as any measurement can tell, because a global phase changes no probability (Module 2). The same is true of Rx(π) and X, and of Ry(π) and Y."""),
md("""## Step 5: gates in sequence

When gates run one after another, their matrices multiply. The order is the opposite of the circuit diagram: in the circuit H then S, H acts first, so the matrix is **S @ H**, with the first gate on the right.

Complete `sequence_matrix(matrices)`. It gets a list of 2 × 2 matrices in **time order** (first gate first) and returns the single matrix of the whole sequence. Start with the identity and, for each gate G in the list, set U = G @ U. With an empty list it should return the identity. Then run the cell."""),
code("""def sequence_matrix(matrices):
    # YOUR CODE: replace the next line. Return the 2 x 2 matrix of all the gates, first gate applied first.
    raise NotImplementedError("Complete sequence_matrix() first.")


for names in (["H", "S"], ["S", "H"], ["H", "Z", "H"], ["H", "X", "H"]):
    qc = QuantumCircuit(1)
    for n in names:
        getattr(qc, n.lower())(0)
    U = sequence_matrix([gate_matrix(n) for n in names])
    print(" then ".join(names), "  matches Operator:", Operator(U) == Operator(qc))
    print(np.round(np.asarray(U), 4) + 0, "\\n")"""),
md("""H then S and S then H give different matrices: for gates, order matters. H then Z then H is the X gate, and H then X then H is the Z gate. H swaps the roles of the x and z axes."""),
md("""### Why `measure_in_basis()` works

In Module 2 you wrote `measure_in_basis(qc, "Y")`, which adds S† then H before the measurement. Now you can see why. The matrix of "S† then H" turns |+i⟩ into |0⟩ (and |−i⟩ into |1⟩), so a Z measurement after these two gates is a Y measurement before them. Your Module 2 function is copied below. Run the cell."""),
code("""def measure_in_basis(qc, basis):          # your function from Module 2
    new = qc.copy()
    if basis == "X":
        new.h(0)
    elif basis == "Y":
        new.sdg(0)
        new.h(0)
    new.measure(0, 0)
    return new


Sdg = np.array([[1, 0], [0, -1j]])
plus_i = np.array([1, 1j]) / np.sqrt(2)
U = np.asarray(sequence_matrix([Sdg, gate_matrix("H")]))
print("(S-dagger then H) |+i> =", np.round(U @ plus_i, 4) + 0, " <- the state |0>")

prep = QuantumCircuit(1, 1)
prep.h(0)
prep.s(0)                                 # H then S makes |+i>
counts = AerSimulator(seed_simulator=3).run(measure_in_basis(prep, "Y"), shots=1000).result().get_counts()
print("|+i> measured in the Y basis:", counts)"""),
md("""## Step 6: interference

Put a phase between two H gates: H, then the phase gate P(φ), then H. The first H splits |0⟩ into equal amplitudes for 0 and 1. The phase changes the |1⟩ part. The second H recombines them, and the two paths to each result **interfere**:

P(0) = cos²(φ/2).

With φ = 0 the paths add up and the result is always 0. With φ = π they cancel for 0 and the result is always 1. In between, P(0) follows a smooth curve, the same fringe pattern as light in an interferometer. Run the cell: it measures the circuit 1,000 times with seed 7 at 13 values of φ and plots the fraction of 0 results against the prediction. Canvas asks for the count at φ = 2π/3."""),
code("""def interference_circuit(phi):
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.p(phi, 0)
    qc.h(0)
    qc.measure(0, 0)
    return qc


phis = np.linspace(0, 2 * np.pi, 13)
zeros = [AerSimulator(seed_simulator=7).run(interference_circuit(p), shots=1000).result().get_counts().get("0", 0)
         for p in phis]

fig, ax = plt.subplots(figsize=(6.5, 3.2))
grid = np.linspace(0, 2 * np.pi, 200)
ax.plot(grid, np.cos(grid / 2) ** 2, color="#24313D", label="predicted cos²(φ/2)")
ax.plot(phis, np.array(zeros) / 1000, "o", color="#99004C", label="measured, 1,000 shots")
ax.set_xlabel("phase φ (radians)")
ax.set_ylabel("fraction of 0 results")
ax.legend(loc="upper center")
fig.tight_layout()
plt.close(fig)
display(fig)

k = 4                                   # phis[4] is 2*pi/3
print(f"phi = 2*pi/3: predicted {1000 * np.cos(phis[k] / 2) ** 2:.0f} zeros, measured {zeros[k]} zeros")"""),
md("""Without the phase, H then H is the identity (Step 3). The phase gate alone does not change P(0) either. Only the combination shows the phase, because the second H makes the two paths interfere. Interference is what quantum algorithms use: they arrange the phases so that wrong answers cancel and right answers add up."""),
md("""## Step 7: fix the circuit with one wrong gate

The circuit below runs four gates, H, T, S, H, and then tries to **undo** them, so the qubit should end in |0⟩ and every shot should give 0. To undo a sequence, you run the inverse gates in **reverse order**: the inverse of S is S† (`sdg`), the inverse of T is T† (`tdg`), and H is its own inverse.

Run the cell. The counts are wrong: one gate in the undo part is wrong. Use `Operator` to look at the whole circuit's matrix, find the wrong line, fix it, and run the cell again until every shot gives 0."""),
code("""def round_trip():
    qc = QuantumCircuit(1, 1)
    # forward: H, T, S, H
    qc.h(0)
    qc.t(0)
    qc.s(0)
    qc.h(0)
    # undo: should bring the qubit back to |0>
    qc.h(0)
    qc.sdg(0)
    qc.t(0)
    qc.h(0)
    qc.measure(0, 0)
    return qc


counts = AerSimulator(seed_simulator=11).run(round_trip(), shots=1000).result().get_counts()
print("counts:", counts, "  (should be {'0': 1000})")

body = round_trip()
body.remove_final_measurements()
print("matrix of the whole circuit:")
print(np.round(Operator(body).data, 4) + 0)
print("same as doing nothing (up to a global phase):", Operator(body).equiv(np.eye(2)))"""),
md("""When the circuit is right, the matrix of the whole circuit is the identity and every shot gives 0. Canvas asks which change fixes it."""),
md("""## Step 8: your personal circuit

Open the Canvas quiz **Module 3 lab check**. Question 1 gives you three numbers: **THETA_DEG**, **PHI_DEG** (both in degrees) and **SEED**. Type them below in place of `None` and run the cell. Your circuit is Ry(THETA) followed by Rx(PHI), then a measurement."""),
code("""THETA_DEG = None   # for example: THETA_DEG = 73
PHI_DEG = None     # for example: PHI_DEG = 141
SEED = None        # for example: SEED = 512

if None in (THETA_DEG, PHI_DEG, SEED):
    print("Enter THETA_DEG, PHI_DEG and SEED from Canvas first.")
else:
    U = np.asarray(sequence_matrix([rotation("Y", np.radians(THETA_DEG)), rotation("X", np.radians(PHI_DEG))]))
    final = U @ np.array([1, 0], dtype=complex)
    print("your final state:", np.round(final, 4))
    print("P(0) =", round(abs(final[0]) ** 2, 4))
    a, b = final
    display(plot_bloch_vector([2 * (np.conj(a) * b).real, 2 * (np.conj(a) * b).imag, abs(a) ** 2 - abs(b) ** 2],
                              title="your final state"))"""),
md("""## Step 9: run the check cell

**Do not edit this cell.** Run it. It tests your three functions on cases you have not seen, and your repaired `round_trip()` from Step 7. If they all pass, it runs your personal circuit 1,000 times with your SEED and prints your **verification value**: the number of 0 results. Type it into Canvas question 1.

If a test does not pass, read its message, fix that function, run its cell again, then run this cell again."""),
code("# CHECK CELL: do not edit\n" + CHECK + '''

ok, messages, value = check_module3(gate_matrix, rotation, sequence_matrix, THETA_DEG, PHI_DEG, SEED, round_trip)
for m in messages:
    print(m)
if ok:
    print(f"\\nAll tests passed. Your verification value for THETA_DEG = {THETA_DEG}, PHI_DEG = {PHI_DEG}, SEED = {SEED}: {value}")
    print("Type this number into Canvas question 1.")
else:
    print("\\nNot passed yet: no verification value.")
'''),
md("""## Finish

1. Answer the five questions of the **Module 3 lab check** in Canvas and submit.
2. Take the **Module 3 quiz**.
3. Fill in the short **Module 3 time log**.

Your notebook is saved in this browser as you work. It is not saved anywhere else, so to keep a copy choose **File > Download**. In Module 4 the same matrices act on two qubits, and the CNOT gate makes them entangled."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl1m3{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / NAME)
print("wrote", OUT / NAME)
