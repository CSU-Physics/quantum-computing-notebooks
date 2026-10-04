"""Build the Level 1 Module 2 browser notebook (JupyterLite, Pyodide kernel) from checks/m2_check.py."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level1"
CHECK = (ROOT / "checks" / "m2_check.py").read_text().split('"""', 2)[2].lstrip()
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md("""# Module 2 lab: the Bloch sphere and measurement bases

**Quantum Computing Foundations · Module 2 · about 75 minutes** · notebook version 2026-10-04

In this lab you write three functions:

1. `state_from_angles(theta, phi)`: the qubit state at the point (θ, φ) of the Bloch sphere;
2. `bloch_vector(state)`: the point (x, y, z) of the Bloch sphere for a state;
3. `measure_in_basis(qc, basis)`: a circuit that measures in the Z, X or Y basis.

Then you compare predicted and sampled counts in three bases and use them to find a state's Bloch vector from measurements alone. The **Module 2 lab check** asks for results from this notebook, so keep it open in a second tab. Run each code cell with **Shift + Enter**, in order."""),
md("""## Step 0: start Python

Run the cell below. The first time, Python can take up to a minute to start in your browser. `qsim` is the course's simulator; it uses the same names as Qiskit."""),
code("""import numpy as np
import matplotlib.pyplot as plt   # used by plot_bloch_vector
from qsim import QuantumCircuit, Statevector, AerSimulator, plot_bloch_vector

print("Ready. NumPy version", np.__version__)"""),
md("""## Step 1: Dirac notation in code

A **ket** |ψ⟩ is a column of amplitudes, here a NumPy array. Its **bra** ⟨ψ| is the same list with every amplitude replaced by its complex conjugate. The **inner product** ⟨φ|ψ⟩ multiplies the conjugated amplitudes of φ by the amplitudes of ψ and adds them up; `np.vdot(phi, psi)` does exactly that.

The amplitude of a basis state |k⟩ in |ψ⟩ is ⟨k|ψ⟩, and the probability of finding |k⟩ is |⟨k|ψ⟩|². Run the cell."""),
code("""ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)
plus = (ket0 + ket1) / np.sqrt(2)
minus = (ket0 - ket1) / np.sqrt(2)
psi = np.array([0.6, 0.8j])

print("<0|1> =", np.vdot(ket0, ket1))          # orthogonal: 0
print("<+|+> =", np.round(np.vdot(plus, plus), 6))   # normalized: 1
print("<+|-> =", np.round(np.vdot(plus, minus), 6))  # orthogonal: 0
amp = np.vdot(plus, psi)
print("<+|psi> =", np.round(amp, 4), "  probability |<+|psi>|^2 =", np.round(abs(amp) ** 2, 4))"""),
md("""The state [0.6, 0.8i] has probability 0.36 of giving 0 in the usual (Z) basis, but probability 0.5 of being found as |+⟩. Which probabilities you see depends on the basis you measure in. That is the subject of this lab."""),
md("""## Step 2: a state from two angles

Every qubit state can be written, up to a global phase, as

|ψ⟩ = cos(θ/2) |0⟩ + e^{iφ} sin(θ/2) |1⟩,

where θ (between 0 and π) and φ (between 0 and 2π) are the angles of a point on the **Bloch sphere**. In NumPy, e^{iφ} is `np.exp(1j * phi)`. The angles are in radians.

Complete `state_from_angles()`, then run the cell. It builds the state with θ = π/2, φ = π/2 and compares it with a circuit that makes the same state: `ry(θ)` sets θ, and the phase gate `p(φ)` sets φ."""),
code("""def state_from_angles(theta, phi):
    # YOUR CODE: replace the next line. Return np.array([amplitude of |0>, amplitude of |1>]).
    raise NotImplementedError("Complete state_from_angles() first.")


mine = state_from_angles(np.pi / 2, np.pi / 2)
print("state_from_angles(pi/2, pi/2) =", np.round(mine, 4))

qc = QuantumCircuit(1)
qc.ry(np.pi / 2, 0)
qc.p(np.pi / 2, 0)
print("circuit ry(pi/2), p(pi/2)     =", Statevector(qc))"""),
md("""Both should be [0.7071, 0.7071i]: the state |+i⟩, on the +y axis of the Bloch sphere."""),
md("""## Step 3: from a state to its Bloch vector

The point (x, y, z) of a state [a, b] on the Bloch sphere is

- x = 2 Re(a* b)
- y = 2 Im(a* b)
- z = |a|² − |b|²

where a* is the complex conjugate of a (`np.conj(a)`), Re is the real part (`.real`) and Im the imaginary part (`.imag`). Complete `bloch_vector()` so it returns the three real numbers as an array, then run the cell. Canvas asks for the y value of [0.6, 0.8i]."""),
code("""def bloch_vector(state):
    # YOUR CODE: replace the next line. Return np.array([x, y, z]) with three real numbers.
    raise NotImplementedError("Complete bloch_vector() first.")


b = bloch_vector(np.array([0.6, 0.8j]))
print("Bloch vector of [0.6, 0.8i]:", np.round(b, 4))
plot_bloch_vector(b, title="[0.6, 0.8i]")"""),
md("""You should see (0, 0.96, −0.28): the arrow points almost straight along +y and a little below the equator, because P(1) = 0.64 is a little more than one half."""),
md("""## Step 4: global phase and relative phase

Multiplying the whole state by e^{iγ} is a **global phase**. It changes the amplitudes but no probability in any basis, so it is the same physical state. Changing φ is a **relative phase**: it changes the phase of |1⟩ compared with |0⟩, and that moves the point on the sphere.

Run the cell and compare the three Bloch vectors."""),
code("""psi = state_from_angles(1.2, 0.5)
global_phase = np.exp(1j * 0.9) * psi           # the same state times e^(0.9 i)
relative_phase = state_from_angles(1.2, 0.5 + 0.9)  # phi changed by 0.9

print("psi                :", np.round(bloch_vector(psi), 4))
print("e^(0.9i) * psi     :", np.round(bloch_vector(global_phase), 4), "  <- the same point")
print("phi + 0.9 (relative):", np.round(bloch_vector(relative_phase), 4), "  <- a different point")
print("P(0) in the Z basis:", round(abs(psi[0])**2, 4), round(abs(global_phase[0])**2, 4), round(abs(relative_phase[0])**2, 4))"""),
md("""All three have the same Z-basis probabilities, so a Z measurement cannot tell them apart. The relative phase still changes the state, and measuring in another basis shows it. That is the next step."""),
md("""## Step 5: measuring in another basis

A quantum computer measures only in the Z basis (|0⟩ and |1⟩). To measure in another basis, you first **rotate** the state so that the basis you want lines up with Z, then measure:

| Basis | Basis states | Add before `measure` | Result 0 means |
|---|---|---|---|
| Z | \\|0⟩, \\|1⟩ | nothing | \\|0⟩ |
| X | \\|+⟩, \\|−⟩ | `h` | \\|+⟩ |
| Y | \\|+i⟩, \\|−i⟩ | `sdg`, then `h` | \\|+i⟩ |

Complete `measure_in_basis()`. It must **not** change the circuit it is given: start with `new = qc.copy()`, add the gates for the basis, add `new.measure(0, 0)`, and return `new`. Then run the cell."""),
code("""def measure_in_basis(qc, basis):
    # YOUR CODE: replace the next line. basis is "Z", "X" or "Y".
    raise NotImplementedError("Complete measure_in_basis() first.")


prep = QuantumCircuit(1, 1)
prep.h(0)                     # the state |+>
for basis in ["Z", "X"]:
    counts = AerSimulator(seed_simulator=1).run(measure_in_basis(prep, basis), shots=1000).result().get_counts()
    print(basis, "basis:", counts)"""),
md("""For |+⟩, the Z basis gives about 500 of each result, but the X basis gives 0 every time: in its own basis, |+⟩ is certain."""),
md("""## Step 6: predicted and sampled counts in three bases

The probability of reading 0 in each basis comes straight from the Bloch vector:

P(0 in Z) = (1 + z)/2  P(0 in X) = (1 + x)/2  P(0 in Y) = (1 + y)/2

Run the cell. It prepares the state with θ = 2π/3 and φ = π/3, predicts the three probabilities, then measures 1,000 shots in each basis with seed 7. Canvas asks for the number of 0 results in the **X** basis."""),
code("""theta, phi = 2 * np.pi / 3, np.pi / 3
prep = QuantumCircuit(1, 1)
prep.ry(theta, 0)
prep.p(phi, 0)

x, y, z = bloch_vector(state_from_angles(theta, phi))
predicted = {"Z": (1 + z) / 2, "X": (1 + x) / 2, "Y": (1 + y) / 2}

zeros = {}
print("basis  predicted P(0)  sampled 0s of 1000")
for basis in ["Z", "X", "Y"]:
    counts = AerSimulator(seed_simulator=7).run(measure_in_basis(prep, basis), shots=1000).result().get_counts()
    zeros[basis] = counts.get("0", 0)
    print(f"  {basis}        {predicted[basis]:.3f}            {zeros[basis]}")"""),
md("""Each sampled count lands within a few spreads √(N·p·(1 − p)) of 1000 × P(0), the shot-noise rule from Module 1."""),
md("""## Step 7: find the Bloch vector from counts alone

Turn the rule around: from the fraction of 0 results f in each basis, estimate the coordinate as 2f − 1. Measuring in three bases is the simplest form of **state tomography**: it finds an unknown state from measurement results. Run the cell."""),
code("""estimate = np.array([2 * zeros["X"] / 1000 - 1, 2 * zeros["Y"] / 1000 - 1, 2 * zeros["Z"] / 1000 - 1])
true = np.array([x, y, z])
print("estimated (x, y, z):", np.round(estimate, 3))
print("true      (x, y, z):", np.round(true, 3))
print("distance between them:", round(float(np.linalg.norm(estimate - true)), 3))
plot_bloch_vector(estimate, title="estimated from 3 x 1,000 shots")"""),
md("""With 1,000 shots per basis, each coordinate is off by about 0.03 or less, so the estimated arrow is close to the true one. More shots give a better estimate, slowly, as in Module 1."""),
md("""## Step 8: your personal state

Open the Canvas quiz **Module 2 lab check**. Question 1 gives you three numbers: **THETA_DEG**, **PHI_DEG** (both in degrees) and **SEED**. Type them below in place of `None` and run the cell. It converts the angles to radians and shows your state."""),
code("""THETA_DEG = None   # for example: THETA_DEG = 73
PHI_DEG = None     # for example: PHI_DEG = 141
SEED = None        # for example: SEED = 512

if None in (THETA_DEG, PHI_DEG, SEED):
    print("Enter THETA_DEG, PHI_DEG and SEED from Canvas first.")
else:
    my_state = state_from_angles(np.radians(THETA_DEG), np.radians(PHI_DEG))
    print("your state:", np.round(my_state, 4))
    print("its Bloch vector:", np.round(bloch_vector(my_state), 4))
    display(plot_bloch_vector(bloch_vector(my_state), title="your state"))"""),
md("""## Step 9: run the check cell

**Do not edit this cell.** Run it. It tests your three functions on cases you have not seen. If they all pass, it measures your personal state 1,000 times in the **Y** basis with your SEED and prints your **verification value**: the number of 0 results. Type it into Canvas question 1.

If a test does not pass, read its message, fix that function, run its cell again, then run this cell again."""),
code("# CHECK CELL: do not edit\n" + CHECK + '''

ok, messages, value = check_module2(state_from_angles, bloch_vector, measure_in_basis, THETA_DEG, PHI_DEG, SEED)
for m in messages:
    print(m)
if ok:
    print(f"\\nAll tests passed. Your verification value for THETA_DEG = {THETA_DEG}, PHI_DEG = {PHI_DEG}, SEED = {SEED}: {value}")
    print("Type this number into Canvas question 1.")
else:
    print("\\nNot passed yet: no verification value.")
'''),
md("""## Finish

1. Answer the five questions of the **Module 2 lab check** in Canvas and submit.
2. Take the **Module 2 quiz**.
3. Fill in the short **Module 2 time log**.

Your notebook is saved in this browser as you work. It is not saved anywhere else, so to keep a copy choose **File > Download**. You use `measure_in_basis()` again in Module 3, where gates rotate states around the Bloch sphere."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl1m2{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / "QC-L1-M2-lab-bloch-sphere.ipynb")
print("wrote", OUT / "QC-L1-M2-lab-bloch-sphere.ipynb")
