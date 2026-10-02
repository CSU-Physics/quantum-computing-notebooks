"""Build the Level 1 browser notebooks (JupyterLite, Pyodide kernel) from the check files."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level1"


def body(path):
    return (ROOT / "checks" / path).read_text().split('"""', 2)[2].lstrip()


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}

STEP0 = """## Step 0: start Python

This lab runs entirely in your web browser: there is nothing to install and no account to create. Run the cell below with **Shift + Enter**. The first time, Python can take up to a minute to start in your browser; after that it is quick.

The labs use **qsim**, a small simulator written for this course. It uses the same names as **Qiskit**, the quantum programming toolkit used in industry and research (`QuantumCircuit`, `h`, `cx`, `measure`, `AerSimulator` and so on), so the code you write here also works in Qiskit. In Module 5 you run the same kind of code with Qiskit itself."""

STEP0_CODE = """import numpy as np
import qsim

print("Python is running in your browser. qsim version", qsim.__version__)"""

SAVE = """Your notebook is saved in this browser as you work. It is not saved anywhere else, so to keep a copy choose **File > Download**. If you open the lab in a different browser or on another computer, you start from a fresh copy."""

# ------------------------------------------------------------------ Module 0
m0 = [
md("""# Module 0 lab: notebooks, NumPy and your first qubit

**Quantum Computing Foundations · Module 0 · about 40 minutes**

In this lab you will:

1. run and change cells in a notebook;
2. refresh the NumPy and complex-number tools the course uses;
3. build, draw and run your first one-qubit circuit;
4. find and fix a bug;
5. rotate a qubit by your own angle and get a verification value for Canvas.

The **Module 0 lab check** in Canvas asks for results from this notebook. Keep both open side by side. Run each code cell with **Shift + Enter**, in order."""),
md(STEP0),
code(STEP0_CODE),
md("""## Step 1: how a notebook works

A notebook mixes text cells like this one with **code cells**. Running a code cell prints its result underneath. Later cells can use what earlier cells created, so run them in order.

**Try it:** change `"Ada"` to your own first name, then run the cell again."""),
code("""name = "Ada"
print(f"Hello, {name}! Your notebook is working.")"""),
md("""## Step 2: NumPy and complex numbers

Quantum states are written with **complex numbers**, numbers of the form a + bi. Python writes i as `j`, so 0.6 + 0.8i is `0.6 + 0.8j`.

Three tools you will use in every module:

| Tool | Python | Example |
|---|---|---|
| Size squared, \\|z\\|² | `abs(z)**2` | \\|0.6 + 0.8i\\|² = 0.36 + 0.64 = 1 |
| Complex conjugate, z* | `np.conj(z)` | (0.6 + 0.8i)* = 0.6 − 0.8i |
| A vector of amplitudes | `np.array([a, b])` | a state of one qubit |

Run the cell and compare its output with the table."""),
code("""z = 0.6 + 0.8j
print("z =", z)
print("|z|^2 =", round(abs(z)**2, 6))
print("conjugate of z =", np.conj(z))"""),
md("""A one-qubit state is a vector of two amplitudes: the first for |0⟩, the second for |1⟩. **The probability of each measurement result is the size squared of its amplitude**, and the probabilities add up to 1.

The state below has amplitude 0.6 for |0⟩ and 0.8i for |1⟩. Run the cell. Canvas asks for the probability of measuring 1."""),
code("""psi = np.array([0.6, 0.8j])

probabilities = np.abs(psi)**2
print("P(0) =", round(probabilities[0], 3))
print("P(1) =", round(probabilities[1], 3))
print("Total =", round(probabilities.sum(), 3))"""),
md("""## Step 3: your first quantum circuit

A **circuit** is a list of operations on qubits. This one has one qubit and one classical bit to store the measurement:

- `qc.h(0)` applies the **Hadamard gate**, which puts the qubit into an equal superposition of 0 and 1;
- `qc.measure(0, 0)` measures qubit 0 into classical bit 0.

Run the cell to draw the circuit, then run the next cell to measure it 1,000 times on the simulator."""),
code("""from qsim import QuantumCircuit

qc = QuantumCircuit(1, 1)
qc.h(0)
qc.measure(0, 0)
qc.draw()"""),
code("""from qsim import transpile, AerSimulator

sim = AerSimulator()
counts = sim.run(transpile(qc, sim), shots=1000).result().get_counts()
print(counts)"""),
md("""You should see about 500 of each result. The exact numbers change each run: measurement results are random.

> **The same code in Qiskit.** With Qiskit installed, the import lines would be `from qiskit import QuantumCircuit, transpile` and `from qiskit_aer import AerSimulator`. Everything else in these two cells stays the same.

**Change one thing:** in the circuit cell above, add a second line `qc.h(0)` directly after the first, so the Hadamard gate is applied **twice**. Run the circuit cell and the simulator cell again. Canvas asks how many times out of 1,000 you get the result 1. (You will see why in Module 3.)"""),
md("""## Step 4: find and fix the bug

The cell below has one bug. Run it and read the error message at the bottom. Then fix the line so the cell runs. Canvas asks which change fixes it."""),
code("""qc_bug = QuantumCircuit(1, 1)
qc_bug.h(0)
qc_bug.measure(0, 1)
print(sim.run(transpile(qc_bug, sim), shots=100).result().get_counts())"""),
md("""## Step 5: rotate the qubit by your own angle

Open the Canvas quiz **Module 0 lab check**. Question 1 shows your personal angle θ in radians, for example `θ = 1.25`. Type it below in place of `None` and run the cell."""),
code("""THETA = None   # for example: THETA = 1.25

print("Your angle:", THETA)"""),
md("""The gate `qc.ry(angle, 0)` rotates the qubit by an angle around the y-axis of the Bloch sphere (Module 2). Complete `build_rotation()` so it applies `ry` with **your** `THETA` to qubit 0, then measures it. Run the cell."""),
code("""def build_rotation():
    qc = QuantumCircuit(1, 1)

    # YOUR CODE: apply ry with your THETA to qubit 0


    qc.measure(0, 0)
    return qc

build_rotation().draw()"""),
md("""Now run the check cell. **Do not edit it.** If your circuit is right, it prints your verification value. Type it into Canvas question 1 with three decimal places, for example `0.315`."""),
code("# CHECK CELL: do not edit\n" + body("m0_check.py") + '''

if THETA is None:
    print("Enter your angle THETA in Step 5 first, then run this cell again.")
else:
    ok, messages, value = check_rotation(build_rotation(), float(THETA))
    for m in messages:
        print(m)
    if ok:
        print(f"\\nYour verification value for theta = {float(THETA)}: {value:.3f}")
        print("Type this value into Canvas question 1.")
    else:
        print("\\nNot passed yet: no verification value. Fix build_rotation() and run both cells again.")
'''),
md("""## Finish

1. Answer the five questions of the **Module 0 lab check** in Canvas and submit.
2. Fill in the short **Module 0 time log**: how many minutes you spent working, and how many waiting.

""" + SAVE),
]

# ------------------------------------------------------------------ GHZ
ghz = [
md("""# Level 1 final coding task: a three-qubit GHZ state

**Quantum Computing Foundations · Course completion · about 40 minutes**

In Module 4 you built a Bell state, which entangles two qubits. Here you extend that circuit to three qubits and make the **GHZ state**

$$|\\text{GHZ}\\rangle = \\frac{|000\\rangle + |111\\rangle}{\\sqrt{2}}$$

**How this task is graded.** Everything is graded automatically in Canvas:

1. Canvas gives you a personal parameter, θ (theta). You enter it in this notebook.
2. You build the circuit.
3. The check cell at the end tests your circuit. If every test passes, it prints a **verification value** that only a correct circuit produces for your θ. You type that value into Canvas.
4. You answer two short questions in Canvas about what your results mean.

Work through the steps in order. Run each code cell with **Shift + Enter**."""),
md(STEP0.replace("This lab runs", "Like every lab in this course, this one runs")),
code(STEP0_CODE),
md("""## Step 1: enter your parameter

Open the Canvas quiz **Final coding task: GHZ state**. Question 1 shows your θ, for example `θ = 1.25`. Type it below in place of `None`, then run the cell."""),
code("""THETA = None   # for example: THETA = 1.25

print("Your parameter:", THETA)"""),
md("""## Step 2: build the GHZ circuit

The starter code below is your Module 4 Bell circuit, written inside a function. It already has three qubits and three classical bits.

Add what is needed so that **qubit 2 joins the entangled state**, giving (|000⟩ + |111⟩)/√2 before the measurements.

Rules the check cell expects:

- exactly 3 qubits and 3 classical bits;
- measure each qubit **once, at the end**;
- the function must be called `build_ghz` and return the circuit."""),
code("""from qsim import QuantumCircuit

def build_ghz():
    qc = QuantumCircuit(3, 3)

    # Module 4 Bell circuit: H on qubit 0, then CNOT from qubit 0 to qubit 1
    qc.h(0)
    qc.cx(0, 1)

    # YOUR CODE: make qubit 2 part of the same entangled state


    qc.measure([0, 1, 2], [0, 1, 2])
    return qc

build_ghz().draw()"""),
md("""## Step 3: run it and look at the counts

Run your circuit 1,000 times on the simulator. Before you run it, predict: which outcomes should appear, and how often?"""),
code("""import matplotlib.pyplot as plt   # used by plot_histogram
from qsim import transpile, AerSimulator, plot_histogram

sim = AerSimulator()
counts = sim.run(transpile(build_ghz(), sim), shots=1000).result().get_counts()
print(counts)
plot_histogram(counts)"""),
md("""Look at your counts and think about two questions; Canvas asks about them in Step 5:

- If you knew only the result of qubit 0, what could you say about qubits 1 and 2?
- Would these counts alone prove the three qubits share one entangled state?"""),
md("""## Step 4: run the check cell

**Do not edit this cell.** Run it. It checks that:

1. your circuit runs and gives only `000` and `111`, each about half the time;
2. the state just before the measurements is the GHZ state (fidelity at least 0.99).

If both pass, it prints your verification value. Type it into Canvas question 1 with three decimal places, for example `0.315` or `-0.955`. If a test does not pass, read the message, fix your circuit in Step 2, run Step 2 again, then run this cell again."""),
code("# CHECK CELL: do not edit\n" + body("ghz_check.py") + '''

if THETA is None:
    print("Enter your parameter THETA in Step 1 first, then run this cell again.")
else:
    ok, messages, value = check_ghz(build_ghz(), float(THETA))
    for m in messages:
        print(m)
    if ok:
        print(f"\\nAll tests passed. Your verification value for theta = {float(THETA)}: {value:.3f}")
        print("Type this value into Canvas question 1.")
    else:
        print("\\nNot passed yet: no verification value. Fix your circuit in Step 2 and run both cells again.")
'''),
md("""## Step 5: finish in Canvas

1. Type your verification value into question 1.
2. Answer questions 2 and 3 about your results.
3. Submit the quiz.

If you take the quiz again, Canvas may show a new θ. Enter the new value in Step 1, run the check cell again, and use the new verification value.

""" + SAVE),
]

OUT.mkdir(parents=True, exist_ok=True)
for name, cells in [("QC-L1-M0-lab-first-qubit.ipynb", m0), ("QC-L1-final-coding-task-GHZ.ipynb", ghz)]:
    nb = nbf.v4.new_notebook(cells=cells, metadata=META)
    for i, c in enumerate(nb.cells):
        c["id"] = f"{name[:9].lower().replace('-', '')}{i:02d}"
    nbf.write(nb, OUT / name)
    print("wrote", OUT / name)
