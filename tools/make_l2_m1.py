"""Build the Level 2 Module 1 browser notebook (the quantum Fourier transform and phase estimation)
from checks/l2_m1_check.py."""
import shutil
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level2"
CHECK = (ROOT / "checks" / "l2_m1_check.py").read_text().split('"""', 2)[2].lstrip()
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}
NAME = "QC-L2-M1-lab-qft-phase-estimation.ipynb"
VERSION = "2026-10-05"


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md(f"""# Module 1 lab: the quantum Fourier transform and phase estimation

**Quantum Computing Intermediate · Module 1 · about 90 minutes** · notebook version {VERSION}

The quantum Fourier transform (QFT) turns a pattern of **phases** into a number you can measure. Phase estimation uses it to read the eigenphase of a gate into a register of qubits. Both are building blocks of Shor's algorithm (Module 3) and of many chemistry algorithms.

In this lab you:

1. find eigenvalues and eigenphases of gates with NumPy;
2. build the 3-qubit QFT gate by gate, `qft3()`, and check it against the discrete Fourier transform (DFT) matrix;
3. write the QFT for any number of qubits, `qft(n)`;
4. write `phase_estimation()` and use it on the T gate (phase 1/8) and on a phase of 1/3;
5. get your verification value.

The **Module 1 lab check** in Canvas asks for results from this notebook, so keep it open in a second tab. Run each code cell with **Shift + Enter**, in order."""),

md("""## Step 0: start Python

Run the cell below. The first time, Python can take up to a minute to start in your browser. It also prepares two helpers: `dft_matrix(n)`, the 2ⁿ × 2ⁿ DFT matrix, and `show_amplitudes(state)`, which prints each amplitude's size and phase."""),
code("""import math
import numpy as np
import qsim
from qsim import QuantumCircuit, Statevector, Operator, AerSimulator, plot_histogram

sim = AerSimulator(seed_simulator=7)


def dft_matrix(n):
    \"\"\"Prepared for you: the 2^n x 2^n DFT matrix, entry (k, j) = exp(2 pi i j k / 2^n) / sqrt(2^n).\"\"\"
    N = 2 ** n
    w = np.exp(2j * np.pi / N)
    return np.array([[w ** (j * k) for j in range(N)] for k in range(N)]) / math.sqrt(N)


def show_amplitudes(state):
    \"\"\"Prepared for you: each basis state's amplitude as a size and a phase in degrees (0 to 360).\"\"\"
    v = np.asarray(state)
    n = int(round(math.log2(len(v))))
    for k, a in enumerate(v):
        if abs(a) > 1e-9:
            deg = (math.degrees(math.atan2(a.imag, a.real)) + 360) % 360
            print(f"|{format(k, f'0{n}b')}>  (k = {k}):  size {abs(a):.4f},  phase {round(deg, 2) % 360:6.1f} degrees")


print("Ready. qsim", qsim.__version__, "and NumPy", np.__version__)"""),

md("""## Step 1: eigenvalues and eigenphases

An **eigenvector** of a gate U is a state that U leaves unchanged except for a factor: U|ψ⟩ = λ|ψ⟩. For a unitary gate every eigenvalue λ has size 1, so it can be written as λ = e^(2πiφ) with φ between 0 and 1. That number φ is the **eigenphase**, and phase estimation measures it.

The T gate is diag(1, e^(iπ/4)). Its eigenvectors are |0⟩ and |1⟩, with eigenphases 0 and 1/8. Run the cell: NumPy finds the eigenvalues, and the cell converts each to an eigenphase."""),
code("""def eigenphases(U):
    \"\"\"The eigenphases phi (0 <= phi < 1) of a unitary matrix, from its eigenvalues exp(2 pi i phi).\"\"\"
    values = np.linalg.eigvals(np.asarray(U))
    return sorted(float(round((np.angle(v) / (2 * math.pi)) % 1, 6)) for v in values)


T = np.diag([1, np.exp(1j * math.pi / 4)])
P_third = np.diag([1, np.exp(2j * math.pi / 3)])     # the phase gate P(2 pi / 3)
print("T gate eigenphases:      ", eigenphases(T))
print("P(2 pi / 3) eigenphases: ", eigenphases(P_third))"""),
md("""**Your turn (not graded).** In the next cell, find the eigenphases of the S gate, diag(1, i), and of the Hadamard gate. For H, which eigenphase does the eigenvalue −1 give?"""),
code("""# Your turn: eigenphases of S and of H
"""),

md("""## Step 2: the 3-qubit QFT, gate by gate

The QFT on n qubits does to a state's amplitudes what the DFT matrix does to a vector: a basis state |j⟩ becomes

  QFT|j⟩ = (1/√2ⁿ) Σₖ e^(2πi·j·k/2ⁿ) |k⟩,

an equal superposition in which the phase of |k⟩ grows in steps of j/2ⁿ of a full turn.

The circuit works on the highest qubit first. For 3 qubits, with Qiskit's order (qubit 0 on the right):

1. H on qubit 2, then a controlled-phase `cp(pi/2, 1, 2)` and `cp(pi/4, 0, 2)`;
2. H on qubit 1, then `cp(pi/2, 0, 1)`;
3. H on qubit 0;
4. swap qubits 0 and 2, so the output comes out in the right order.

The angle of the controlled phase between qubits k and j is π/2^(j − k). **Your task:** write `qft3()` that returns this 3-qubit circuit. Build it from gates; do not use a matrix."""),
code("""def qft3():
    \"\"\"Return the 3-qubit QFT as a QuantumCircuit built from h, cp and swap gates.\"\"\"
    qc = QuantumCircuit(3)
    # YOUR CODE HERE
    raise NotImplementedError("Complete qft3() first.")
    return qc"""),
md("""Check it against the DFT matrix. `Operator(circuit)` gives the circuit's 8 × 8 matrix."""),
code("""diff = np.abs(Operator(qft3()).data - dft_matrix(3)).max()
print(qft3().draw())
print("largest difference from the DFT matrix:", f"{diff:.1e}")
print("qft3() matches the DFT" if diff < 1e-8 else "Not yet: compare your angles and the final swap with the steps above.")"""),

md("""## Step 3: what the QFT does to a basis state

Prepare |3⟩ = |011⟩ (X on qubits 0 and 1), apply your `qft3()`, and look at the amplitudes. Every amplitude has size 1/√8 ≈ 0.354; the phases step by 3/8 of a full turn (135°) from one k to the next. The lab check asks for one of these phases."""),
code("""qc = QuantumCircuit(3)
qc.x(0)
qc.x(1)
qc.compose(qft3(), inplace=True)
state = Statevector(qc)
show_amplitudes(state)
print()
print("phase of |010> (k = 2):", round((math.degrees(np.angle(state[2])) + 360) % 360, 1), "degrees")"""),

md("""## Step 4: the QFT for any number of qubits

The same pattern works for n qubits: for each qubit j from n − 1 down to 0, an H on qubit j, then `cp(pi / 2**(j - k), k, j)` for every k below j; at the end, swap qubit i with qubit n − 1 − i for i = 0 up to n // 2 − 1.

**Your task:** complete `qft(n)`. The loops are started for you."""),
code("""def qft(n):
    \"\"\"Return the n-qubit QFT as a QuantumCircuit.\"\"\"
    qc = QuantumCircuit(n)
    # YOUR CODE HERE: for j in reversed(range(n)): an H on j, then the controlled phases from each k < j;
    # then the swaps.
    raise NotImplementedError("Complete qft(n) first.")
    return qc"""),
code("""for n in range(1, 6):
    c = qft(n)
    diff = np.abs(Operator(c).data - dft_matrix(n)).max()
    names = [i.name for i in c.data]
    print(f"n = {n}: largest difference {diff:.1e};  {names.count('h')} H, {names.count('cp')} controlled-phase, "
          f"{names.count('swap')} swap gates")"""),
md("""**What to notice.** The number of controlled-phase gates is n(n − 1)/2, so the QFT needs about n²/2 gates. The classical fast Fourier transform needs about n·2ⁿ operations on 2ⁿ numbers. The QFT is exponentially smaller, but its output is a quantum state: you cannot read all 2ⁿ amplitudes, only measure it. Algorithms use it where one measurement is enough, as in phase estimation."""),

md("""## Step 5: phase estimation

Phase estimation finds the eigenphase φ of a gate U for one of its eigenvectors. Here U is the phase gate P(2πφ) = diag(1, e^(2πiφ)), and its eigenvector |1⟩ has eigenphase φ. The circuit has t **counting qubits** (0 to t − 1) and one **target qubit** (qubit t):

1. Put the target in the eigenvector |1⟩ (an X gate) and each counting qubit in |+⟩ (H gates).
2. Counting qubit k controls U raised to the power 2ᵏ. For the phase gate, U^(2ᵏ) is P(2π·φ·2ᵏ), so this is one `cp(2 * math.pi * phase * 2**k, k, t)`. By phase kickback (Level 1, Module 6), counting qubit k picks up the phase e^(2πi·φ·2ᵏ) on its |1⟩ part.
3. The counting register now holds (1/√2ᵗ) Σⱼ e^(2πi·φ·j)|j⟩: exactly the QFT of |φ·2ᵗ⟩. So apply the **inverse** QFT, `qft(t).inverse()`, to the counting qubits.
4. Measure counting qubit k into classical bit k. The result m, read as a binary number, gives the estimate φ ≈ m / 2ᵗ.

**Your task:** write `phase_estimation(phase, t)` following these four steps. Use `qc.compose(qft(t).inverse(), qubits=list(range(t)), inplace=True)` for step 3."""),
code("""def phase_estimation(phase, t):
    \"\"\"Phase estimation of P(2 pi phase) on its eigenvector |1>: t counting qubits, target qubit t.\"\"\"
    qc = QuantumCircuit(t + 1, t)
    # YOUR CODE HERE
    raise NotImplementedError("Complete phase_estimation() first.")
    return qc"""),

md("""## Step 6: the T gate, an exact case

The T gate's eigenphase is 1/8 = 0.001 in binary, which fits exactly in 3 bits. With 3 counting qubits, every shot should give the result 001, so the estimate is exactly 1/8. The lab check asks for the estimate."""),
code("""qc = phase_estimation(1 / 8, 3)
print(qc.draw())
counts = sim.run(qc, shots=1000).result().get_counts()
print(counts)
m = int(max(counts, key=counts.get), 2)
print("most frequent result:", format(m, "03b"), "= m =", m, "  estimate m / 8 =", m / 8)"""),

md("""## Step 7: a phase of 1/3, which no number of bits can hold exactly

1/3 = 0.010101... in binary never ends, so no result is exactly right. Phase estimation then gives the closest t-bit fraction with probability at least 4/π² ≈ 0.405, and nearby fractions the rest of the time. Each extra counting qubit halves the gap between neighbouring estimates.

The cell computes the exact probabilities for t = 3 to 6 (with `Statevector(...).probabilities(qargs)`, new in qsim 1.6.0, which measures only the counting qubits) and draws the histogram for t = 5. The lab check asks for the estimate with 6 counting qubits."""),
code("""for t in range(3, 7):
    qc = phase_estimation(1 / 3, t)
    qc.remove_final_measurements(inplace=True)
    p = Statevector(qc).probabilities(list(range(t)))
    m = int(np.argmax(p))
    print(f"t = {t}: most likely m = {m:2d} of {2 ** t},  estimate {m / 2 ** t:.6f},  error {abs(m / 2 ** t - 1 / 3):.6f},  "
          f"probability {p[m]:.4f},  largest possible error of the best estimate 1/2^(t+1) = {1 / 2 ** (t + 1):.6f}")

counts = sim.run(phase_estimation(1 / 3, 5), shots=1000).result().get_counts()
plot_histogram(counts, title="phase 1/3, 5 counting qubits, 1,000 shots")"""),

md("""## Step 8: your personal check

Open the **Module 1 lab check** in Canvas. Question 1 shows your own phase PHI and seed SEED. Type them below and run the next two cells. The check cell tests `qft3()`, `qft(n)` for n = 1 to 5, and `phase_estimation()` on seven phases. Only if every test passes does it print your **verification value**: the number of shots, out of 1,000, in which the course's reference phase-estimation circuit with 5 counting qubits returns the best 5-bit estimate of PHI, run with your seed. Type it into Canvas."""),
code("""PHI = 0.0     # your phase from Canvas, for example 0.432
SEED = 0      # your seed from Canvas, for example 512"""),
code(CHECK + '''

passed, messages, value = check_l2_module1(qft3, qft, phase_estimation, PHI, SEED)
for m in messages:
    print(m)
if passed:
    print("\\nAll tests passed. Your verification value is", value)
else:
    print("\\nNot yet: fix the item above and run this cell again.")'''),

md("""## What you should notice

- A unitary gate's eigenvalues are e^(2πiφ); the eigenphase φ is what phase estimation measures.
- The QFT is the DFT matrix built from about n²/2 gates: H gates, controlled phases π/2^(j − k), and swaps at the end.
- Phase estimation kicks the phase back onto the counting qubits, then the inverse QFT turns that phase pattern into a binary number m with φ ≈ m / 2ᵗ.
- A phase with an exact t-bit expansion (1/8) is found in every shot; any other (1/3) gives the closest t-bit fraction with probability at least 4/π², and more counting qubits make the estimate finer.

Module 3 uses phase estimation to find the period in Shor's algorithm.

Developed through the Intel Semiconductor Education Program at Central State University (ISEP-CSU). CC BY 4.0. Questions: mhadizadeh@centralstate.edu"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl2m1c{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / NAME)
shutil.copyfile(ROOT / "content" / "level1" / "qsim.py", OUT / "qsim.py")
print("wrote", OUT / NAME, "and synced qsim.py")
