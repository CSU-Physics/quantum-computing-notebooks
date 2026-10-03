"""Build the Level 1 Module 6 browser notebook (JupyterLite, Pyodide kernel) from checks/m6_check.py."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level1"
CHECK = (ROOT / "checks" / "m6_check.py").read_text().split('"""', 2)[2].lstrip()
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}
NAME = "QC-L1-M6-lab-bernstein-vazirani.ipynb"


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md("""# Module 6 lab: a first quantum algorithm, Bernstein-Vazirani

**Quantum Computing Foundations · Module 6 · about 75 minutes**

A box hides a string of bits s. You may ask it questions: give it a string x, and it answers one bit, s·x mod 2. A classical computer needs one question per bit of s. The Bernstein-Vazirani algorithm finds all of s with **one** question. In this lab you write three functions:

1. `classical_bv(query, n)`: the classical method, one query per bit;
2. `bv_oracle(s)`: the quantum version of the box, as a circuit;
3. `bv_circuit(oracle, n)`: the Bernstein-Vazirani circuit, which finds the hidden string of any oracle with one use of it.

Then you run the same circuit as the Deutsch-Jozsa algorithm, compare the number of queries, and see what noise does to it. The **Module 6 lab check** asks for results from this notebook, so keep it open in a second tab. Run each code cell with **Shift + Enter**, in order."""),
md("""## Step 0: start Python

Run the cell below. The first time, Python can take up to a minute to start in your browser. qsim 1.4.0 adds `QuantumCircuit.compose`, which joins two circuits as in Qiskit."""),
code("""import numpy as np
import matplotlib.pyplot as plt   # used for the plots below
import qsim
from qsim import (QuantumCircuit, Statevector, Operator, AerSimulator, plot_histogram, bloch_vectors,
                  NoiseModel, depolarizing_error)

sim = AerSimulator(seed_simulator=7)


def f_hidden(s, x):
    \"\"\"Prepared for you: s . x mod 2, the parity of the positions where both strings have a 1.\"\"\"
    return sum(int(a) * int(b) for a, b in zip(s, x)) % 2


print("Ready. qsim", qsim.__version__, "and NumPy", np.__version__)"""),
md("""## Step 1: the problem, and the classical answer

Strings are written like Qiskit results: the **rightmost** character belongs to qubit 0. The hidden function is

f(x) = s·x mod 2 = (s<sub>n−1</sub>x<sub>n−1</sub> + … + s<sub>1</sub>x<sub>1</sub> + s<sub>0</sub>x<sub>0</sub>) mod 2.

Run the first cell: it prints f(x) for s = 1011 and all 16 inputs. Canvas asks how many of them give f(x) = 1."""),
code("""s = "1011"
print("x     f(x)")
for k in range(16):
    x = format(k, "04b")
    print(x, "   ", f_hidden(s, x))
ones = sum(f_hidden(s, format(k, "04b")) for k in range(16))
print(f"\\n{ones} of the 16 inputs give f(x) = 1")"""),
md("""A classical computer can only ask f about one x at a time, and each answer is one bit. The best strategy asks about the inputs with a single 1: the input with a 1 in position j answers with s[j]. That takes **n queries** for an n-bit string, and no classical method can do it with fewer, because each answer carries one bit of information.

Complete `classical_bv(query, n)`. `query` is a function: `query("0100")` returns f for x = 0100. Return the hidden string as a text string, such as `"1011"`. Then run the cell: it tests your function on a hidden 8-bit string and counts the queries."""),
code("""def classical_bv(query, n):
    # YOUR CODE: replace the next line. Ask n questions, one per bit, and return s as a string.
    raise NotImplementedError("Complete classical_bv() first.")


def make_query(secret):
    \"\"\"A classical box for a hidden string, which counts how often it is asked.\"\"\"
    def query(x):
        query.calls += 1
        return f_hidden(secret, x)
    query.calls = 0
    return query


box = make_query("10110010")
found = classical_bv(box, 8)
print("hidden string found:", found, "  queries used:", box.calls)"""),
md("""## Step 2: phase kickback

The quantum version of the box is a circuit called an **oracle**. It acts on n input qubits and one extra **answer qubit** (qubit n): |x⟩|y⟩ → |x⟩|y ⊕ f(x)⟩. On its own that just writes f(x) into the answer qubit. The trick is to put the answer qubit in |−⟩ first. Then flipping it does nothing to it, but multiplies the state by −1: the answer goes into the **phase** of the input, as (−1)<sup>f(x)</sup>|x⟩. This is **phase kickback**.

Run the cell. It puts qubit 0 in |+⟩ and the answer qubit 1 in |−⟩, applies one CNOT, and prints both Bloch vectors before and after."""),
code("""qc = QuantumCircuit(2)
qc.h(0)              # input qubit 0 in |+>
qc.x(1)
qc.h(1)              # answer qubit 1 in |->
before = bloch_vectors(Statevector(qc))
qc.cx(0, 1)
after = bloch_vectors(Statevector(qc))
for q, name in ((0, "input qubit 0 "), (1, "answer qubit 1")):
    print(f"{name}: before {np.round(before[q], 3) + 0}   after {np.round(after[q], 3) + 0}")"""),
md("""The CNOT targets the answer qubit, yet the answer qubit is unchanged and the **control** turned from |+⟩ (Bloch vector along +x) into |−⟩ (along −x). The −1 for the |1⟩ part of the input was kicked back from the target to the control."""),
md("""## Step 3: the oracle as a circuit

For f(x) = s·x mod 2, the oracle is simple: for every 1 in s, a CNOT from that input qubit to the answer qubit. Each CNOT adds x<sub>i</sub> to the answer, so together they add s·x mod 2. Remember the order: character `s[n - 1 - i]` belongs to qubit i.

Complete `bv_oracle(s)`. It returns a circuit with n + 1 qubits (inputs 0 to n − 1, answer qubit n) and no measurements. The cell checks it against f for all 16 inputs of s = 1011 and draws it."""),
code("""def bv_oracle(s):
    # YOUR CODE: replace the next line. Return a QuantumCircuit with len(s) + 1 qubits.
    raise NotImplementedError("Complete bv_oracle() first.")


s = "1011"
n = len(s)
oracle = bv_oracle(s)
right = 0
for k in range(2 ** n):
    x = format(k, f"0{n}b")
    qc = QuantumCircuit(n + 1)
    for i in range(n):
        if x[n - 1 - i] == "1":
            qc.x(i)                      # prepare the input |x>, answer qubit |0>
    out = Statevector(qc.compose(oracle)).probabilities_dict()
    result = list(out)[0]                # one basis state; its leftmost bit is the answer qubit
    right += int(result[0]) == f_hidden(s, x)
print(f"the oracle wrote f(x) correctly for {right} of {2 ** n} inputs")
print(oracle.draw())"""),
md("""## Step 4: the Bernstein-Vazirani circuit

The algorithm:

1. Put the answer qubit in |−⟩ (X, then H) and every input qubit in |+⟩ (H).
2. Use the oracle **once**. By phase kickback the input register becomes the sum of (−1)<sup>s·x</sup>|x⟩ over all x.
3. Put H on every input qubit again. The signs interfere so that everything cancels except |s⟩.
4. Measure the n input qubits: qubit i into classical bit i.

Complete `bv_circuit(oracle, n)`. It receives an oracle whose hidden string it does **not** know, adds it once with `qc.compose(oracle)`, and returns a circuit with n + 1 qubits and n classical bits. Then run the cell. It first shows the signs after step 2 for s = 101, then finds 1011, then a mystery 8-bit string. Canvas asks for the mystery string as a decimal number."""),
code("""def bv_circuit(oracle, n):
    # YOUR CODE: replace the next line. Return a QuantumCircuit(n + 1, n) that uses the oracle once.
    raise NotImplementedError("Complete bv_circuit() first.")


# The signs after the oracle, for s = 101: the input register is the sum of (-1)^(s.x) |x> / sqrt(8)
s = "101"
prep = QuantumCircuit(4)
prep.x(3)
prep.h(range(4))
amps = Statevector(prep.compose(bv_oracle(s))).data
signs = np.sign(np.real(amps[:8]))     # answer qubit |-> = (|0> - |1>)/sqrt(2): its |0> half
for k in range(8):
    x = format(k, "03b")
    print(f"x = {x}: sign {'+' if signs[k] > 0 else '-'}   (s.x mod 2 = {f_hidden(s, x)})")

counts = sim.run(bv_circuit(bv_oracle("1011"), 4), shots=1000).result().get_counts()
print("\\nhidden string 1011:", counts)


def mystery_oracle():
    \"\"\"Prepared for you: an oracle for a hidden 8-bit string.\"\"\"
    return bv_oracle(format(181, "08b"))


counts = sim.run(bv_circuit(mystery_oracle(), 8), shots=1000).result().get_counts()
found = max(counts, key=counts.get)
print("mystery oracle:", counts, "  as a decimal number:", int(found, 2))"""),
md("""One query found all of s. Each query to the quantum oracle acts on a superposition of all 2<sup>n</sup> inputs at once, and the final H gates turn the pattern of signs into the answer. Note that the circuit never "reads" all the values f(x): only the interference pattern matters."""),
md("""## Step 5 (guided): Deutsch-Jozsa, the same circuit

The **Deutsch-Jozsa** problem: f is promised to be either **constant** (the same answer for every x) or **balanced** (0 for exactly half of the inputs). Which is it? Classically, in the worst case you must check more than half of the inputs. The Deutsch-Jozsa algorithm uses exactly the Bernstein-Vazirani circuit: if f is constant, every shot gives 00…0; if it is balanced, no shot does.

Run the cell. It tries five prepared 3-bit oracles with your `bv_circuit()`: A and E are constant, B, C and D are balanced. D is not of the form s·x, so its results are spread out, but they still never include 000."""),
code("""def dj_oracle(kind, n=3):
    \"\"\"Prepared for you: five oracles for f on 3 bits, with the answer qubit 3.\"\"\"
    qc = QuantumCircuit(n + 1)
    if kind == "A":
        qc.x(n)                        # f(x) = 1 for every x: constant
    elif kind == "B":
        qc.cx(1, n)                    # f(x) = x1: balanced
    elif kind == "C":
        qc.cx(0, n)
        qc.cx(2, n)                    # f(x) = x0 XOR x2: balanced
    elif kind == "D":
        qc.ccx(0, 1, n)
        qc.cx(2, n)                    # f(x) = (x0 AND x1) XOR x2: balanced, not of the form s.x
    return qc                          # "E": f(x) = 0 for every x: constant


for kind in "ABCDE":
    ones = 0
    for k in range(8):                 # the classical truth table, to label the oracle
        x = format(k, "03b")
        qc = QuantumCircuit(4)
        for i in range(3):
            if x[2 - i] == "1":
                qc.x(i)
        ones += int(list(Statevector(qc.compose(dj_oracle(kind))).probabilities_dict())[0][0])
    label = "constant" if ones in (0, 8) else "balanced" if ones == 4 else "neither"
    c = sim.run(bv_circuit(dj_oracle(kind), 3), shots=1000).result().get_counts()
    print(f"oracle {kind}: f(x) = 1 for {ones} of 8 inputs ({label:8s})  P(000) = {c.get('000', 0) / 1000:.3f}   {c}")"""),
md("""## Step 6: counting queries

The table below compares the number of queries for n-bit inputs. For Bernstein-Vazirani, a classical computer needs n queries and the quantum circuit 1: a large saving, but it grows only in proportion to n. For Deutsch-Jozsa, a classical computer that must be **certain** needs 2<sup>n−1</sup> + 1 queries in the worst case, against 1: an exponential gap. (A classical computer that may be wrong with a small probability needs only a few random queries, which is why Deutsch-Jozsa is a demonstration rather than a practical speedup.)"""),
code("""ns = np.arange(1, 11)
print(" n   BV classical   DJ classical (certain, worst case)   quantum")
for n in ns:
    print(f"{n:2d}   {n:12d}   {2 ** (n - 1) + 1:34d}   {1:7d}")

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.semilogy(ns, ns, "o-", color="#24313D", label="Bernstein-Vazirani, classical")
ax.semilogy(ns, 2 ** (ns - 1) + 1, "s-", color="#6A626B", label="Deutsch-Jozsa, classical and certain")
ax.semilogy(ns, np.ones_like(ns), "D-", color="#99004C", label="quantum, both problems")
ax.set_xlabel("number of input bits n")
ax.set_ylabel("queries")
ax.legend(fontsize=8)
plt.show()"""),
md("""## Step 7: the algorithm with noise

Real hardware adds errors (Module 5). Run the circuit for s = 111111 (six CNOTs in the oracle) with a depolarizing error λ on every CNOT, 1,000 shots and seed 7, and count how often it still returns the hidden string. Canvas asks for the count at λ = 0.05.

**Queries and shots.** "One query" means one use of the oracle in one ideal run of the circuit. Here each of the 1,000 shots is a separate run, so the oracle is used 1,000 times in all, and repeating runs to take the most frequent answer uses it more. The comparison with the classical n queries is about the ideal circuit; on noisy hardware you pay extra shots to read s reliably."""),
code("""s = "111111"
for lam in (0, 0.01, 0.02, 0.05, 0.10):
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(lam, 2), ["cx"])
    noisy = AerSimulator(seed_simulator=7, noise_model=model)
    c = noisy.run(bv_circuit(bv_oracle(s), 6), shots=1000).result().get_counts()
    print(f"lambda = {lam:.2f}: hidden string returned in {c.get(s, 0):4d} of 1000 shots")"""),
md("""Each CNOT in the oracle is one more chance for an error, so a hidden string with more 1s is found less reliably. The most common result is still the right one at these error rates, so repeating the run and taking the most frequent answer works; for larger circuits the errors add up quickly, which is why error rates matter so much."""),
md("""## Step 8: your personal oracle

Open the Canvas quiz **Module 6 lab check**. Question 1 gives you two numbers: **KEY** (which chooses a hidden 6-bit string and the random seed) and **DEP_PCT** (a depolarizing error on each CNOT, in percent). Type them below in place of `None` and run the cell."""),
code("""KEY = None       # for example: KEY = 512
DEP_PCT = None   # for example: DEP_PCT = 5

if None in (KEY, DEP_PCT):
    print("Enter KEY and DEP_PCT from Canvas first.")
else:
    print(f"KEY = {KEY} chooses your hidden 6-bit string; the check cell runs the Bernstein-Vazirani circuit "
          f"for it with a {DEP_PCT}% depolarizing error on each CNOT.")"""),
md("""## Step 9: run the check cell

**Do not edit this cell.** Run it. It tests your three functions on cases you have not seen, including hidden oracles. If they all pass, it runs the Bernstein-Vazirani circuit for your personal oracle 1,000 times with the noise from Step 8 and prints your **verification value**: the number of shots that returned your hidden string. Type it into Canvas question 1.

If a test does not pass, read its message, fix that function, run its cell again, then run this cell again."""),
code("# CHECK CELL: do not edit\n" + CHECK + '''

ok, messages, value = check_module6(bv_oracle, bv_circuit, classical_bv, KEY, DEP_PCT)
for m in messages:
    print(m)
if ok:
    print(f"\\nAll tests passed. Your verification value for KEY = {KEY}, DEP_PCT = {DEP_PCT}: {value}")
    print("Type this number into Canvas question 1.")
else:
    print("\\nNot passed yet: no verification value.")
'''),
md("""## Finish

1. Answer the five questions of the **Module 6 lab check** in Canvas and submit.
2. Take the **Module 6 quiz**.
3. Fill in the short **Module 6 time log**.

**Optional, not graded:** run your own `bv_oracle()` and `bv_circuit()` unchanged in real Qiskit, transpiled for a model of an IBM computer and run with its calibrated noise: [open the Module 6 Qiskit notebook in Colab](https://colab.research.google.com/github/CSU-Physics/quantum-computing-notebooks/blob/main/colab/level1/QC-L1-M6-bernstein-vazirani-qiskit-colab.ipynb) (a Google account is needed for Colab; no IBM account).

Your notebook is saved in this browser as you work. It is not saved anywhere else, so to keep a copy choose **File > Download**. Next is Course Completion: the final quiz and the final coding task, which extends your Bell circuit to three qubits (a GHZ state)."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl1m6{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / NAME)
print("wrote", OUT / NAME)
