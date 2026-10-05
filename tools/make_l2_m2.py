"""Build the Level 2 Module 2 browser notebook (Grover's search) from checks/l2_m2_check.py."""
import shutil
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level2"
CHECK = (ROOT / "checks" / "l2_m2_check.py").read_text().split('"""', 2)[2].lstrip()
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}
NAME = "QC-L2-M2-lab-grover-search.ipynb"
VERSION = "2026-10-05"


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md(f"""# Module 2 lab: Grover's search

**Quantum Computing Intermediate · Module 2 · about 90 minutes** · notebook version {VERSION}

Grover's search finds a marked item among N possibilities with about (π/4)√N questions to an oracle, where a classical search needs about N/2 on average. It works by **amplitude amplification**: the oracle marks the answer with a minus sign, and the diffuser turns that sign into a larger amplitude, a little more each round.

In this lab you:

1. meet the multi-controlled Z gate, the building block of oracles and diffusers;
2. write a phase oracle, `oracle(marked)`, for any marked bit string;
3. write the diffuser, `diffuser(n)`, the reflection about the mean;
4. put them together in `grover(marked, iterations)` and run it on 2 and 3 qubits;
5. see how the success probability rises and falls with the number of iterations, and what happens with two marked items;
6. get your verification value.

The **Module 2 lab check** in Canvas asks for results from this notebook, so keep it open in a second tab. Run each code cell with **Shift + Enter**, in order."""),

md("""## Step 0: start Python

Run the cell below. The first time, Python can take up to a minute to start in your browser. It also prepares `show_amplitudes(state)`, which prints each amplitude (Grover's amplitudes stay real numbers, so it prints them with their sign) and its probability."""),
code("""import math
import numpy as np
import qsim
from qsim import QuantumCircuit, Statevector, Operator, AerSimulator, plot_histogram

sim = AerSimulator(seed_simulator=7)


def show_amplitudes(state):
    \"\"\"Prepared for you: each basis state's amplitude (a real number with its sign) and its probability.\"\"\"
    v = np.asarray(state)
    n = int(round(math.log2(len(v))))
    for k, a in enumerate(v):
        r = a.real if abs(a.real) > 5e-5 else 0.0
        bar = "#" * int(round(40 * abs(r)))
        print(f"|{format(k, f'0{n}b')}>  amplitude {r:+.4f}   probability {abs(a) ** 2:.4f}   {bar}")


print("Ready. qsim", qsim.__version__, "and NumPy", np.__version__)"""),

md("""## Step 1: the multi-controlled Z gate

Oracles and diffusers both need one gate: a **multi-controlled Z**, which multiplies the amplitude of the all-ones state |11...1⟩ by −1 and leaves every other basis state alone. On 2 qubits it is `cz(0, 1)`; on 3 qubits Qiskit has `ccz(0, 1, 2)`. For any number of qubits, put an H on one qubit, use `mcx` (a multi-controlled X) with that qubit as the target, and put an H on it again: H X H = Z, so the controlled X becomes a controlled Z.

The helper `mcz(qc, qubits)` below does this for you. Run the cell: on 3 qubits only |111⟩ gets a −1."""),
code("""def mcz(qc, qubits):
    \"\"\"Prepared for you: a multi-controlled Z on the given qubits (flips the sign of the all-ones state).\"\"\"
    qubits = list(qubits)
    if len(qubits) == 1:
        qc.z(qubits[0])
    elif len(qubits) == 2:
        qc.cz(qubits[0], qubits[1])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])


qc = QuantumCircuit(3)
mcz(qc, [0, 1, 2])
print(qc.draw())
print("diagonal of the matrix:", np.round(np.diag(Operator(qc).data).real, 6))"""),

md("""## Step 2: a phase oracle on 2 qubits

A **phase oracle** for a marked string w multiplies the amplitude of |w⟩ by −1 and does nothing else. For w = 11 on 2 qubits, that is exactly `cz(0, 1)`.

Run the cell: H on both qubits gives the equal superposition (every amplitude +0.5), and the oracle flips the sign of |11⟩. Notice that the **probabilities do not change**. Measuring now would still give each result a quarter of the time: the oracle hides its answer in a sign, and the diffuser's job (Step 4) is to turn that sign into probability."""),
code("""qc = QuantumCircuit(2)
qc.h([0, 1])
qc.cz(0, 1)          # the phase oracle for 11
show_amplitudes(Statevector(qc))"""),
md("""**To mark a string with zeros in it**, surround the multi-controlled Z with X gates on the qubits whose bit is 0: the X gates turn |w⟩ into |11...1⟩, the multi-controlled Z flips its sign, and the same X gates turn it back.

Remember Qiskit's bit order: in the string "01", the **rightmost** bit belongs to qubit 0. So for w = 01, qubit 0 has bit 1 and qubit 1 has bit 0, and the X gates go on qubit 1. In code, the bit for qubit q is `marked[n - 1 - q]`.

**Your turn (not graded).** Build the oracle for w = 01 in the next cell and check that only |01⟩ has a minus sign."""),
code("""qc = QuantumCircuit(2)
qc.h([0, 1])
# Your turn: the oracle for 01 (X on qubit 1, cz, X on qubit 1)

show_amplitudes(Statevector(qc))"""),

md("""## Step 3: a phase oracle for any marked string

**Your task:** write `oracle(marked)`. It receives a bit string such as `"101"` and returns a circuit on n = len(marked) qubits that multiplies the amplitude of |marked⟩ by −1:

1. X on every qubit q whose bit `marked[n - 1 - q]` is "0";
2. `mcz(qc, range(n))`;
3. the same X gates again.

Do not use H gates on all the qubits here: the oracle should not create a superposition, only change one sign."""),
code("""def oracle(marked):
    \"\"\"Return the phase oracle for the bit string marked (Qiskit order: the bit for qubit q is marked[n - 1 - q]).\"\"\"
    n = len(marked)
    qc = QuantumCircuit(n)
    # YOUR CODE HERE
    raise NotImplementedError("Complete oracle() first.")
    return qc"""),
md("""Check it: the cell prints the diagonal of the oracle's matrix for every 3-qubit string. Each row should have exactly one −1, in the column of its own string."""),
code("""print("marked   diagonal of the matrix (|000> ... |111>)")
all_ok = True
for k in range(8):
    w = format(k, "03b")
    d = np.round(np.diag(Operator(oracle(w)).data).real).astype(int)
    ok = d[k] == -1 and sum(d == -1) == 1
    all_ok = all_ok and ok
    print(f"  {w}    {d}   {'ok' if ok else 'not yet'}")
print("oracle() marks every 3-qubit string correctly" if all_ok else "Not yet: compare the X gates with the bit order rule above.")"""),

md("""## Step 4: the diffuser, a reflection about the mean

The **diffuser** replaces every amplitude a by 2·m − a, where m is the mean (average) of all the amplitudes. An amplitude below the mean ends up as far above it as it was below. After the oracle, the marked amplitude is negative, far below the mean, so the diffuser lifts it well above everything else.

Example on 2 qubits after the oracle for 10: the amplitudes are +0.5, +0.5, −0.5, +0.5, the mean is 0.25, and 2·0.25 − a gives 0, 0, 1, 0. One round finds the answer with certainty.

As a matrix the diffuser is 2|s⟩⟨s| − I, where |s⟩ is the equal superposition. Its circuit is:

1. H on every qubit (|s⟩ becomes |00...0⟩);
2. X on every qubit (|00...0⟩ becomes |11...1⟩);
3. `mcz(qc, range(n))`;
4. X on every qubit;
5. H on every qubit.

This circuit gives I − 2|s⟩⟨s|, which is the diffuser times −1. An overall factor of −1 on a whole state is invisible to every measurement, so it does no harm; Qiskit's own Grover operator has it too.

**Your task:** write `diffuser(n)`."""),
code("""def diffuser(n):
    \"\"\"Return the reflection about the mean on n qubits (up to an overall sign).\"\"\"
    qc = QuantumCircuit(n)
    # YOUR CODE HERE
    raise NotImplementedError("Complete diffuser() first.")
    return qc"""),
md("""Check it on the example: H on both qubits, the oracle for 10, then your diffuser. The marked state should have amplitude −1 (the overall sign from the circuit) and probability 1."""),
code("""qc = QuantumCircuit(2)
qc.h([0, 1])
qc.compose(oracle("10"), inplace=True)
qc.compose(diffuser(2), inplace=True)
show_amplitudes(Statevector(qc))"""),

md("""## Step 5: Grover's search

One **Grover iteration** is the oracle followed by the diffuser. Grover's search starts from the equal superposition, repeats the iteration, and measures.

**Your task:** write `grover(marked, iterations)`:

1. a circuit with n qubits and n classical bits, n = len(marked);
2. H on every qubit;
3. `iterations` times: compose `oracle(marked)`, then `diffuser(n)`;
4. measure qubit q into classical bit q.

Then run it on 2 qubits with the marked string 10 and one iteration. The lab check asks how many of the 1,000 shots give 10."""),
code("""def grover(marked, iterations):
    \"\"\"Grover's search for one marked bit string: H on all qubits, `iterations` rounds of oracle and diffuser, measure.\"\"\"
    n = len(marked)
    qc = QuantumCircuit(n, n)
    # YOUR CODE HERE
    raise NotImplementedError("Complete grover() first.")
    return qc"""),
code("""qc = grover("10", 1)
print(qc.draw())
counts = sim.run(qc, shots=1000).result().get_counts()
print(counts)
print("shots that gave 10:", counts.get("10", 0), "of 1000")"""),

md("""## Step 6: how many iterations?

With N = 2ⁿ items and one marked, the state after t iterations stays in the plane of two states: |w⟩ (the marked one) and the equal superposition of the others. It starts at a small angle θ from the unmarked direction, with sin θ = 1/√N, and **each iteration rotates it by 2θ** towards |w⟩. After t iterations the angle is (2t + 1)θ, so

  P(success after t iterations) = sin²((2t + 1)θ).

The best t makes (2t + 1)θ close to 90°: about (π/4)√N. More iterations rotate the state **past** |w⟩, and the success probability falls again.

Run the cell: for 3 qubits and the marked string 101, it computes the exact probability of measuring 101 after 0 to 6 iterations (with `Statevector`, after removing the final measurements), next to the formula, and plots it. The lab check asks for the probability at the best number of iterations and one iteration later."""),
code("""marked = "101"
n = len(marked)
theta = math.asin(1 / math.sqrt(2 ** n))
print(f"N = {2 ** n},  theta = {math.degrees(theta):.4f} degrees,  (pi/4)*sqrt(N) = {math.pi / 4 * math.sqrt(2 ** n):.3f}")
probs = []
for t in range(7):
    qc = grover(marked, t)
    qc.remove_final_measurements(inplace=True)
    p = Statevector(qc).probabilities_dict().get(marked, 0.0)
    probs.append(p)
    print(f"t = {t}:  P({marked}) = {p:.5f}    formula sin^2((2t+1) theta) = {math.sin((2 * t + 1) * theta) ** 2:.5f}")
best = int(np.argmax(probs[:4]))
print("best number of iterations (first peak):", best, " probability", round(probs[best], 5))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(range(7), probs, "o-", color="#99004C")
ax.set_xlabel("iterations t")
ax.set_ylabel(f"P({marked})")
ax.set_ylim(0, 1.05)
ax.set_title("Grover's search on 3 qubits: success against iterations")
ax.grid(alpha=0.3)
fig"""),
md("""**What to notice.** The success probability peaks at 2 iterations (about 0.945, not 1: (2t + 1)θ = 5θ is about 103.5°, near 90° but not on it), drops sharply at 3, almost vanishes at 4, and comes back later. Unlike most algorithms, Grover's search gets **worse** if you run it too long."""),

md("""## Step 7: two marked items

An oracle can mark several strings: compose one oracle per string, and each flips the sign of its own state. With s marked items out of N, sin θ = √(s/N). For s = 2 and N = 8, θ = 30°, so one iteration gives (2·1 + 1)·30° = 90°: certainty.

The diffuser is the same as before. Run the cell: it marks 011 and 110 and runs one iteration. The probability of getting one of the two marked strings should be 1. (This also previews a mini-project option, Grover on 2 or 3 qubits.)"""),
code("""def oracle_two(w1, w2):
    \"\"\"An oracle that marks two strings: one phase oracle after the other.\"\"\"
    qc = oracle(w1)
    qc.compose(oracle(w2), inplace=True)
    return qc


qc = QuantumCircuit(3)
qc.h([0, 1, 2])
qc.compose(oracle_two("011", "110"), inplace=True)
qc.compose(diffuser(3), inplace=True)
p = Statevector(qc).probabilities_dict()
show_amplitudes(Statevector(qc))
print("P(011) + P(110) =", round(p.get("011", 0) + p.get("110", 0), 6))"""),

md("""## Step 8: your personal check

Open the **Module 2 lab check** in Canvas. Question 1 shows your own number MARKED (0 to 15) and seed SEED. Type them below and run the next two cells. The check cell tests `oracle()` on every string of 2, 3 and 4 qubits, `diffuser()` on 2 to 5 qubits, and `grover()` on eight cases. Only if every test passes does it print your **verification value**: the course's reference Grover circuit on 4 qubits searches for the 4-bit string of MARKED with 1, 2, 3, 4 and 5 iterations (1,000 shots each, seeded with your SEED), and the value is the total number of shots, out of 5,000, that found it. Type it into Canvas."""),
code("""MARKED = -1    # your number from Canvas, for example 13 (the 4-bit string 1101)
SEED = 0       # your seed from Canvas, for example 512"""),
code(CHECK + '''

passed, messages, value = check_l2_module2(oracle, diffuser, grover, MARKED, SEED)
for m in messages:
    print(m)
if passed:
    print("\\nAll tests passed. Your verification value is", value)
else:
    print("\\nNot yet: fix the item above and run this cell again.")'''),

md("""## What you should notice

- A phase oracle hides the answer in a sign: it changes no probability on its own.
- The diffuser reflects every amplitude about the mean, which turns the marked minus sign into a larger amplitude.
- Each Grover iteration rotates the state by 2θ, with sin θ = √(s/N). The best number of iterations is about (π/4)√(N/s), and running longer makes the result worse.
- For N items, Grover needs about √N oracle calls; a classical search needs about N/2 on average. That is a quadratic speed-up, not an exponential one.

**Next in Canvas:** the lab check, the quiz, and the short mini-project choice."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl2m2c{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / NAME)
shutil.copyfile(ROOT / "content" / "level1" / "qsim.py", OUT / "qsim.py")
print("wrote", OUT / NAME, "and synced qsim.py")
