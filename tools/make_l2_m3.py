"""Build the Level 2 Module 3 browser notebook (Shor's algorithm and period finding for N = 15) from checks/l2_m3_check.py."""
import shutil
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level2"
SRC = (ROOT / "checks" / "l2_m3_check.py").read_text()
CHECK = SRC.split('"""', 2)[2].lstrip()
PREPARED = SRC[SRC.index("def c_amod15"):SRC.index("# ---------------------------------------------------------------- reference solutions")].rstrip()
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}
NAME = "QC-L2-M3-lab-shor-period-finding.ipynb"
VERSION = "2026-10-05"


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md(f"""# Module 3 lab: Shor's algorithm and period finding

**Quantum Computing Intermediate · Module 3 · about 60 minutes** · notebook version {VERSION}

Shor's algorithm factors a number N by finding the **order** r of a number a: the smallest r with aʳ mod N = 1. A classical computer finds the factors from r in a few steps; the hard part, finding r, is done by **phase estimation** (Module 1) on the gate that multiplies by a mod N.

In this lab you factor N = 15:

1. find orders classically, `order(a, N)`, and turn an order into factors, `factors_from_order(a, r, N)`;
2. use a **prepared** controlled multiplication circuit `c_amod15(a, power)`, hard-coded for N = 15;
3. build the quantum order-finding circuit, `order_finding(a, t)`;
4. read r from the measured results with continued fractions, `period_from_counts(counts, t, a, N)`;
5. factor 15 for every allowed a, and get your verification value.

The **Module 3 lab check** in Canvas asks for results from this notebook, so keep it open in a second tab. Run each code cell with **Shift + Enter**, in order."""),

md("""## Step 0: start Python

Run the cell below. The first time, Python can take up to a minute to start in your browser."""),
code("""import math
from fractions import Fraction
import numpy as np
import qsim
from qsim import QuantumCircuit, Statevector, AerSimulator, plot_histogram

sim = AerSimulator(seed_simulator=7)
print("Ready. qsim", qsim.__version__, "and NumPy", np.__version__)"""),

md("""## Step 1: the order of a mod N

For a number a that shares no factor with N, the powers a¹, a², a³, ... mod N repeat. The **order** r is the smallest r > 0 with aʳ mod N = 1. For example, for a = 2 and N = 15: 2, 4, 8, 16 mod 15 = 1, so r = 4.

**Your task:** write `order(a, N)` by trying r = 1, 2, 3, ... and returning the first r with `pow(a, r, N) == 1`. This classical search takes about r steps, and r can be almost as large as N; for a 2,048-bit N that is hopeless, which is why the quantum part is needed.

The cell after it prints the order of every a that the prepared circuit supports. The lab check asks for the order of 7."""),
code("""def order(a, N):
    \"\"\"The smallest r > 0 with a^r mod N == 1 (classical search).\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete order() first.")"""),
code("""for a in (2, 4, 7, 8, 11, 13):
    powers = [pow(a, k, 15) for k in range(1, 7)]
    print(f"a = {a:2d}:  a^1 .. a^6 mod 15 = {powers}   order r = {order(a, 15)}")"""),

md("""## Step 2: from the order to the factors

If r is **even**, write h = a^(r/2) mod N. Then h² − 1 = (h − 1)(h + 1) is a multiple of N. If also h ≠ N − 1, neither h − 1 nor h + 1 is a multiple of N, so each shares a factor with N, and the greatest common divisors gcd(h − 1, N) and gcd(h + 1, N) are factors of N. If r is odd, or h = N − 1, this a gives no factor: pick another a.

**Your task:** write `factors_from_order(a, r, N)`. Return the two factors as a sorted tuple, or `None` when r is odd or h = N − 1. Use `math.gcd` and `pow(a, r // 2, N)`.

The cell after it uses a = 7 and r = 4: h = 7² mod 15 = 4, gcd(3, 15) = 3 and gcd(5, 15) = 5. The lab check asks for gcd(7² + 1, 15)."""),
code("""def factors_from_order(a, r, N):
    \"\"\"The factors (p, q), sorted, from an even order r with a^(r/2) mod N != N - 1; otherwise None.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete factors_from_order() first.")"""),
code("""print("factors from a = 7, r = 4:", factors_from_order(7, 4, 15))
print("gcd(7**2 - 1, 15) =", math.gcd(7 ** 2 - 1, 15), "   gcd(7**2 + 1, 15) =", math.gcd(7 ** 2 + 1, 15))
print("a = 14, r = 2 (14 = -1 mod 15):", factors_from_order(14, 2, 15))"""),

md("""## Step 3: the prepared modular multiplication for N = 15

The quantum part needs the gate U that multiplies a 4-qubit register by a mod 15: U|x⟩ = |a·x mod 15⟩. Its eigenphases are s/r for s = 0, 1, ..., r − 1, so phase estimation on U reveals r.

**Read this before you use it.** The function `c_amod15(a, power)` below is **prepared for you and hard-coded for N = 15**: for each allowed a it is a fixed pattern of controlled swaps (`cswap`, new in qsim 1.8.0) and controlled X gates that happens to permute the residues correctly. It is not a general method. A real run of Shor's algorithm on a large N builds modular multiplication from quantum adders, from a and N alone, and that circuit is far larger; it is the main cost of the algorithm.

Run the cell: with the control qubit set to 1, the register value x becomes 7·x mod 15."""),
code("A_VALUES = (2, 4, 7, 8, 11, 13)   # the values of a that c_amod15 supports\n\n\n" + PREPARED + '''


qc_test = QuantumCircuit(5)
qc_test.x(0)            # control qubit on
qc_test.x(1)            # register x = 1 (qubit 1 is the least significant bit of x)
for p in range(5):
    state = Statevector(qc_test)
    x = int(np.argmax(np.abs(np.asarray(state)))) >> 1
    print(f"after {p} multiplications by 7: x = {x:2d}    (7^{p} mod 15 = {pow(7, p, 15)})")
    qc_test.compose(c_amod15(7, 1), inplace=True)'''),

md("""## Step 4: the order-finding circuit

This is phase estimation (Module 1) with U = multiplication by a mod 15. The circuit has t **counting qubits** (0 to t − 1) and a 4-qubit **register** (qubits t to t + 3):

1. an X on qubit t, so the register holds |1⟩;
2. an H on every counting qubit;
3. counting qubit k controls U raised to the power 2ᵏ: `qc.compose(c_amod15(a, 2**k), qubits=[k] + list(range(t, t + 4)), inplace=True)`;
4. the inverse QFT on the counting qubits: `qc.compose(qft_dagger(t), qubits=list(range(t)), inplace=True)`;
5. measure counting qubit k into classical bit k.

Why the register starts in |1⟩: |1⟩ is not an eigenvector of U, but it is an equal superposition of r eigenvectors, with eigenphases s/r. Each shot of phase estimation then reads one of them at random: the result m gives m/2ᵗ ≈ s/r for a random s.

**Your task:** write `order_finding(a, t)`."""),
code("""def order_finding(a, t):
    \"\"\"Order finding for N = 15: t counting qubits, a 4-qubit register (qubits t to t + 3) starting in |1>.\"\"\"
    qc = QuantumCircuit(t + 4, t)
    # YOUR CODE HERE
    raise NotImplementedError("Complete order_finding() first.")
    return qc"""),

md("""## Step 5: run it for a = 7

With 8 counting qubits, 2⁸ = 256 is a multiple of r = 4, so every phase s/4 is exact and only r results appear, each about a quarter of the time. The cell prints each result m, its phase m/256 and the fraction it equals. The lab check asks which four results appear."""),
code("""t = 8
counts = sim.run(order_finding(7, t), shots=1000).result().get_counts()
for key in sorted(counts):
    m = int(key, 2)
    print(f"result {key} = m = {m:3d}   shots {counts[key]:4d}   m/256 = {m / 2 ** t:.4f} = {Fraction(m, 2 ** t)}")
plot_histogram(counts, title="order finding for a = 7, 8 counting qubits")"""),

md("""## Step 6: read r with continued fractions

Each result gives a fraction m/2ᵗ ≈ s/r. `Fraction(m, 2**t).limit_denominator(N)` finds the closest fraction with a denominator of at most N, which is s/r in lowest terms. Its denominator is r, **unless s and r share a factor**: for m = 128, 128/256 = 1/2, whose denominator 2 is not the order of 7 (7² mod 15 = 4). So check each candidate with `pow(a, r, N) == 1`, and try the next result if it fails. For a real N with no exact phases, the same method works on the results near s/r.

**Your task:** write `period_from_counts(counts, t, a, N)`: go through the results, most frequent first; for each, take the denominator of `Fraction(int(key, 2), 2**t).limit_denominator(N)`; return the first one with `pow(a, r, N) == 1`. Return `None` if no result gives it."""),
code("""def period_from_counts(counts, t, a, N):
    \"\"\"The order r read from measured results with continued fractions, or None.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete period_from_counts() first.")"""),
code("""for key in sorted(counts):
    m = int(key, 2)
    f = Fraction(m, 2 ** t).limit_denominator(15)
    print(f"m = {m:3d}:  {f}  denominator {f.denominator}   7^{f.denominator} mod 15 = {pow(7, f.denominator, 15)}")
r = period_from_counts(counts, t, 7, 15)
print("order found:", r, "   factors:", factors_from_order(7, r, 15))"""),

md("""## Step 7: factor 15 for every allowed a

The cell runs the whole algorithm, quantum order finding followed by the classical steps, for each a. With a = 4 or 11 the order is 2; with the others it is 4. Every a here gives the factors 3 and 5."""),
code("""for a in (2, 4, 7, 8, 11, 13):
    c = sim.run(order_finding(a, 8), shots=200).result().get_counts()
    r = period_from_counts(c, 8, a, 15)
    print(f"a = {a:2d}: {len(c)} different results,  order r = {r},  factors {factors_from_order(a, r, 15)}")"""),

md("""## Step 8: your personal check

Open the **Module 3 lab check** in Canvas. Question 1 shows your own A (one of 2, 4, 7, 8, 11, 13) and seed SEED. Type them below and run the next two cells. The check cell tests `order()`, `factors_from_order()`, `order_finding()` (every a, 3 to 6 counting qubits) and `period_from_counts()`. Only if every test passes does it print your **verification value**: the number of shots, out of 4,000, in which the course's reference order-finding circuit for A with 8 counting qubits gives a result whose continued fraction has the true order as its denominator, run with your seed. Type it into Canvas."""),
code("""A = 0         # your number from Canvas, for example 7
SEED = 0      # your seed from Canvas, for example 512"""),
code(CHECK + '''

passed, messages, value = check_l2_module3(order, factors_from_order, order_finding, period_from_counts, A, SEED)
for m in messages:
    print(m)
if passed:
    print("\\nAll tests passed. Your verification value is", value)
else:
    print("\\nNot yet: fix the item above and run this cell again.")'''),

md("""## What you should notice

- Factoring reduces to order finding: an even order r with a^(r/2) ≠ −1 mod N gives the factors gcd(a^(r/2) ± 1, N).
- Order finding is phase estimation of "multiply by a mod N", started in |1⟩; each shot gives m/2ᵗ ≈ s/r for a random s.
- Continued fractions turn m/2ᵗ into s/r; when s shares a factor with r, the denominator is too small, so check it and use another shot.
- The modular multiplication here was hard-coded for N = 15. For a real N it must be built from arithmetic circuits, and that, with error correction, is why breaking RSA-2048 needs a large fault-tolerant quantum computer that does not exist today."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl2m3c{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / NAME)
shutil.copyfile(ROOT / "content" / "level1" / "qsim.py", OUT / "qsim.py")
print("wrote", OUT / NAME, "and synced qsim.py")
