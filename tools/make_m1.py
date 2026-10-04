"""Build the Level 1 Module 1 browser notebook (JupyterLite, Pyodide kernel) from checks/m1_check.py."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level1"
CHECK = (ROOT / "checks" / "m1_check.py").read_text().split('"""', 2)[2].lstrip()
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md("""# Module 1 lab: state vectors and measurement

**Quantum Computing Foundations · Module 1 · about 75 minutes** · notebook version 2026-10-04

In this lab you write three small functions that every later module uses:

1. `probabilities(state)`: the probability of each measurement result;
2. `normalize(state)`: turns any list of amplitudes into a valid quantum state;
3. `simulate(state, shots, seed)`: measures the state many times, the way a quantum computer does.

Then you use them on your own personal state from Canvas. The **Module 1 lab check** asks for results from this notebook, so keep it open in a second tab. Run each code cell with **Shift + Enter**, in order."""),
md("""## Step 0: start Python

Run the cell below. The first time, Python can take up to a minute to start in your browser."""),
code("""import numpy as np

print("Python is running in your browser. NumPy version", np.__version__)"""),
md("""## Step 1: a qubit state is a vector

A one-qubit state |ψ⟩ = α|0⟩ + β|1⟩ is stored as a vector of two **amplitudes**, `[α, β]`. The two basis states are

- |0⟩ = `[1, 0]`
- |1⟩ = `[0, 1]`

and any other state is a combination of them. Python writes the imaginary unit i as `1j`. Run the cell."""),
code("""ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)

plus = (ket0 + ket1) / np.sqrt(2)          # the state (|0> + |1>)/√2
psi = 0.6 * ket0 + 0.8j * ket1              # the state from Module 0

print("|+>  =", np.round(plus, 3))
print("psi  =", psi)"""),
md("""## Step 2: probabilities from amplitudes

The probability of each result is the **size squared** of its amplitude: P(0) = |α|², P(1) = |β|². In NumPy, `np.abs(state)**2` does this for every amplitude at once.

Complete `probabilities()` so it returns that array: replace the line that starts with `raise`. Then run the cell: it prints the probabilities of the two states from Step 1."""),
code("""def probabilities(state):
    # YOUR CODE: replace the next line. Return the probability of each result, |amplitude|^2.
    raise NotImplementedError("Complete probabilities() first.")


print("P for |+>:", probabilities(plus))
print("P for psi:", probabilities(psi))"""),
md("""You should see `[0.5 0.5]` for |+⟩ and `[0.36 0.64]` for psi. The probabilities add up to 1, as they must for a valid state."""),
md("""### Your turn (not graded, about 5 minutes)

A classmate wrote `probabilities()` as `return state**2`. It gives the right answer for |+⟩. Try it on psi in the cell below: compute `psi**2` yourself and compare it with `probabilities(psi)`.

Then, in the text cell under it, explain in one or two sentences why squaring an amplitude is not the same as taking the size squared of a **complex** amplitude, and why the mistake does not show up for |+⟩. (Hint: what is (0.8i)²?) There is no check for this task; it prepares you for complex amplitudes in every later module."""),
code("""# YOUR CODE: compute psi**2 and compare it with probabilities(psi).
"""),
md("""*Your explanation:* (double-click this cell to write here)"""),
md("""## Step 3: normalization

A list of amplitudes is a valid state only if its probabilities add up to 1, that is, if the vector has **length 1**. Any other nonzero vector can be **normalized** by dividing it by its length: `state / np.linalg.norm(state)`. This keeps the ratio between the amplitudes and fixes the length.

Complete `normalize()`, then run the cell. Canvas asks for the length of `[3, 4i]` before normalizing."""),
code("""def normalize(state):
    # YOUR CODE: replace the next line. Return a new array: the state divided by its length.
    raise NotImplementedError("Complete normalize() first.")


raw = np.array([3, 4j])
print("length of [3, 4i] before:", np.linalg.norm(raw))
print("normalized:", normalize(raw))
print("probabilities:", probabilities(normalize(raw)))"""),
md("""## Step 4: one measurement

Measuring a qubit gives 0 or 1 at random, with the probabilities from Step 2. After the measurement the qubit **is** in the state it reported, so measuring it again in the same basis, with nothing done to it in between, gives the same result.

NumPy's random generator can make that choice. `rng.choice(2, p=...)` picks 0 or 1 with the probabilities you give it. The **seed** makes the random numbers repeatable, so everyone who uses the same seed gets the same results. Run the cell a few times: the result stays the same because the seed is fixed. Change the seed to see other results."""),
code("""rng = np.random.default_rng(seed=2026)

result = rng.choice(2, p=probabilities(psi))
print("one measurement of psi gave:", result)"""),
md("""## Step 5: many measurements (shots)

A quantum computer runs the same circuit many times. Each run is a **shot**, and the results are counted. Complete `simulate()` so it returns one result (0 or 1) for every shot:

```python
rng = np.random.default_rng(seed)
return rng.choice(2, size=shots, p=probabilities(state))
```

Use exactly this method. There are other correct ways to sample, but they give different shots for the same seed, and the lab check needs everyone's counts to match; the check cell tests this. Each shot stands for a fresh run: the state is prepared again and measured once, so the shots are independent of each other.

Then run the cell. It measures the normalized state `[3, 4i]` 1,000 times with seed 7. Canvas asks how many times you got 1."""),
code("""def simulate(state, shots, seed):
    # YOUR CODE: replace the next line. Make a generator from the seed, then return 'shots' results, each 0 or 1.
    raise NotImplementedError("Complete simulate() first.")


results = simulate(normalize(np.array([3, 4j])), 1000, 7)
print("first 20 results:", results[:20])
print("number of 1s in 1,000 shots:", results.sum())"""),
md("""## Step 6: why the count is not exactly 640

P(1) = 0.64, so the expected number of 1s in 1,000 shots is 640. The real count moves around 640 from one run to the next. Its typical spread (the **standard deviation**) is √(N·p·(1 − p)) = √(1000 × 0.64 × 0.36) ≈ 15.

Run the cell. It repeats the 1,000-shot experiment with 20 different seeds and compares the spread with that formula."""),
code("""state = normalize(np.array([3, 4j]))
counts = [int(simulate(state, 1000, s).sum()) for s in range(1, 21)]

print("counts of 1 for seeds 1 to 20:", counts)
print("average:", np.mean(counts))
print("spread (standard deviation):", round(np.std(counts), 1))
print("formula sqrt(N p (1-p)):", round(np.sqrt(1000 * 0.64 * 0.36), 1))"""),
md("""Most counts land within about 15 of 640, and almost all within 3 × 15 = 45. A count much farther away, such as 700, is very unlikely from shot noise alone. It does not prove that something is wrong, but it is a reason to check the setup or repeat the experiment. Later modules use this rule to judge whether a difference between two histograms is larger than shot noise usually produces."""),
md("""## Step 7: your personal state

Open the Canvas quiz **Module 1 lab check**. Question 1 gives you three numbers: **A**, **B** and **SEED**. Your personal state is A|0⟩ + iB|1⟩, normalized. Type the three numbers below in place of `None` and run the cell."""),
code("""A = None      # for example: A = 3
B = None      # for example: B = 4
SEED = None   # for example: SEED = 7

if None in (A, B, SEED):
    print("Enter A, B and SEED from Canvas first.")
else:
    mine = normalize(np.array([A, 1j * B]))
    print("your state:", np.round(mine, 4))
    print("its probabilities:", np.round(probabilities(mine), 4))"""),
md("""## Step 8: run the check cell

**Do not edit this cell.** Run it. It tests your three functions on cases you have not seen. If they all pass, it measures your personal state 1,000 times with your SEED and prints your **verification value**: the number of 1s. Type it into Canvas question 1.

If a test does not pass, read its message, fix that function, run its cell again, then run this cell again."""),
code("# CHECK CELL: do not edit\n" + CHECK + '''

ok, messages, value = check_module1(probabilities, normalize, simulate, A, B, SEED)
for m in messages:
    print(m)
if ok:
    print(f"\\nAll tests passed. Your verification value for A = {A}, B = {B}, SEED = {SEED}: {value}")
    print("Type this number into Canvas question 1.")
else:
    print("\\nNot passed yet: no verification value.")
'''),
md("""## Finish

1. Answer the five questions of the **Module 1 lab check** in Canvas and submit.
2. Take the **Module 1 quiz**.
3. Fill in the short **Module 1 time log**.

Your notebook is saved in this browser as you work. It is not saved anywhere else, so to keep a copy choose **File > Download**. You will reuse `probabilities()`, `normalize()` and `simulate()` in Module 2."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl1m1{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / "QC-L1-M1-lab-state-vectors.ipynb")
print("wrote", OUT / "QC-L1-M1-lab-state-vectors.ipynb")
