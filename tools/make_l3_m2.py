"""Build the Level 3 Module 2 browser notebook (quantum error correction: the three-qubit bit-flip repetition code
with mid-circuit syndrome measurement and feedforward correction, its limits, and saved device syndrome data).
The check code is checks/l3_m2_check.py, published next to the notebook by l3_hidden.
Device data: content/level3/l3_m2_device_run.json (tools/make_l3_m2_data.py)."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level3"
from l3_hidden import hidden_check  # noqa: E402
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}
NAME = "QC-L3-M2-lab-repetition-code.ipynb"
VERSION = "2026-10-10"


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md(f"""# Module 2 lab: the three-qubit repetition code

**Quantum Computing Advanced · Module 2 · about 95 minutes** · notebook version {VERSION}

A quantum computer cannot copy a qubit to protect it, and it cannot look at a qubit without disturbing it. Error correction gets around both: it spreads one qubit's state over several qubits, and it measures only **parities**, which reveal where an error happened but nothing about the state itself. In this lab you build the simplest quantum code, the three-qubit **bit-flip repetition code**, the way a real quantum computer runs it:

1. compare with the classical repetition code and its majority vote;
2. write `encode(qc)`: one qubit's state spread over three;
3. write `measure_syndrome(qc)`: two parities measured with two extra qubits, in the middle of the circuit;
4. write `correct(qc)`: an X gate chosen by the measured syndrome, with `if_test`;
5. measure how often the code succeeds on a noisy simulator, and find what limits it;
6. read the syndrome records of a three-round run with a real device's noise;
7. get your verification value.

The **Module 2 lab check** in Canvas asks for results from this notebook, so keep it open in a second tab. Run each code cell with **Shift + Enter**, in order.

**Layout used in every cell:** data qubits 0, 1 and 2 (the state to protect starts on qubit 0); **ancilla** qubits 3 and 4, which measure the parities; classical bits 0 and 1 hold the syndrome."""),

md("""## Step 0: start Python

Run the cell below. The first time, Python can take up to a minute to start in your browser. It also loads the saved device run used in Step 7."""),
code("""import json
import numpy as np
import matplotlib.pyplot as plt
import qsim
from qsim import (QuantumCircuit, Statevector, AerSimulator, NoiseModel, pauli_error, depolarizing_error,
                  ReadoutError, simulate_density_matrix, partial_trace)

run = json.load(open("l3_m2_device_run.json"))
sim = AerSimulator()
print("Ready. qsim", qsim.__version__, "and NumPy", np.__version__)
print("Device run:", run["backend"], "-", run["kind"])"""),

md("""## Step 1: the classical repetition code

To send one bit over a noisy line that flips each bit with probability p, send it three times and take the **majority vote**. The vote is wrong only if two or three copies flip:

P_L = 3p²(1 − p) + p³ = 3p² − 2p³.

The cell checks this formula with 100,000 simulated messages and plots P_L against p. *Predict first:* for which p does repetition help, that is, P_L < p?"""),
code("""rng = np.random.default_rng(1)
p = 0.10
flips = rng.random((100_000, 3)) < p            # True where a copy is flipped
vote_wrong = flips.sum(axis=1) >= 2
print(f"p = {p}: majority vote wrong in {vote_wrong.mean():.4f} of the messages; formula 3p^2 - 2p^3 = {3*p**2 - 2*p**3:.3f}")

ps = np.linspace(0, 1, 101)
plt.figure(figsize=(6, 3.3))
plt.plot(ps, ps, "k--", lw=1, label="one bit: p")
plt.plot(ps, 3 * ps**2 - 2 * ps**3, label="three bits and a vote: 3p² − 2p³")
plt.xlabel("flip probability p"); plt.ylabel("error probability"); plt.legend(fontsize=8); plt.tight_layout(); plt.show()"""),
md("""**What to notice.** For p = 0.1 the vote is wrong 2.8% of the time instead of 10%. The curves cross at p = 0.5: below it, repetition always helps, and the smaller p is, the more it helps, because the error rate falls from p to about 3p². That is the whole idea of error correction: redundancy turns a small error rate into a much smaller one. The lab check asks for the formula's value at p = 0.1.

A qubit cannot simply be copied (the no-cloning theorem), and reading the three copies would destroy a superposition. The next steps show how the quantum code avoids both problems."""),

md("""## Step 2: encoding

The quantum repetition code does not copy the state α|0⟩ + β|1⟩; it **entangles** it with two more qubits:

α|0⟩ + β|1⟩ → α|000⟩ + β|111⟩.

Two CNOTs do it, both controlled by qubit 0. A bit flip on any one qubit now leaves a trace that can be found without learning α or β.

**Your task:** write `encode(qc)`, which adds the two CNOTs to the circuit `qc` (qubits 1 and 2 start in |0⟩). The cell after it checks the state for an input RY(1.1)|0⟩."""),
code("""def encode(qc):
    \"\"\"Add gates to qc so that the state of qubit 0 is spread over qubits 0, 1 and 2: a|000> + b|111>.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete encode() first.")"""),
code("""qc = QuantumCircuit(5, 2)
qc.ry(1.1, 0)
encode(qc)
psi = Statevector(qc)
print("Nonzero amplitudes (labels are q4 q3 q2 q1 q0):")
for label, amp in psi.probabilities_dict().items():
    print(f"  |{label}>  probability {amp:.4f}")
a, b = np.cos(0.55), np.sin(0.55)
print(f"expected: |00000> with {a**2:.4f} and |00111> with {b**2:.4f}")

copies = np.kron(np.kron([a, b], [a, b]), [a, b])        # three copies (a|0> + b|1>)^3: what cloning would give
print(f"overlap of the encoded state with three copies: {abs(np.vdot(copies, psi.data[:8]))**2:.4f}")"""),
md("""**What to notice.** Only |000⟩ and |111⟩ appear (the ancillas 3 and 4 stay in |0⟩). The encoded state is not three copies of the qubit: its overlap with (α|0⟩ + β|1⟩)⊗3 is well below 1. Measuring any one data qubit now would give 0 or 1 at random and collapse the whole state, so the code must never measure the data qubits directly."""),

md("""## Step 3: measuring the syndrome

The code asks two yes-or-no questions: *do qubits 0 and 1 agree?* and *do qubits 1 and 2 agree?* These are the **parities** Z₀Z₁ and Z₁Z₂. For α|000⟩ + β|111⟩ both answers are "agree" whatever α and β are, so measuring the parities does not disturb the encoded state. A flip of one qubit changes one or both answers: the pair of answers is the **syndrome**.

An ancilla measures a parity: CNOTs from the two data qubits onto the ancilla leave it in |1⟩ exactly when the two data qubits differ. The ancilla is then measured, in the middle of the circuit, and **reset** to |0⟩ so it can be used again.

**Your task:** write `measure_syndrome(qc)`:
- ancilla 3 gets CNOTs from data qubits 0 and 1, and is measured into classical bit 0 (s₁ = parity of qubits 0 and 1);
- ancilla 4 gets CNOTs from data qubits 1 and 2, and is measured into classical bit 1 (s₂ = parity of qubits 1 and 2);
- then reset both ancillas with `qc.reset(3)` and `qc.reset(4)`.

The cell after it puts an X error on each data qubit in turn and prints the syndrome. In a counts key the classical bits read right to left: the key '01' means s₁ = 1, s₂ = 0."""),
code("""def measure_syndrome(qc):
    \"\"\"Measure the parities of data qubits (0, 1) and (1, 2) with ancillas 3 and 4 into classical bits 0 and 1,
    then reset the ancillas. Do not measure the data qubits.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete measure_syndrome() first.")"""),
code("""theta = 1.1
for error in [None, 0, 1, 2]:
    qc = QuantumCircuit(5, 2)
    qc.ry(theta, 0)
    encode(qc)
    if error is not None:
        qc.x(error)
    measure_syndrome(qc)
    counts = sim.run(qc, shots=100, seed_simulator=3).result().get_counts()
    data = partial_trace(simulate_density_matrix(qc), [3, 4]).data       # the data qubits after the syndrome
    good = np.zeros(8, dtype=complex)
    k = 0 if error is None else 2 ** error
    good[k], good[7 - k] = np.cos(theta / 2), np.sin(theta / 2)          # the encoded state with that one flip
    kept = np.vdot(good, data @ good).real
    where = "no error      " if error is None else f"X on qubit {error}"
    print(f"{where}: counts {counts}   superposition kept: {kept:.4f}")"""),
md("""**What to notice.** The syndrome is the same in all 100 shots: it depends only on the error, not on α and β. No error gives (s₁, s₂) = (0, 0); an X on qubit 0 gives (1, 0), the key '01'; qubit 1 gives (1, 1); qubit 2 gives (0, 1), the key '10'. Each single flip has its own syndrome, and the superposition survives the measurement: the fidelity stays 1. The syndrome tells you **where** the error is and nothing about **what state** is stored."""),

md("""## Step 4: correcting with feedforward

Now act on the syndrome, in the same shot: apply X to the qubit the syndrome points to. A gate that depends on a measurement result made earlier in the circuit is **feedforward**; in Qiskit and qsim it is written with `if_test`, as in the teleportation circuit of Level 2:

```python
with qc.if_test((qc.clbits[0], 1)):        # runs only when classical bit 0 is 1
    with qc.if_test((qc.clbits[1], 1)):    # ... and classical bit 1 is 1
        qc.x(1)
```

`qc.clbits[0]` is classical bit 0, the form in Qiskit's documentation of `if_test`. Qiskit and qsim also accept the plain index, `qc.if_test((0, 1))`, but the explicit bit is the portable choice.

**Your task:** write `correct(qc)` for all three single-flip syndromes: (s₁, s₂) = (1, 0) → X on qubit 0; (1, 1) → X on qubit 1; (0, 1) → X on qubit 2; (0, 0) → nothing. Nest one `if_test` inside another to test both bits (`with qc.if_test((qc.clbits[1], 0)):` tests for bit 1 being 0)."""),
code("""def correct(qc):
    \"\"\"Apply X to the data qubit that the syndrome in classical bits 0 and 1 points to.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete correct() first.")"""),
code("""def one_round(theta, errors=()):
    \"\"\"Prepare RY(theta)|0>, encode, an id gate on each data qubit (where noise models put their errors),
    the X errors listed, one syndrome measurement and the correction.\"\"\"
    qc = QuantumCircuit(5, 2)
    qc.ry(theta, 0)
    encode(qc)
    for d in (0, 1, 2):
        qc.id(d)
    for d in errors:
        qc.x(d)
    measure_syndrome(qc)
    correct(qc)
    return qc


def fidelity(qc, theta, noise_model=None):
    \"\"\"Fidelity of the data qubits with the encoded state cos(theta/2)|000> + sin(theta/2)|111>.\"\"\"
    data = partial_trace(simulate_density_matrix(qc, noise_model), [3, 4]).data
    good = np.zeros(8)
    good[0], good[7] = np.cos(theta / 2), np.sin(theta / 2)
    return float(np.real(good @ data @ good))


theta = 1.1
for errors in [(), (0,), (1,), (2,), (0, 1), (1, 2)]:
    print(f"X errors on {str(list(errors)):8s} -> fidelity after correction {fidelity(one_round(theta, errors), theta):.4f}")
print(f"sin^2(theta) = {np.sin(theta)**2:.4f}")
one_round(theta).draw()"""),
md("""**What to notice.** Every single flip is corrected exactly: fidelity 1. Two flips are "corrected" the wrong way: the syndrome of flips on qubits 0 and 1 is the same as that of one flip on qubit 2, so the correction completes a flip of all three qubits. That is a **logical** X, and the fidelity drops to sin²θ (it would be 0 for a basis state). A code with three qubits corrects any one bit flip but not two: its **distance against bit flips** is 3. As a full quantum code its distance is only 1, because a single Z error is already an undetected logical error (Step 6)."""),

md("""## Step 5: the code on a noisy simulator

Now let the noise choose the errors. The noise model flips each data qubit with probability p (on the `id` gate after encoding) and misreads each syndrome bit with probability q. `simulate_density_matrix` follows every measurement outcome and its `if_test` branch exactly, so there are no shots and no shot noise here.

Call one round a **success** when the data qubits are back in the encoded state; for θ = 0 the probability of failure is 1 − fidelity. Without the code, a qubit with flip probability p fails with probability p."""),
code("""def noise(p, q=0.0, p_cx=0.0):
    \"\"\"Flip probability p on each data qubit, readout error q on each ancilla, depolarizing error p_cx on every CNOT.\"\"\"
    nm = NoiseModel()
    if p > 0:
        for d in (0, 1, 2):
            nm.add_quantum_error(pauli_error([("X", p), ("I", 1 - p)]), ["id"], [d])
    if q > 0:
        for a in (3, 4):
            nm.add_readout_error(ReadoutError([[1 - q, q], [q, 1 - q]]), [a])
    if p_cx > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p_cx, 2), ["cx"])
    return nm


def failure(p, q=0.0, p_cx=0.0):
    return 1 - fidelity(one_round(0.0), 0.0, noise(p, q, p_cx))


print("   p     code fails (q = 0)   3p^2 - 2p^3   no code")
for p in (0.01, 0.05, 0.10, 0.20, 0.30):
    print(f"  {p:.2f}       {failure(p):.4f}            {3*p**2 - 2*p**3:.4f}       {p:.2f}")
print(f"\\np = 0.05 with a 2% syndrome readout error: the code fails with probability {failure(0.05, 0.02):.3f}")

ps = np.linspace(0.005, 0.30, 30)
plt.figure(figsize=(6, 3.3))
plt.plot(ps, ps, "k--", lw=1, label="no code")
for q in (0.0, 0.02, 0.05):
    plt.plot(ps, [failure(p, q) for p in ps], label=f"code, syndrome readout error q = {q}")
plt.xlabel("flip probability p"); plt.ylabel("failure probability of one round"); plt.legend(fontsize=8); plt.tight_layout(); plt.show()"""),
md("""**What to notice.** With perfect syndromes the code reproduces the classical formula exactly: one round fails with probability 3p² − 2p³. But the syndrome is measured by noisy hardware too. A misread syndrome bit makes the correction put a flip **in**, so with q = 0.02 and p = 0.05 the round fails 4.7% of the time, hardly better than no code at all (5%). The lab check asks for this number. Real codes therefore measure the syndrome **repeatedly** and trust a change only if it persists, which is what Step 6's data shows."""),

md("""## Step 6: two limits of this code

**(a) Phase flips.** The code watches for X errors only. A Z error on any qubit turns α|000⟩ + β|111⟩ into α|000⟩ − β|111⟩: both parities still agree, the syndrome is (0, 0), and nothing is corrected. The first cell shows it for |+⟩ encoded (θ = π/2), and then the **phase-flip code**: H gates after encoding (and before the syndrome) turn Z errors into X errors, so the same syndrome circuit catches them. Shor's nine-qubit code combines both ideas to correct any single-qubit error.

**(b) Noisy gates.** The syndrome circuit itself uses four CNOTs, and each can cause errors. The second cell adds a depolarizing error of 1% to every CNOT (including the two of the encoder) and finds the **break-even point**: the flip probability p above which the code beats an unprotected qubit (below it, the CNOT errors cost more than the code removes)."""),
code("""theta = np.pi / 2                                   # |+> encoded
qc = QuantumCircuit(5, 2)
qc.ry(theta, 0); encode(qc)
qc.z(1)                                             # a phase flip on data qubit 1
measure_syndrome(qc); correct(qc)
print("bit-flip code, Z error on qubit 1: syndrome counts", sim.run(qc, shots=100, seed_simulator=4).result().get_counts(),
      f"  fidelity {fidelity(qc, theta):.4f}")

qc = QuantumCircuit(5, 2)
qc.ry(theta, 0); encode(qc)
for d in (0, 1, 2):
    qc.h(d)                                         # phase-flip code: |+++> and |--->
qc.z(1)                                             # the same phase flip
for d in (0, 1, 2):
    qc.h(d)                                         # back to the Z basis: the Z error is now an X error
measure_syndrome(qc); correct(qc)
print("phase-flip code, Z error on qubit 1: syndrome counts", sim.run(qc, shots=100, seed_simulator=4).result().get_counts(),
      f"  fidelity {fidelity(qc, theta):.4f}")"""),
code("""def break_even(p_cx, q=0.0):
    \"\"\"The p where failure(p) = p, by bisection (failure(p) > p below it, < p above it ... or the other way).\"\"\"
    lo, hi = 1e-4, 0.45
    f = lambda p: failure(p, q, p_cx) - p
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


pb = break_even(0.01)
print(f"With 1% CNOT errors: at p = 0.002 the code fails with {failure(0.002, 0, 0.01):.4f} (no code: 0.002)")
print(f"                     at p = 0.10  the code fails with {failure(0.10, 0, 0.01):.4f} (no code: 0.10)")
print(f"Break-even flip probability: p = {pb:.3f}")"""),
md("""**What to notice.** (a) The bit-flip code is blind to the phase flip: syndrome 00, fidelity 0. The phase-flip code catches the same error and restores the state. (b) With noisy CNOTs the code adds errors of its own: for small p it does **worse** than an unprotected qubit, and it only helps once p is above about 0.033, the break-even point printed above (the lab check asks for it). The extra qubits and gates must be good enough for correction to win. This is the idea behind a **threshold**: if every operation is good enough, larger codes do better and better; if not, adding qubits makes things worse. The 0.033 here belongs to this simple model; real thresholds depend on the code, the operations and the noise."""),

md("""## Step 7: three rounds with a real device's noise

The file loaded in Step 0 holds the syndrome records of the repetition code run for **three rounds** in a row, 4,000 shots for logical |0⟩ = |000⟩ and 4,000 for logical |1⟩ = |111⟩, with the noise model of **FakePittsburgh** (ibm_pittsburgh's calibration of 17 April 2026, on five neighbouring qubits). It is a simulation with real device noise, the way IBM's own hardware runs are usually rehearsed. No correction is applied during the run: each round only records (s₁, s₂), and the data qubits are measured at the end.

Each key reads `r1 r2 r3 data`, for example `00 01 00 000`: rounds 1 to 3, then data qubits 0, 1, 2. Unlike the counts in Step 3, these keys are written in reading order, left to right: each round shows s₁ then s₂, so `01` here means s₁ = 0, s₂ = 1 (a flip on qubit 2), and the data part shows qubit 0 first. The cell counts how often each round reports an error, finds **detection events** (a syndrome that differs from the round before; round 0 is 00), and compares two ways of reading the final data."""),
code("""print("Physical qubits:", run["physical_qubits"], "  CNOT (CZ) errors:", run["cz_error"])
for logical in (0, 1):
    counts = run[f"counts_logical_{logical}"]
    shots = sum(counts.values())
    rounds = {r: 0 for r in range(3)}
    events = {r: 0 for r in range(3)}
    flipped = 0
    wrong_vote = 0
    for key, n in counts.items():
        *syn, data = key.split()
        prev = "00"
        for r, s in enumerate(syn):
            rounds[r] += n * (s == "00")
            events[r] += n * (s != prev)
            prev = s
        bits = [int(b) for b in data]
        flipped += n * sum(b != logical for b in bits)
        wrong_vote += n * ((sum(bits) >= 2) != bool(logical))
    print(f"\\nlogical {logical}: {shots} shots")
    print("  shots with syndrome 00 in rounds 1, 2, 3:", [rounds[r] for r in range(3)])
    print("  detection events in rounds 1, 2, 3:     ", [events[r] for r in range(3)])
    print(f"  a single data qubit read wrong: {flipped / (3 * shots):.4f} per qubit")
    print(f"  majority vote of the three data qubits wrong: {wrong_vote / shots:.4f}")

top = sorted(run["counts_logical_0"].items(), key=lambda kv: -kv[1])[:8]
print("\\nmost frequent records for logical 0:", top)"""),
md("""**What to notice.** About 98% of the shots report syndrome 00 in each round, and the detection events grow a little from round to round as errors build up. Most nonzero syndromes appear in **one round only** and are gone in the next: this pattern most likely comes from measurement errors on the ancillas, not real data errors. A real flip of a data qubit would most likely make the syndrome change once and then **stay** changed. These patterns suggest a cause; they do not prove it. That is why decoders for real codes look at detection events across rounds rather than at one round's syndrome. In the final data, a single data qubit reads wrong about 1% of the time, but the majority vote of the three is wrong only about once in 4,000 shots: the code works on these qubits for bit flips. The run stores the basis states |000⟩ and |111⟩ only and applies no correction, so it tests bit-flip protection of a stored 0 or 1, not the protection of a superposition. The lab check asks how many logical-0 shots show syndrome 00 in round 1.

This is a simulation of a device, not a real run. The course team may add a real run of the same circuits to this file; the analysis stays the same."""),

md("""## Step 8: beyond the repetition code

You will not build these here, but they are where the repetition code leads:

- **Shor's nine-qubit code** (1995) puts three phase-flip blocks around three bit-flip blocks and corrects any error on one qubit.
- **Stabilizer codes** describe a code by the Pauli operators it measures (here Z₀Z₁ and Z₁Z₂). The **Steane code** stores one qubit in seven and also corrects any single error.
- The **surface code** places qubits on a square grid and measures only nearby parities; its distance d grows with the grid. In 2024 Google ran surface codes of distance 3, 5 and 7 below the **threshold**: each step up in distance roughly halved the logical error rate (Nature 638, 920, 2025).
- IBM plans **qLDPC** codes, such as the [[144, 12, 12]] "gross" code (12 logical qubits in 144 data qubits), for its fault-tolerant system Starling, announced for 2029.

The reading page has short videos on each of these."""),

md("""## Step 9: your personal check

Open the **Module 2 lab check** in Canvas. Question 1 shows your own THETA (0.30 to 2.80), P (0.01 to 0.30) and Q (0.00 to 0.10). Type them below and run the next two cells. The check cell tests `encode()`, `measure_syndrome()` and `correct()`. Only if every test passes does it print your **verification value**: 1,000 times the fidelity after one round of the code, as in Step 5, for the state RY(THETA)|0⟩ with flip probability P and syndrome readout error Q, rounded to a whole number."""),
code("""THETA = 0.0    # your number from Canvas, for example 1.25
P = 0.0        # for example 0.10
Q = 0.0        # for example 0.05"""),
code(hidden_check(['l3_m2_check'], 'check_l3_module2', '''

passed, messages, value = check_l3_module2(encode, measure_syndrome, correct, THETA, P, Q)
for m in messages:
    print(m)
if passed:
    print("\\nAll tests passed. Your verification value is", value)
else:
    print("\\nNot yet: fix the item above and run this cell again.")''')),

md("""## What you should notice

- The repetition code turns a flip probability p into about 3p² by measuring parities, not the data: the syndrome says where a flip happened and nothing about the stored state.
- Mid-circuit measurement, reset and feedforward (`if_test`) are the hardware features that let a quantum computer run a code in real time.
- One code corrects one kind of error: the bit-flip code is blind to phase flips. Shor's code and the stabilizer codes that followed handle both.
- The code's own operations are noisy: misread syndromes and noisy CNOTs can make a code worse than no code. Repeated syndrome rounds and a low enough error rate per operation (the threshold) are what make error correction work.

**Next in Canvas:** the lab check, the quiz, your capstone choice and the time log."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl3m2c{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / NAME)
print("wrote", OUT / NAME)
