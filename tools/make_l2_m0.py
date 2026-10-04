"""Build the Level 2 Module 0 browser notebook (dynamic circuits and teleportation) from checks/l2_m0_check.py."""
import shutil
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level2"
CHECK = (ROOT / "checks" / "l2_m0_check.py").read_text().split('"""', 2)[2].lstrip()
META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
        "language_info": {"name": "python"}}
NAME = "QC-L2-M0-lab-dynamic-circuits-teleportation.ipynb"
VERSION = "2026-10-04"


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md(f"""# Module 0 lab: dynamic circuits and teleportation

**Quantum Computing Intermediate · Module 0 · about 45 minutes** · notebook version {VERSION}

In Level 1 every measurement came at the end of the circuit. Real algorithms, and real quantum computers, can also measure **in the middle** of a circuit and use the result straight away to choose the next gate. This is called a **dynamic circuit**, and the step "use the result to choose a gate" is **classical feedforward**.

In this lab you:

1. see what a measurement in the middle of a circuit does;
2. write `active_reset()`, which puts a qubit back to |0⟩ with a measurement and feedforward;
3. complete quantum **teleportation** by writing Bob's two corrections, `bob_corrections()`;
4. check that the state really arrives, and that Bob learns nothing before Alice's two bits reach him.

The **Module 0 lab check** in Canvas asks for results from this notebook, so keep it open in a second tab. Run each code cell with **Shift + Enter**, in order."""),
md("""## Step 0: start Python

Run the cell below. The first time, Python can take up to a minute to start in your browser. The course simulator, qsim, keeps Qiskit's names; version 1.5.0 adds Qiskit's `if_test` for feedforward and `partial_trace`."""),
code("""import numpy as np
import qsim
from qsim import (QuantumCircuit, Statevector, AerSimulator, plot_histogram,
                  simulate_density_matrix, partial_trace, state_fidelity)

sim = AerSimulator(seed_simulator=7)


def bloch_of(rho):
    \"\"\"Prepared for you: the Bloch vector (x, y, z) of a one-qubit density matrix.\"\"\"
    r = np.asarray(rho)
    return np.round([2 * r[0, 1].real, 2 * r[1, 0].imag, (r[0, 0] - r[1, 1]).real], 4) + 0.0


print("Ready. qsim", qsim.__version__, "and NumPy", np.__version__)"""),
md("""## Step 1: a measurement in the middle of a circuit

Two Hadamard gates in a row cancel: H·H = I, so the first circuit always gives 0. The second circuit measures **between** the two H gates. The measurement gives 0 or 1 at random, and the qubit is left in |0⟩ or |1⟩, not in |+⟩. The second H then turns either of those into an equal superposition again.

Run the cell and compare the two histograms. The second circuit records two bits: c0 from the middle measurement (right) and c1 from the final one (left)."""),
code("""no_mid = QuantumCircuit(1, 1)
no_mid.h(0)
no_mid.h(0)
no_mid.measure(0, 0)

with_mid = QuantumCircuit(1, 2)
with_mid.h(0)
with_mid.measure(0, 0)      # measure in the middle
with_mid.h(0)
with_mid.measure(0, 1)

print(with_mid.draw())
c_no = sim.run(no_mid, shots=1000).result().get_counts()
c_mid = sim.run(with_mid, shots=1000).result().get_counts()
print("no middle measurement:  ", c_no)
print("with middle measurement:", c_mid)
plot_histogram(c_mid, title="H, measure, H, measure")"""),
md("""**What to notice.** Without the middle measurement the two paths to |1⟩ cancel (interference) and every shot gives 0. Measuring in the middle destroys the superposition, so the cancellation no longer happens and all four results appear, about 250 times each."""),
md("""## Step 2: classical feedforward, and an active reset

In Qiskit 2, and in qsim, feedforward is written as a block:

```python
qc.measure(0, 0)
with qc.if_test((0, 1)):     # only in shots where classical bit 0 was measured as 1
    qc.x(0)
```

The gates inside the block run only in the shots where the classical bit has the value you give. Real IBM computers run such blocks during the circuit, in a few microseconds.

**Your task:** write `active_reset(qc, qubit, clbit)`. It measures `qubit` into `clbit` and, if the result was 1, flips the qubit back with X, so the qubit always ends in |0⟩. Use `qc.if_test`, not `qc.reset()`: this is how a reset can be built from a measurement and feedforward. The function changes `qc` in place; it does not need to return anything."""),
code("""def active_reset(qc, qubit, clbit):
    \"\"\"Measure qubit into clbit; if the result is 1, apply X so the qubit ends in |0>.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete active_reset() first.")"""),
md("""Test it: put the qubit in |+⟩, reset it actively, then measure it again into a second bit. The first result (c0, right) is random; the second (c1, left) should be 0 in every shot."""),
code("""qc = QuantumCircuit(1, 2)
qc.h(0)
active_reset(qc, 0, 0)
qc.measure(0, 1)
print(qc.draw())
counts = sim.run(qc, shots=1000).result().get_counts()
print(counts)
print("shots where the qubit ended in 1:", sum(v for k, v in counts.items() if k[0] == "1"))"""),
md("""## Step 3: quantum teleportation

Alice has a qubit in a state |ψ⟩ = α|0⟩ + β|1⟩ that she does not know, and she wants Bob to have it. She cannot copy it (no quantum state can be copied), and she cannot measure it to learn α and β. Teleportation moves it anyway, using **one shared Bell pair and two classical bits**:

1. **Share a Bell pair.** Qubits 1 (Alice) and 2 (Bob) start in (|00⟩ + |11⟩)/√2.
2. **Alice's Bell measurement.** Alice applies CNOT from qubit 0 to qubit 1 and H to qubit 0, then measures both: c0 from qubit 0 and c1 from qubit 1. Each of the four results comes up with probability 1/4, whatever |ψ⟩ is.
3. **Bob's corrections.** Bob's qubit is now |ψ⟩ with up to two known errors. After Alice's two bits arrive: if **c1 = 1** Bob applies **X**, then if **c0 = 1** he applies **Z**. His qubit is then exactly |ψ⟩.

The circuit for steps 1 and 2 is prepared for you below. Alice's qubit 0 ends up measured: the state is now only on Bob's side, so nothing was copied.

**Your task:** write `bob_corrections(qc)`, which adds the two corrections of step 3 to qubit 2 with `qc.if_test`."""),
code('''def prepare(theta, phi=0.0):
    """The state to send: ry(theta) then rz(phi) on qubit 0."""
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    if phi:
        qc.rz(phi, 0)
    return qc


def teleport_circuit(prep, corrections, measure_bob=False, after=None):
    """Prepared for you. Alice holds qubit 0 (the state) and qubit 1; Bob holds qubit 2.
    c0 and c1 hold Alice's two results. A one-qubit circuit 'after' is applied to Bob's qubit at the end;
    with measure_bob=True, Bob's qubit is then measured into c2."""
    qc = QuantumCircuit(3, 3 if measure_bob else 2)
    qc.compose(prep, qubits=[0], inplace=True)
    qc.barrier()
    qc.h(1)
    qc.cx(1, 2)                  # 1. Alice and Bob share the Bell pair (|00> + |11>)/sqrt(2)
    qc.barrier()
    qc.cx(0, 1)
    qc.h(0)                      # 2. Alice's Bell measurement
    qc.measure(0, 0)
    qc.measure(1, 1)
    qc.barrier()
    if corrections is not None:
        corrections(qc)          # 3. Bob's corrections
    if after is not None:
        qc.compose(after, qubits=[2], inplace=True)
    if measure_bob:
        qc.measure(2, 2)
    return qc


def bob_state(qc):
    """Prepared for you: Bob's qubit (qubit 2) at the end, averaged over Alice's results."""
    return partial_trace(simulate_density_matrix(qc), [0, 1])
'''.lstrip()),
code("""def bob_corrections(qc):
    \"\"\"Add Bob's corrections to qubit 2: X if c1 == 1, then Z if c0 == 1 (use qc.if_test).\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete bob_corrections() first.")"""),
md("""Draw the whole circuit for a state of your choice. The `IF` boxes are your corrections."""),
code("""psi = prepare(1.0, 0.6)
qc = teleport_circuit(psi, bob_corrections)
print(qc.draw())"""),
md("""## Step 4: does the state arrive?

Three checks, all with the state ry(1.0) then rz(0.6):

- **Alice's results.** Each of the four results 00, 01, 10 and 11 should come up about 250 times in 1,000 shots. Canvas asks for the exact probability of each one.
- **Bob's state.** The fidelity between the state Alice started with and Bob's final qubit should be 1.000000, and their Bloch vectors equal.
- **Undo test.** If Bob applies the *inverse* of the preparation to his qubit and measures, he should get 0 in every shot. This is how teleportation is checked on a real computer, where the state itself cannot be looked at. Canvas asks for this count."""),
code("""counts = sim.run(qc, shots=1000).result().get_counts()
print("Alice's results (c1 c0):", counts)

target = Statevector(psi)
bob = bob_state(qc)
print("fidelity of Bob's qubit with the sent state:", round(state_fidelity(target, bob), 6))
print("Bloch vector sent:     ", bloch_of(np.outer(target.data, np.conj(target.data))))
print("Bloch vector received: ", bloch_of(bob))

undo = teleport_circuit(psi, bob_corrections, measure_bob=True, after=psi.inverse())
c_undo = sim.run(undo, shots=1000).result().get_counts()
zeros = sum(v for k, v in c_undo.items() if k[0] == "0")
print("undo test: Bob measured 0 in", zeros, "of 1000 shots")"""),
md("""## Step 5: Bob learns nothing until the bits arrive

Teleportation does not send anything faster than light. Before Alice's two classical bits reach him, Bob does not know which correction to apply. Averaged over Alice's four equally likely results, his qubit is then in the **completely mixed state**, whose Bloch vector is (0, 0, 0): it carries no trace of |ψ⟩, whatever |ψ⟩ is.

Run the cell: it removes the corrections and prints Bob's Bloch vector for three different states. Canvas asks for the length of this vector."""),
code("""for th, ph in [(1.0, 0.6), (0.3, 2.0), (2.5, 4.0)]:
    no_fix = teleport_circuit(prepare(th, ph), None)
    v = bloch_of(bob_state(no_fix))
    print(f"state ry({th}) rz({ph}): Bob's Bloch vector before the bits arrive = {v}, length {np.linalg.norm(v):.4f}")"""),
md("""## Step 6: your personal check

Open the **Module 0 lab check** in Canvas. Question 1 shows your own angle THETA and seed SEED. Type them below and run the next two cells. The check cell tests `active_reset()` on six states (some entangled) and `bob_corrections()` on twelve random states. Only if every test passes does it print your **verification value**: the number of shots, out of 1,000, in which Bob measures 1 after receiving ry(THETA)|0⟩, run with your seed. Type it into Canvas."""),
code("""THETA = 0.0   # your angle from Canvas, for example 1.25
SEED = 0      # your seed from Canvas, for example 512"""),
code(CHECK + '''

passed, messages, value = check_l2_module0(active_reset, bob_corrections, THETA, SEED)
for m in messages:
    print(m)
if passed:
    print("\\nAll tests passed. Your verification value is", value)
else:
    print("\\nNot yet: fix the item above and run this cell again.")'''),
md("""## What you should notice

- A measurement in the middle of a circuit gives a result you can use at once, and it leaves the qubit in |0⟩ or |1⟩.
- `qc.if_test((clbit, value))` runs gates only in the shots where that bit has that value. A measurement plus feedforward can rebuild a reset.
- Teleportation moves an unknown state with one Bell pair and two classical bits. The original is destroyed by Alice's measurement, so nothing is copied, and Bob's qubit means nothing until the bits arrive.

Module 1 uses these ideas again: phase estimation reads a phase into measured bits, and Level 3 uses feedforward to correct errors.

Developed through the Intel Semiconductor Education Program at Central State University (ISEP-CSU). CC BY 4.0. Questions: mhadizadeh@centralstate.edu"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl2m0c{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / NAME)
shutil.copyfile(ROOT / "content" / "level1" / "qsim.py", OUT / "qsim.py")
print("wrote", OUT / NAME, "and synced qsim.py")
