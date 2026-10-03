"""Build the optional Module 6 notebook for Google Colab: the learner's Bernstein-Vazirani code run in real Qiskit."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab" / "level1"
NAME = "QC-L1-M6-bernstein-vazirani-qiskit-colab.ipynb"
META = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"}, "colab": {"provenance": []}}


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md("""# Module 6 (optional): your Bernstein-Vazirani circuit in real Qiskit

**Quantum Computing Foundations · Module 6 · optional, about 20 minutes**

In the Module 6 lab you wrote `bv_oracle()` and `bv_circuit()` with the course's browser simulator, qsim, which uses the same commands as Qiskit. This notebook runs **your own code** in real Qiskit, the toolkit used in industry and research:

1. it checks your oracle against Qiskit's `Operator` and runs your circuit on Qiskit Aer;
2. it transpiles the circuit for a model of the IBM computer `ibm_pittsburgh` and shows what the hardware would really run;
3. it runs the transpiled circuit with that computer's calibrated noise.

It is **optional and not graded**, and needs no IBM account: everything runs on simulators in Colab. (To run on a real IBM computer, use the same steps as the Module 5 hardware notebook.)

Run each cell with **Shift + Enter**, in order."""),
md("""## Step 1: install Qiskit

This takes about a minute. The versions are fixed so that the notebook keeps working the same way."""),
code("""%pip install -q qiskit==2.5.2 qiskit-aer==0.17.2 qiskit-ibm-runtime==0.50.0"""),
code("""import qiskit, qiskit_aer
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakePittsburgh

print("qiskit", qiskit.__version__, "| qiskit-aer", qiskit_aer.__version__)"""),
md("""## Step 2: paste your two functions

Open your Module 6 lab notebook in Canvas, copy your finished `bv_oracle()` and `bv_circuit()` (the whole `def` blocks), and paste them over the two placeholders below. Nothing else needs to change: `QuantumCircuit`, `cx`, `x`, `h`, `compose` and `measure` mean the same in qsim and in Qiskit.

Remember the bit order: the rightmost character of `s` belongs to qubit 0, and the answer qubit is qubit `n`."""),
code("""def bv_oracle(s):
    # PASTE your bv_oracle() from the Module 6 lab here.
    raise NotImplementedError("Paste your bv_oracle() from the lab first.")


def bv_circuit(oracle, n):
    # PASTE your bv_circuit() from the Module 6 lab here.
    raise NotImplementedError("Paste your bv_circuit() from the lab first.")"""),
md("""## Step 3: check your code with Qiskit

The cell compares your oracle with the reference oracle (one CNOT for every 1 in s) using Qiskit's `Operator`, for several hidden strings, and runs your circuit on Qiskit Aer. With no noise, every shot should return s."""),
code("""def reference_oracle(s):
    n = len(s)
    qc = QuantumCircuit(n + 1)
    for i in range(n):
        if s[n - 1 - i] == "1":
            qc.cx(i, n)
    return qc


ideal = AerSimulator(seed_simulator=7)
all_ok = True
for s in ["1", "10", "1011", "0110", "111111", "100000", "10110011"]:
    n = len(s)
    same = Operator(bv_oracle(s)).equiv(Operator(reference_oracle(s)))
    counts = ideal.run(bv_circuit(bv_oracle(s), n), shots=1000).result().get_counts()
    found = counts == {s: 1000}
    all_ok = all_ok and same and found
    print(f"s = {s:>8}: oracle matches Qiskit's Operator: {same};  circuit returned {counts}")
print("\\nAll checks passed: your code runs unchanged in Qiskit." if all_ok else
      "\\nSomething differs: compare with your lab notebook, which runs the same tests.")"""),
md("""## Step 4: what the hardware would really run

A real IBM computer has its own native gates (on `ibm_pittsburgh`: Rz, √X and CZ) and only some pairs of qubits are connected. The transpiler rewrites your circuit for the device. Compare a string with one 1 and a string with six."""),
code("""backend = FakePittsburgh()           # a calibration snapshot of ibm_pittsburgh, no account needed
pm = generate_preset_pass_manager(backend=backend, optimization_level=1, seed_transpiler=7)

isa = {}
for s in ["100000", "111111"]:
    qc = bv_circuit(bv_oracle(s), len(s))
    isa[s] = pm.run(qc)
    ops = dict(isa[s].count_ops())
    k = qc.count_ops().get('cx', 0)
    print(f"s = {s}: your circuit has {k} CNOT{'s' if k != 1 else ''}; transpiled: {ops.get('cz', 0)} CZ, "
          f"{ops.get('sx', 0)} sqrt(X), {ops.get('rz', 0)} Rz, depth {isa[s].depth()}")"""),
md("""## Step 5: run it with the device's noise

`AerSimulator.from_backend` builds a noise model from the computer's calibration data, as in Module 5. The cell runs both transpiled circuits 1,000 times and counts how often each still returns its hidden string. More two-qubit gates mean more chances for an error, so the string with six 1s is usually found less often.

Each of the 1,000 shots is one run of the circuit and uses the oracle once, so the oracle is used 1,000 times here. "One query" refers to one ideal run."""),
code("""noisy = AerSimulator.from_backend(backend, seed_simulator=7)
results = {}
for s, circ in isa.items():
    counts = noisy.run(circ, shots=1000).result().get_counts()
    results[s] = counts
    best = max(counts, key=counts.get)
    print(f"s = {s}: returned s in {counts.get(s, 0)} of 1000 shots; most frequent result {best}")
plot_histogram([results["100000"], results["111111"]], legend=["s = 100000", "s = 111111"],
               title="Bernstein-Vazirani with ibm_pittsburgh's calibrated noise", figsize=(11, 4))"""),
md("""## What you should notice

- Your lab code ran unchanged in Qiskit, because qsim follows Qiskit's commands and bit order.
- The device does not run CNOT directly: each one becomes a CZ with single-qubit gates around it, and qubits that are not neighbours need extra gates.
- With realistic noise the hidden string is still the most frequent result, but not in every shot, and longer oracles are found less reliably. On noisy hardware you pay extra shots to read s with confidence.

Developed through the Intel Semiconductor Education Program at Central State University (ISEP-CSU). CC BY 4.0. Questions: mhadizadeh@centralstate.edu"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl1m6colab{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / NAME)
print("wrote", OUT / NAME)
