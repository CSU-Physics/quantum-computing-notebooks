"""Build both versions of the Level 2 Module 4 lab (running on real hardware) from checks/l2_m4_check.py:
- colab/level2/QC-L2-M4-hardware-colab.ipynb: Google Colab with real Qiskit (pinned versions) and FakePittsburgh;
- content/level2/QC-L2-M4-lab-hardware-browser.ipynb: the browser version (qsim), which reads the same transpiler and
  noisy-simulator results from content/level2/m4_saved_runs.json (made by tools/make_l2_m4_data.py).
Both versions ask the learner to write the same three functions and give the same lab-check answers."""
import shutil
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = (ROOT / "checks" / "l2_m4_check.py").read_text().split('"""', 2)[2].lstrip()
VERSION = "2026-10-05"
PAGES = "https://csu-physics.github.io/quantum-computing-notebooks/files/level2/m4_saved_runs.json"
BROWSER_META = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
                "language_info": {"name": "python"}}
COLAB_META = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
              "language_info": {"name": "python"}, "colab": {"provenance": []}}


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


GROVER_TASK = """def grover_circuit(marked, iterations):
    \"\"\"Grover's search for one marked bit string (your Module 2 code as one function).\"\"\"
    n = len(marked)
    qc = QuantumCircuit(n, n)
    # YOUR CODE HERE: H on every qubit; `iterations` rounds of the oracle and the diffuser; measure qubit q into bit q
    raise NotImplementedError("Complete grover_circuit() first.")
    return qc"""

MCZ = """def mcz(qc, qubits):
    \"\"\"Prepared for you (as in Module 2): a multi-controlled Z on the given qubits.\"\"\"
    qubits = list(qubits)
    if len(qubits) == 2:
        qc.cz(qubits[0], qubits[1])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])"""

TQ_TASK = """def two_qubit_count(ops):
    \"\"\"The number of two-qubit gates in a list of instructions [(name, [qubits]), ...], barriers not counted.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete two_qubit_count() first.")"""

SR_TASK = """def success_rate(counts, marked):
    \"\"\"The fraction of shots in counts that gave the marked string.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete success_rate() first.")"""

CHECK_RUN = CHECK + '''

passed, messages, value = check_l2_module4(grover_circuit, two_qubit_count, success_rate, MARKED, SEED, saved)
for m in messages:
    print(m)
if passed:
    print("\\nAll tests passed. Your verification value is", value)
else:
    print("\\nNot yet: fix the item above and run this cell again.")'''

INTRO_COMMON = """A circuit on paper is not yet a circuit a quantum computer can run. The **transpiler** rewrites it in the computer's **native gates**, places its qubits on physical qubits that are actually connected (**layout**), adds swaps where they are not (**routing**), and simplifies the result (**optimization**). Every extra two-qubit gate adds error, so how well this is done decides how often the answer comes out right.

In this lab you take your Grover circuit from Module 2 (3 qubits, marked string 101, 2 iterations) to **FakePittsburgh**, a model of the IBM computer `ibm_pittsburgh` (156 qubits) that comes with Qiskit, and:

1. read the computer's native gates, connections and error rates;
2. compare the transpiled circuit at optimization levels 0 to 3: depth, gate counts and the two-qubit gates (`two_qubit_count`);
3. run each version on the computer's **noise model** and measure how often it finds 101 (`success_rate`);
4. see what noise does to the best number of iterations;
5. get your verification value."""


def device_md(live):
    src = "the backend object" if live else "the saved results"
    return f"""## Step 1: the computer

The cell prints what the transpiler has to work with, read from {src}: the number of qubits, the **native gates** (every circuit must be rewritten in these), how many pairs of qubits are directly connected, and typical error rates from the calibration. On IBM Heron computers such as `ibm_pittsburgh` the two-qubit native gate is the controlled Z, `cz`, and each qubit is connected to at most three others (a "heavy-hex" lattice). The lab check asks for the number of qubits."""


BUDGET_NOTICE = """**What the numbers show.** At level 0 the transpiler keeps the trivial layout, physical qubits 0, 1 and 2. On this computer those qubits are worse than typical: their `cz` errors are about 0.005, three times the median, and two of them have short T1 times. So the median-error estimate (0.862) is far too hopeful, while the actual-error estimate (0.726) comes close to the noise model (0.683); the rest of the gap is mostly decoherence during the long level-0 circuit. At level 3 the transpiler chooses qubits 113, 114 and 119, whose `cz` errors are below 0.001.

Levels 2 and 3 give the same gate counts here, yet level 3 does better (0.892 against 0.871): the difference is only in which physical qubits were chosen. On real hardware, **where** a circuit runs matters as much as how many gates it has."""


ITER_MD = """## Step 5: noise and the number of iterations

Without noise, 2 iterations are best for 3 qubits (0.9453). Each iteration adds gates, and on a noisy computer each gate costs a little. The table compares the ideal probability with the noise-model result at level 3 for 0 to 3 iterations.

**What to notice.** Each iteration adds about 20 two-qubit gates. On this computer 2 iterations are still the best choice, but the noise takes more from 2 iterations (0.945 to 0.892) than from 1 (0.781 to 0.752). With more qubits, the best number of iterations grows (about (π/4)·√N), the circuit grows with it, and on a noisier computer a smaller number of iterations, or a shorter circuit, can give the better result."""

WHAT_NOTICE = """## What you should notice

- The transpiler rewrites a circuit in the native gates (here `cz`, `rz`, `sx`, `x`), chooses physical qubits, and adds swaps where the qubits are not connected.
- Higher optimization levels give shorter circuits with fewer two-qubit gates, and fewer two-qubit gates mean fewer errors: the success rate rises from level 0 to level 3.
- Two-qubit gates and readout dominate the error budget: (1 − ε)ⁿ shrinks fast with n. The errors differ from qubit to qubit, so the physical qubits the transpiler chooses matter too.
- Noise lowers every success rate, and more iterations mean more gates. Here 2 iterations stay best, but the margin shrinks; on a larger or noisier problem the best number of iterations on hardware can be smaller than in theory, and an algorithm is only useful while its circuit stays short enough.

**Next in Canvas:** the lab check, the quiz and the time log."""


def browser_cells():
    return [
md(f"""# Module 4 lab (browser version): running on real hardware

**Quantum Computing Intermediate · Module 4 · about 90 minutes** · notebook version {VERSION}

This is the **browser version** of the lab, for learners without a Google account. The transpiler and Qiskit Aer cannot run in a browser, so this version reads their results from a file saved with the same versions the Colab version uses (qiskit 2.5.2, qiskit-aer 0.17.2, qiskit-ibm-runtime 0.50.0). You write the same three functions and get the same lab-check answers as in Colab; the difference is that here you analyse the transpiler's output instead of running it.

{INTRO_COMMON}

The **Module 4 lab check** in Canvas asks for results from this notebook, so keep it open in a second tab. Run each code cell with **Shift + Enter**, in order."""),
md("""## Step 0: start Python and load the saved results

Run the cell below. The first time, Python can take up to a minute to start in your browser."""),
code("""import json, math
import numpy as np
import qsim
from qsim import QuantumCircuit, Statevector

saved = json.load(open("m4_saved_runs.json"))
teach = saved["teach"]
print("Ready. qsim", qsim.__version__, "| saved results made with", saved["versions"])"""),
md(device_md(False)),
code("""d = saved["device"]
print("computer:", d["name"], "with", d["num_qubits"], "qubits")
print("native gates:", ", ".join(g for g in d["basis_gates"] if g not in ("delay", "if_else", "measure_2")))
print("two-qubit gate:", d["two_qubit_gate"], "| directly connected pairs:", d["coupled_pairs"],
      "| neighbours of qubit 0:", d["neighbours_of_0"])
print(f"median errors: cz {d['median_cz_error']:.2e}, sx {d['median_sx_error']:.2e}, readout {d['median_readout_error']:.2e}")
print(f"median T1 {d['median_t1_us']:.0f} microseconds, T2 {d['median_t2_us']:.0f} microseconds")"""),
md("""## Step 2: your Grover circuit

**Your task:** write `grover_circuit(marked, iterations)`, your Module 2 search as one function: H on every qubit, `iterations` rounds of the oracle (X gates on the 0 bits, `mcz`, the same X gates) and the diffuser (H, X, `mcz`, X, H on every qubit), then measure qubit q into classical bit q. The helper `mcz` is prepared as in Module 2.

The cell after it checks the ideal success probability for 101 with 2 iterations: 0.9453."""),
code(MCZ + "\n\n\n" + GROVER_TASK),
code("""qc = grover_circuit("101", 2)
c = qc.copy(); c.remove_final_measurements()
print("ideal P(101) with 2 iterations:", round(Statevector(c).probabilities_dict().get("101", 0), 4))
print("gates in your circuit:", dict(qc.count_ops()))"""),
md("""## Step 3: what the transpiler did

The saved results contain the course's reference Grover circuit (the same gates as Module 2's reference) for 101 with 2 iterations, transpiled for FakePittsburgh at optimization levels 0 to 3 with `seed_transpiler=11`. The table shows the depth (the longest chain of gates), the total number of gates, the gates of each kind, and the **physical qubits** the transpiler chose for qubits 0, 1 and 2.

**Your task:** write `two_qubit_count(ops)`. It receives a list of instructions, each a name and a list of qubits, for example `[("cz", [3, 4]), ("rz", [3]), ("barrier", [3, 4])]`, and returns how many are two-qubit gates. Count by the number of qubits, not by name, and do not count barriers (a barrier can span two qubits but is not a gate). The cell after it checks your function on the saved level-0 and level-3 circuits. The lab check asks for the number of two-qubit gates at level 0."""),
code("""print("level  depth  gates  two-qubit  physical qubits  gates by kind")
for lvl in range(4):
    r = teach["runs"]["2"][str(lvl)]
    print(f"  {lvl}    {r['depth']:4d}   {r['size']:4d}      {r['two_qubit']:3d}      {str(r['layout']):15s}  {r['ops']}")"""),
code(TQ_TASK),
code("""for lvl in ("0", "3"):
    ops = teach["instructions"][lvl]
    print(f"level {lvl}: your two_qubit_count gives {two_qubit_count(ops)}, the table says {teach['runs']['2'][lvl]['two_qubit']}")"""),
md("""**What to notice.** Before transpiling, each `mcx` with two controls is one gate; the computer has no such gate, so it becomes several `cz` gates and single-qubit rotations. At level 0 the transpiler only translates and routes; at levels 2 and 3 it also cancels and merges gates, and the circuit becomes much shorter."""),
md("""## Step 4: run it on the noise model

`AerSimulator.from_backend(FakePittsburgh())` builds a noise model from the calibration data: gate errors, readout errors and decoherence on the physical qubits used. The saved results hold 4,000 shots of each transpiled circuit on that model (`seed_simulator=11`).

**Your task:** write `success_rate(counts, marked)`: the fraction of shots that gave the marked string. Use `counts.get(marked, 0)`, because a string that never appeared has no entry, and divide by the total number of shots. The lab check asks for the success rate at level 3."""),
code(SR_TASK),
code("""ideal = teach["ideal"]["2"]
print(f"ideal (no noise): {ideal:.4f}")
for lvl in range(4):
    r = teach["runs"]["2"][str(lvl)]
    print(f"level {lvl}: {r['two_qubit']:3d} two-qubit gates   success rate {success_rate(r['counts'], '101'):.4f}")"""),
md("""**An error budget.** If every two-qubit gate fails with probability ε₂ and every readout with ε_r, the chance that nothing goes wrong is about (1 − ε₂)^(two-qubit gates) × (1 − ε_r)^3. Multiplied by the ideal probability, this gives a rough estimate of the success rate. The cell works it out twice:

- with the **median** errors of the whole computer (the same ε for every gate);
- with the **actual** error of each `cz` gate and each readout in the transpiled circuit, which depends on the physical qubits the transpiler chose.

Both leave out single-qubit errors, decoherence while qubits wait, and errors that happen not to change the answer, so they are only a guide. Compare them with the noise-model result, especially at level 0."""),
code("""d = saved["device"]
print("level  median-error estimate  actual-error estimate  noise model")
for lvl in range(4):
    r = teach["runs"]["2"][str(lvl)]
    est_median = ideal * (1 - d["median_cz_error"]) ** r["two_qubit"] * (1 - d["median_readout_error"]) ** 3
    est_actual = ideal * math.prod(1 - e for e in r["cz_errors"]) * math.prod(1 - e for e in r["readout_errors"])
    print(f"  {lvl}          {est_median:.3f}                  {est_actual:.3f}              {success_rate(r['counts'], '101'):.3f}")"""),
md(BUDGET_NOTICE),
md(ITER_MD),
code("""for t in range(4):
    r = teach["runs"][str(t)]["3"]
    print(f"{t} iterations: {r['two_qubit']:3d} two-qubit gates   ideal {teach['ideal'][str(t)]:.4f}   noise model {success_rate(r['counts'], '101'):.4f}")"""),
md("""## Step 6: your personal check

Open the **Module 4 lab check** in Canvas. Question 1 shows your own MARKED (0 to 7, the 3-bit string to search for) and SEED. Type them below and run the next two cells. The check cell tests `grover_circuit()`, `two_qubit_count()` and `success_rate()`. Only if every test passes does it print your **verification value**: in the saved runs, the number of shots out of 4,000 in which the reference Grover circuit for MARKED with 2 iterations, transpiled at level 3 with seed SEED, found MARKED on the noise model (seed SEED). The Colab version reads the same saved run, so both versions give the same value."""),
code("""MARKED = -1    # your number from Canvas, for example 5 (the string 101)
SEED = 0       # your seed from Canvas, for example 512"""),
code(CHECK_RUN),
md(WHAT_NOTICE),
    ]


def colab_cells():
    return [
md(f"""# Module 4 lab (Colab version): running on real hardware

**Quantum Computing Intermediate · Module 4 · about 90 minutes** · notebook version {VERSION}

This is the **Google Colab version** of the lab: it installs real Qiskit and runs the transpiler and the noise model itself. (No Google account? Use the browser version on the Canvas lab page; it gives the same answers.)

{INTRO_COMMON}

No IBM account is needed: FakePittsburgh runs entirely in this notebook. An optional last step runs your circuit on a real IBM computer if you have an account.

The **Module 4 lab check** in Canvas asks for results from this notebook, so keep it open in a second tab. Run each code cell with **Shift + Enter**, in order. To keep your work, choose **File > Save a copy in Drive**."""),
md("""## Step 0: install Qiskit and load the saved results

This takes about a minute. The versions are fixed so that the results match the lab check. The second cell also downloads the course's saved results (used by the check cell in Step 6)."""),
code("""%pip install -q qiskit==2.5.2 qiskit-aer==0.17.2 qiskit-ibm-runtime==0.50.0"""),
code(f"""import json, math, urllib.request
import numpy as np
import qiskit, qiskit_aer, qiskit_ibm_runtime
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakePittsburgh

backend = FakePittsburgh()
noisy = AerSimulator.from_backend(backend)
saved = json.load(urllib.request.urlopen("{PAGES}"))
print("qiskit", qiskit.__version__, "| qiskit-aer", qiskit_aer.__version__, "| qiskit-ibm-runtime", qiskit_ibm_runtime.__version__)
print("saved results made with", saved["versions"])"""),
md(device_md(True)),
code("""import statistics
t = backend.target
cz_err = statistics.median(p.error for p in t["cz"].values() if p and p.error is not None)
sx_err = statistics.median(p.error for p in t["sx"].values() if p and p.error is not None)
ro_err = statistics.median(p.error for p in t["measure"].values() if p and p.error is not None)
pairs = {tuple(sorted(e)) for e in t["cz"].keys()}
print("computer:", backend.name, "with", backend.num_qubits, "qubits")
print("native gates:", ", ".join(g for g in sorted(backend.operation_names) if g not in ("delay", "if_else", "measure_2")))
print("two-qubit gate: cz | directly connected pairs:", len(pairs),
      "| neighbours of qubit 0:", sorted({b for a, b in pairs if a == 0} | {a for a, b in pairs if b == 0}))
print(f"median errors: cz {cz_err:.2e}, sx {sx_err:.2e}, readout {ro_err:.2e}")"""),
md("""## Step 2: your Grover circuit

**Your task:** write `grover_circuit(marked, iterations)`, your Module 2 search as one function: H on every qubit, `iterations` rounds of the oracle (X gates on the 0 bits, `mcz`, the same X gates) and the diffuser (H, X, `mcz`, X, H on every qubit), then measure qubit q into classical bit q. The helper `mcz` is prepared as in Module 2. Your Module 2 code works here unchanged, because qsim uses Qiskit's names.

The cell after it checks the ideal success probability for 101 with 2 iterations: 0.9453."""),
code(MCZ + "\n\n\n" + GROVER_TASK),
code("""qc = grover_circuit("101", 2)
c = qc.copy(); c.remove_final_measurements()
print("ideal P(101) with 2 iterations:", round(Statevector(c).probabilities_dict().get("101", 0), 4))
print("gates in your circuit:", dict(qc.count_ops()))"""),
md("""## Step 3: transpile at optimization levels 0 to 3

So that everyone's numbers match, the table uses the course's reference Grover circuit (the same gates as Module 2's reference) and `seed_transpiler=11`; Step 3b then transpiles your own circuit. `generate_preset_pass_manager(backend=..., optimization_level=...)` builds the transpiler for this computer, and `.run(circuit)` gives the **ISA circuit** (written only in the computer's instructions). The table shows the depth, the number of gates, the gates of each kind, and the physical qubits chosen for qubits 0, 1 and 2.

**Your task:** write `two_qubit_count(ops)`. It receives a list of instructions, each a name and a list of qubits, for example `[("cz", [3, 4]), ("rz", [3]), ("barrier", [3, 4])]`, and returns how many are two-qubit gates. Count by the number of qubits, not by name, and do not count barriers. The prepared helper `instructions(circuit)` turns a circuit into such a list. The lab check asks for the number of two-qubit gates at level 0."""),
code(TQ_TASK + '''


def instructions(circuit):
    """Prepared: a circuit as a list of (name, [qubit indices])."""
    return [(i.operation.name, [circuit.find_bit(q).index for q in i.qubits]) for i in circuit.data]'''),
code("""def reference_grover(marked, iterations):
    \"\"\"The course's reference circuit, written out as in Module 2.\"\"\"
    n = len(marked)
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iterations):
        zeros = [q for q in range(n) if marked[n - 1 - q] == "0"]
        if zeros:
            qc.x(zeros)
        qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
        if zeros:
            qc.x(zeros)
        qc.h(range(n)); qc.x(range(n))
        qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
        qc.x(range(n)); qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


isa = {}
print("level  depth  gates  two-qubit  physical qubits  gates by kind")
for lvl in range(4):
    pm = generate_preset_pass_manager(backend=backend, optimization_level=lvl, seed_transpiler=11)
    isa[lvl] = pm.run(reference_grover("101", 2))
    ops = instructions(isa[lvl])
    print(f"  {lvl}    {isa[lvl].depth():4d}   {isa[lvl].size():4d}      {two_qubit_count(ops):3d}      "
          f"{str(list(isa[lvl].layout.final_index_layout())):15s}  {dict(isa[lvl].count_ops())}")"""),
md("""**What to notice.** Before transpiling, each `mcx` with two controls is one gate; the computer has no such gate, so it becomes several `cz` gates and single-qubit rotations. At level 0 the transpiler only translates and routes; at levels 2 and 3 it also cancels and merges gates, and the circuit becomes much shorter.

### Step 3b: your own circuit

Run the cell to transpile your own `grover_circuit("101", 2)` at level 3. If you wrote the gates in another order, the counts can differ a little from the reference: the transpiler works on the gates you give it."""),
code("""mine = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=11).run(grover_circuit("101", 2))
print("your circuit at level 3:", mine.depth(), "depth,", two_qubit_count(instructions(mine)), "two-qubit gates",
      "| reference:", isa[3].depth(), "depth,", two_qubit_count(instructions(isa[3])), "two-qubit gates")"""),
md("""## Step 4: run it on the noise model

`AerSimulator.from_backend(backend)` builds a noise model from the calibration data: gate errors, readout errors and decoherence on the physical qubits used. The cell runs 4,000 shots of each level with `seed_simulator=11`; it takes about 10 seconds.

**Your task:** write `success_rate(counts, marked)`: the fraction of shots that gave the marked string. Use `counts.get(marked, 0)`, because a string that never appeared has no entry, and divide by the total number of shots. The lab check asks for the success rate at level 3."""),
code(SR_TASK),
code("""c = reference_grover("101", 2); c.remove_final_measurements()
ideal = Statevector(c).probabilities_dict()["101"]
print(f"ideal (no noise): {ideal:.4f}")
noisy_counts = {}
for lvl in range(4):
    noisy_counts[lvl] = noisy.run(isa[lvl], shots=4000, seed_simulator=11).result().get_counts()
    print(f"level {lvl}: {two_qubit_count(instructions(isa[lvl])):3d} two-qubit gates   success rate {success_rate(noisy_counts[lvl], '101'):.4f}")"""),
md("""**An error budget.** If every two-qubit gate fails with probability ε₂ and every readout with ε_r, the chance that nothing goes wrong is about (1 − ε₂)^(two-qubit gates) × (1 − ε_r)^3. Multiplied by the ideal probability, this gives a rough estimate of the success rate. The cell works it out twice:

- with the **median** errors of the whole computer (the same ε for every gate);
- with the **actual** error of each `cz` gate and each readout in the transpiled circuit, which depends on the physical qubits the transpiler chose.

Both leave out single-qubit errors, decoherence while qubits wait, and errors that happen not to change the answer, so they are only a guide. Compare them with the noise-model result, especially at level 0."""),
code("""print("level  median-error estimate  actual-error estimate  noise model")
for lvl in range(4):
    ops = instructions(isa[lvl])
    est_median = ideal * (1 - cz_err) ** two_qubit_count(ops) * (1 - ro_err) ** 3
    est_actual = ideal * math.prod(1 - t["cz"][tuple(q)].error for name, q in ops if name == "cz") \\
                       * math.prod(1 - t["measure"][(q[0],)].error for name, q in ops if name == "measure")
    print(f"  {lvl}          {est_median:.3f}                  {est_actual:.3f}              {success_rate(noisy_counts[lvl], '101'):.3f}")"""),
md(BUDGET_NOTICE),
md(ITER_MD),
code("""pm3 = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=11)
for t_ in range(4):
    circ = reference_grover("101", t_)
    c = circ.copy(); c.remove_final_measurements()
    isa_t = pm3.run(circ)
    counts_t = noisy.run(isa_t, shots=4000, seed_simulator=11).result().get_counts()
    print(f"{t_} iterations: {two_qubit_count(instructions(isa_t)):3d} two-qubit gates   ideal {Statevector(c).probabilities_dict().get('101', 0):.4f}   noise model {success_rate(counts_t, '101'):.4f}")"""),
md("""## Step 6: your personal check

Open the **Module 4 lab check** in Canvas. Question 1 shows your own MARKED (0 to 7, the 3-bit string to search for) and SEED. Type them below and run the next two cells. The check cell tests `grover_circuit()`, `two_qubit_count()` and `success_rate()`. Only if every test passes does it print your **verification value**: in the course's saved runs, the number of shots out of 4,000 in which the reference Grover circuit for MARKED with 2 iterations, transpiled at level 3 with seed SEED, found MARKED on the noise model (seed SEED). The cell after it repeats that run live here, to show that this notebook reproduces it."""),
code("""MARKED = -1    # your number from Canvas, for example 5 (the string 101)
SEED = 0       # your seed from Canvas, for example 512"""),
code(CHECK_RUN),
code("""if passed:
    w = format(int(MARKED), "03b")
    live_isa = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=int(SEED)).run(reference_grover(w, 2))
    live = noisy.run(live_isa, shots=4000, seed_simulator=int(SEED)).result().get_counts().get(w, 0)
    print("this notebook's live run:", live, "| saved run:", value,
          "| they match" if live == value else "| they differ: use the saved value printed above in Canvas")"""),
md("""## Step 7 (optional): a real IBM quantum computer

If you have an IBM Quantum account with an Open Plan instance (see **Set Up Your Tools** in Start Here), you can run the level-3 circuit on a real computer. Save your API key and instance as Colab secrets named `IBM_QUANTUM_API_KEY` and `IBM_QUANTUM_INSTANCE`; never type a key into a cell. The run uses a few seconds of quantum time, but the queue can take from seconds to hours. This step is not graded."""),
code("""RUN_ON_HARDWARE = False   # set to True to use your IBM Quantum account

if RUN_ON_HARDWARE:
    from google.colab import userdata
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime.executor_sampler import Sampler
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=userdata.get("IBM_QUANTUM_API_KEY"),
                                   instance=userdata.get("IBM_QUANTUM_INSTANCE"))
    real = service.least_busy(operational=True, simulator=False, min_num_qubits=3)
    real_isa = generate_preset_pass_manager(backend=real, optimization_level=3, seed_transpiler=11).run(reference_grover("101", 2))
    job = Sampler(mode=real).run([real_isa], shots=4000)
    print("running on", real.name, "- job ID", job.job_id())
    real_counts = job.result()[0].data.c.get_counts()
    print("success rate on", real.name, ":", round(success_rate(real_counts, "101"), 4))
else:
    print("Skipped. Set RUN_ON_HARDWARE = True to run on a real IBM computer.")"""),
md(WHAT_NOTICE),
    ]


for cells, out, name, meta, prefix in (
        (browser_cells(), ROOT / "content" / "level2", "QC-L2-M4-lab-hardware-browser.ipynb", BROWSER_META, "qcl2m4b"),
        (colab_cells(), ROOT / "colab" / "level2", "QC-L2-M4-hardware-colab.ipynb", COLAB_META, "qcl2m4c")):
    nb = nbf.v4.new_notebook(cells=cells, metadata=meta)
    for i, c in enumerate(nb.cells):
        c["id"] = f"{prefix}{i:02d}"
    out.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, out / name)
    print("wrote", out / name)
shutil.copyfile(ROOT / "content" / "level1" / "qsim.py", ROOT / "content" / "level2" / "qsim.py")
