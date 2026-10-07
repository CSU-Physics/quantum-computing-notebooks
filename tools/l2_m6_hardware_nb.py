"""Cells of the optional Module 6 Colab notebook: the project's reference circuit on a real IBM quantum computer."""
from l2_m6_text import md, code, VERSION


def cells():
    return [
        md(f"""# Module 6 (optional): your experiment on a real IBM quantum computer

**Quantum Computing Intermediate · Module 6 · optional, not graded · about 30 minutes plus the queue** · notebook version {VERSION}

This notebook runs the **reference circuit of your project** on a real IBM quantum computer and compares the result with three predictions:

1. the **ideal** value, without noise;
2. the **course noise model** (P2 = 0.01, READOUT = 0.02) used in the project notebooks;
3. IBM's **noise model of that computer**, built from its latest calibration (as FakePittsburgh was in Module 4).

**What you need.** An IBM Quantum account with an Open Plan instance (see **Set Up Your Tools** in Start Here). The Open Plan gives up to 10 minutes of quantum computer time every 28 days. One run of 4,000 shots of these small circuits uses a few seconds of it, but the job waits in a queue that can take from seconds to hours.

**Your API key stays secret.** In Colab, click the key icon (**Secrets**) in the left bar and add two secrets: `IBM_QUANTUM_API_KEY` (your API key) and `IBM_QUANTUM_INSTANCE` (your instance's CRN), and allow this notebook to read them. Never type a key into a cell: a notebook can be shared, and its cells with it.

**Without an account,** leave `RUN_ON_HARDWARE = False`: the notebook then runs everything on FakePittsburgh, through the same code, so you can see what the comparison looks like.

Run each cell with **Shift + Enter**, in order."""),
        md("## Step 0: install Qiskit\n\nThis takes about a minute. The versions are the same as in Module 4 and the transpiler project."),
        code("%pip install -q qiskit==2.5.2 qiskit-aer==0.17.2 qiskit-ibm-runtime==0.50.0"),
        code("""import math
import numpy as np
import matplotlib.pyplot as plt
import qiskit, qiskit_aer, qiskit_ibm_runtime
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit_ibm_runtime.executor_sampler import Sampler
from qiskit_ibm_runtime.fake_provider import FakePittsburgh

print("qiskit", qiskit.__version__, "| qiskit-aer", qiskit_aer.__version__, "| qiskit-ibm-runtime", qiskit_ibm_runtime.__version__)"""),

        md("""## Step 1: choose your experiment

Set `EXPERIMENT` to your project: `"grover"`, `"qpe"`, `"qaoa"` or `"mirror"` (the transpiler comparison). The cell builds the project's reference circuit, with the same gates as in the project notebook:

| Experiment | Circuit | What is measured |
|---|---|---|
| `grover` | 3 qubits, marked 101, 2 iterations | fraction of shots giving 101 |
| `qpe` | phase estimation of 1/3 with 4 counting qubits | fraction giving the best estimate 0101 |
| `qaoa` | 2 layers, reference angles, 5-node graph | fraction giving a maximum cut (5 edges) |
| `mirror` | QFT, barrier, inverse QFT on input 1011 | fraction giving 1011 |"""),
        code("""EXPERIMENT = "grover"     # "grover", "qpe", "qaoa" or "mirror"


def mcz(qc, a, b, c):
    qc.cx(b, c); qc.tdg(c); qc.cx(a, c); qc.t(c); qc.cx(b, c); qc.tdg(c); qc.cx(a, c)
    qc.t(b); qc.t(c); qc.cx(a, b); qc.t(a); qc.tdg(b); qc.cx(a, b)


def grover_reference():
    qc = QuantumCircuit(3, 3)
    qc.h(range(3))
    for _ in range(2):
        qc.x(1); mcz(qc, 0, 1, 2); qc.x(1)                       # oracle for 101 (qubit 1 is the 0 bit)
        qc.h(range(3)); qc.x(range(3)); mcz(qc, 0, 1, 2); qc.x(range(3)); qc.h(range(3))
    qc.measure(range(3), range(3))
    return qc


def qft_gates(qc, qubits, inverse=False):
    n = len(qubits)
    if not inverse:
        for j in reversed(range(n)):
            qc.h(qubits[j])
            for k in reversed(range(j)):
                qc.cp(np.pi / 2 ** (j - k), qubits[k], qubits[j])
        for j in range(n // 2):
            qc.swap(qubits[j], qubits[n - 1 - j])
    else:
        for j in range(n // 2):
            qc.swap(qubits[j], qubits[n - 1 - j])
        for j in range(n):
            for k in range(j):
                qc.cp(-np.pi / 2 ** (j - k), qubits[k], qubits[j])
            qc.h(qubits[j])


def qpe_reference(m=4):
    qc = QuantumCircuit(m + 1, m)
    qc.x(m); qc.h(range(m))
    for k in range(m):
        qc.cp(2 * np.pi / 3 * 2 ** k, k, m)
    qft_gates(qc, list(range(m)), inverse=True)
    qc.measure(range(m), range(m))
    return qc


EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2)]


def qaoa_reference():
    gammas, betas = [2.849, 2.627], [0.517, 0.309]
    qc = QuantumCircuit(5, 5)
    qc.h(range(5))
    for g, b in zip(gammas, betas):
        for i, j in EDGES:
            qc.rzz(2 * g, i, j)
        qc.rx(2 * b, range(5))
    qc.measure(range(5), range(5))
    return qc


def mirror_reference(bits="1011"):
    n = len(bits)
    qc = QuantumCircuit(n, n)
    qc.x([q for q in range(n) if bits[n - 1 - q] == "1"])
    qft_gates(qc, list(range(n)))
    qc.barrier()
    qft_gates(qc, list(range(n)), inverse=True)
    qc.measure(range(n), range(n))
    return qc


def cut(key):
    return sum(key[-1 - i] != key[-1 - j] for i, j in EDGES)


circuits = {"grover": grover_reference, "qpe": qpe_reference, "qaoa": qaoa_reference, "mirror": mirror_reference}
targets = {"grover": {"101"}, "qpe": {"0101"}, "mirror": {"1011"},
           "qaoa": {format(s, "05b") for s in range(32) if cut(format(s, "05b")) == 5}}
qc = circuits[EXPERIMENT]()
target = targets[EXPERIMENT]
print(EXPERIMENT, "| target results:", sorted(target), "| gates:", dict(qc.count_ops()))"""),

        md("""## Step 2: the predictions without hardware

The ideal value comes from the state vector. The course noise model's value is computed exactly, as in the project notebooks: Aer follows the density matrix through the noisy gates, then the readout errors are applied to the recorded bits."""),
        code("""def course_noise_model(p2=0.01, readout=0.02):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["h", "x", "rx", "ry", "rz", "p", "t", "tdg", "s", "sdg", "sx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx", "cz", "cp", "rzz", "swap"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - readout, readout], [2 * readout, 1 - 2 * readout]]))
    return nm


def course_expected(qc, p2=0.01, readout=0.02):
    \"\"\"The course noise model's exact probability of each recorded result.\"\"\"
    meas = sorted((qc.find_bit(i.clbits[0]).index, qc.find_bit(i.qubits[0]).index) for i in qc.data if i.operation.name == "measure")
    body = qc.copy()
    body.remove_final_measurements()
    body.save_probabilities(qubits=[q for _, q in meas])
    probs = np.asarray(AerSimulator(method="density_matrix", noise_model=course_noise_model(p2, readout)).run(body).result().data()["probabilities"])
    m = len(meas)
    conf = np.array([[1 - readout, readout], [2 * readout, 1 - 2 * readout]])
    for j in range(m):
        probs = np.einsum("aib,ij->ajb", probs.reshape(2 ** (m - 1 - j), 2, 2 ** j), conf).reshape(-1)
    return {format(i, f"0{m}b"): float(p) for i, p in enumerate(probs)}


body = qc.copy()
body.remove_final_measurements()
p_ideal = sum(v for k, v in Statevector(body).probabilities_dict().items() if k in target)
p_course = sum(v for k, v in course_expected(qc).items() if k in target)
print(f"ideal: {p_ideal:.4f}   course noise model: {p_course:.4f}")"""),

        md("""## Step 3: choose the computer and transpile

With `RUN_ON_HARDWARE = True` the cell connects to your account with the two Colab secrets and picks the least busy real computer with enough qubits. Otherwise it uses FakePittsburgh. It then transpiles the circuit at optimization level 3 for that computer and runs 4,000 shots on **that computer's noise model** (Aer), the third prediction."""),
        code("""RUN_ON_HARDWARE = False   # set to True to use your IBM Quantum account

if RUN_ON_HARDWARE:
    from google.colab import userdata
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=userdata.get("IBM_QUANTUM_API_KEY"),
                                   instance=userdata.get("IBM_QUANTUM_INSTANCE"))
    computer = service.least_busy(operational=True, simulator=False, min_num_qubits=qc.num_qubits)
else:
    computer = FakePittsburgh()

isa = generate_preset_pass_manager(backend=computer, optimization_level=3, seed_transpiler=11).run(qc)
two_q = sum(1 for i in isa.data if i.operation.num_qubits == 2 and i.operation.name != "barrier")
model_counts = AerSimulator.from_backend(computer).run(isa, shots=4000, seed_simulator=11).result().get_counts()
p_device_model = sum(v for k, v in model_counts.items() if k in target) / 4000
print("computer:", computer.name, "| two-qubit gates after transpiling:", two_q, "| physical qubits:", list(isa.layout.final_index_layout()))
print(f"this computer's noise model, 4000 shots: {p_device_model:.4f}")"""),

        md("""## Step 4: run it

The Sampler sends the transpiled circuit to the computer (or, without hardware, runs it locally on FakePittsburgh's noise model) with 4,000 shots. On a real computer, the cell prints the job ID and waits for the queue; if Colab disconnects, you can find the result later under **Workloads** on the IBM Quantum Platform."""),
        code("""job = Sampler(mode=computer).run([isa], shots=4000)
if RUN_ON_HARDWARE:
    print("job ID:", job.job_id(), "- waiting for the queue...")
counts = job.result()[0].data.c.get_counts()
k = sum(v for key, v in counts.items() if key in target)
f = k / 4000
sigma = math.sqrt(f * (1 - f) / 4000)
print(f"measured on {computer.name}: {k} of 4000 shots, a fraction {f:.4f} ± {sigma:.4f} (1 sigma)")
for name, p in (("ideal", p_ideal), ("course noise model", p_course), (f"{computer.name}'s noise model", p_device_model)):
    print(f"  vs {name:30s} {p:.4f}: difference {f - p:+.4f} = {(f - p) / sigma:+.1f} sigma")
if EXPERIMENT == "qaoa":
    print(f"mean cut: {sum(v * cut(key) for key, v in counts.items()) / 4000:.4f}  (ideal 4.6234, course noise model 4.3326)")

labels = sorted(set(counts) | set(model_counts), key=lambda s: -counts.get(s, 0))[:12]
x = np.arange(len(labels))
plt.figure(figsize=(8, 3.3))
plt.bar(x - 0.2, [model_counts.get(s, 0) / 4000 for s in labels], 0.4, label="computer's noise model")
plt.bar(x + 0.2, [counts.get(s, 0) / 4000 for s in labels], 0.4, label="measured")
plt.xticks(x, labels, rotation=90); plt.ylabel("fraction of shots"); plt.legend(); plt.tight_layout(); plt.show()"""),

        md("""## What to look for

- **σ here comes from the measurement itself**, √(f(1 − f)/4000), because no one knows the computer's exact expected value. A difference of more than 3σ from a prediction means that prediction does not describe this run.
- **The ideal value** is always far away: a real computer is noisy.
- **The course noise model** is several times noisier than IBM's current computers, so the real result is often **better** than it predicts.
- **The computer's own noise model** is usually the closest prediction, but real results can differ from it, often for the worse and sometimes by more than 3σ, because the model leaves out crosstalk, leakage, drift since the last calibration and errors that repeat the same way every time.
- A second run, an hour or a day later, can differ from the first by more than its σ: the computer itself changes. That uncertainty is not in σ.

In your experiment summary (required, at the start of the Course Completion module), this run fits under **Limits**: it shows how far a noise model is from a real computer."""),
    ]
