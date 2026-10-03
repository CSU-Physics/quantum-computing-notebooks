"""Build the optional Module 5 real-hardware notebook for Google Colab (Qiskit and IBM Quantum)."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab" / "level1"
NAME = "QC-L1-M5-real-hardware-colab.ipynb"
META = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"}, "colab": {"provenance": []}}


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md("""# Module 5 (optional): your Bell circuit on a real IBM quantum computer

**Quantum Computing Foundations · Module 5 · optional, about 15 minutes plus the queue**

This notebook runs the Module 4 Bell circuit on a real IBM quantum computer, with 1,000 shots, and prints the counts so you can analyse them in Step 8 of the Module 5 lab in Canvas. It is **optional and not graded**: the Module 5 run required for the badge is the simulator work in the Canvas lab.

**What you need**

- An IBM Quantum account with an Open Plan instance, and its API key and instance CRN saved as Colab secrets named `IBM_QUANTUM_API_KEY` and `IBM_QUANTUM_INSTANCE` (see **Set Up Your Tools** in Start Here). Never type your key into a notebook cell.
- A few seconds of quantum time. The Open Plan gives up to 10 minutes per 28 days; this run uses a few seconds.

No account? Set `PRACTICE = True` in Step 2. The notebook then runs everything on `FakePittsburgh`, a simulator with a calibration snapshot of the IBM computer `ibm_pittsburgh`, and uses no account and no quantum time.

Run each cell with **Shift + Enter**, in order."""),
md("""## Step 1: install Qiskit

This takes about a minute. The versions are fixed so that the notebook keeps working the same way."""),
code("""%pip install -q qiskit==2.5.2 qiskit-aer==0.17.2 qiskit-ibm-runtime==0.50.0"""),
code("""import qiskit, qiskit_aer, qiskit_ibm_runtime
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.executor_sampler import Sampler   # the client-side Sampler (SamplerV2 is deprecated from 0.50.0)

print("qiskit", qiskit.__version__, "| qiskit-aer", qiskit_aer.__version__,
      "| qiskit-ibm-runtime", qiskit_ibm_runtime.__version__)

bell = QuantumCircuit(2)
bell.h(0)
bell.cx(0, 1)
bell.measure_all()          # the results go to a register named "meas"
bell.draw()"""),
md("""## Step 2: connect to IBM Quantum

The cell reads your key and instance from Colab's secrets (the key icon in the left bar; allow this notebook to use them when Colab asks) and picks the least busy IBM computer your instance can use."""),
code("""PRACTICE = False    # True: no account, no quantum time, a simulator of ibm_pittsburgh instead

if PRACTICE:
    from qiskit_ibm_runtime.fake_provider import FakePittsburgh
    backend = FakePittsburgh()
else:
    from google.colab import userdata
    service = QiskitRuntimeService(channel="ibm_quantum_platform",
                                   token=userdata.get("IBM_QUANTUM_API_KEY"),
                                   instance=userdata.get("IBM_QUANTUM_INSTANCE"))
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=2)

print("Running on:", backend.name, f"({backend.num_qubits} qubits)")"""),
md("""## Step 3: transpile for the hardware

The transpiler rewrites the circuit in the computer's native gates and chooses two connected physical qubits. Compare it with the circuit in Step 2 of the Canvas lab."""),
code("""pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa = pm.run(bell)
physical = isa.layout.final_index_layout()          # physical qubits used for qubits 0 and 1
print("physical qubits for qubits 0 and 1:", physical)
print("gates:", dict(isa.count_ops()))
isa.draw(idle_wires=False)"""),
md("""## Step 4: the readout errors of those qubits

These come from the computer's latest calibration. e0 = P(read 1 | was 0) and e1 = P(read 0 | was 1)."""),
code("""props = backend.properties()
my_read = {}
for logical, q in enumerate(physical):
    e0 = e1 = None
    if props is not None:
        qp = props.qubit_property(q)
        e0 = qp.get("prob_meas1_prep0", (None,))[0]
        e1 = qp.get("prob_meas0_prep1", (None,))[0]
    if e0 is None or e1 is None:                  # only the average is published
        e0 = e1 = backend.target["measure"][(q,)].error
    my_read[logical] = (round(float(e0), 5), round(float(e1), 5))
    print(f"qubit {logical} (physical {q}): e0 = {e0:.4f}, e1 = {e1:.4f}")"""),
md("""## Step 5: run on the quantum computer

The job waits in a queue first; that can take from seconds to an hour or more. You can close the tab and come back: the job keeps its place, and the job ID lets you find it on the IBM Quantum Platform. Run the cell once only."""),
code("""sampler = Sampler(mode=backend)
job = sampler.run([isa], shots=1000)
print("job ID:", job.job_id(), "- waiting for the result ...")
result = job.result()
counts = result[0].data.meas.get_counts()
print("counts:", dict(sorted(counts.items())))
try:
    print("quantum time used:", job.metrics()["usage"]["quantum_seconds"], "seconds")
except Exception:
    pass
plot_histogram(counts)"""),
md("""## Step 6: compare with a simulator of the same computer

`AerSimulator.from_backend` builds a noise model from the computer's calibration data, like the model you built in Step 6 of the Canvas lab, and an ideal simulator shows the result without noise."""),
code("""noisy = AerSimulator.from_backend(backend)
ideal = AerSimulator()
model_counts = noisy.run(isa, shots=1000, seed_simulator=7).result().get_counts()
ideal_counts = ideal.run(bell, shots=1000, seed_simulator=7).result().get_counts()

def different(c):
    return c.get("01", 0) + c.get("10", 0)

run_name = "practice run" if PRACTICE else "real hardware"
for name, c in (("ideal simulator", ideal_counts), ("noise model", model_counts), (run_name, counts)):
    print(f"{name:16s} different results: {different(c):4d} of 1000   {dict(sorted(c.items()))}")
plot_histogram([ideal_counts, model_counts, counts], legend=["ideal", "noise model", run_name])"""),
md("""## Step 7: copy your results into the Canvas lab

Copy the two lines printed below into Step 8 of the Module 5 lab in Canvas, in place of the two `None` lines, and run that cell."""),
code("""print("my_counts =", {k: int(counts.get(k, 0)) for k in ["00", "01", "10", "11"]})
print("my_read =", my_read)
print(f"# {backend.name}, physical qubits {list(physical)}, 1,000 shots" + (" (practice run, not real hardware)" if PRACTICE else ""))"""),
md("""Developed through the Intel Semiconductor Education Program at Central State University (ISEP-CSU). CC BY 4.0. The circuit and the run steps follow IBM Quantum Learning's lesson "Build and run your first quantum program" (CC BY-SA 4.0). Questions: mhadizadeh@centralstate.edu"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl1m5colab{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / NAME)
print("wrote", OUT / NAME)
