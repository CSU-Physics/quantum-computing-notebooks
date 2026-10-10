"""Build the optional Colab notebook of Level 3 Module 3: the lab's experiment with Qiskit Runtime's client-side
Estimator (qiskit_ibm_runtime.executor_estimator.Estimator) and its suppression and mitigation options, on a real IBM
computer when the learner has an IBM Quantum account, otherwise on a small fake device (FakeManilaV2: the client-side
Estimator simulates the whole device, so the 156-qubit FakePittsburgh is too large for Colab).
Writes colab/level3/QC-L3-M3-estimator-colab.ipynb. Not graded."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab" / "level3"
NAME = "QC-L3-M3-estimator-colab.ipynb"
VERSION = "2026-10-10"
META = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"}, "colab": {"provenance": []}}


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


cells = [
md(f"""# Module 3 (optional, Colab): IBM's Estimator does the mitigation

**Quantum Computing Advanced · Module 3 · optional, about 30 minutes** · notebook version {VERSION}

In the browser lab you wrote dynamical decoupling, readout mitigation and zero-noise extrapolation yourself. Here the same experiment, ⟨XX⟩ of a Bell state after a 20 µs wait, goes to Qiskit Runtime's **Estimator**, and you switch the same tools on as **options**. Nothing in this notebook is graded.

- **With an IBM Quantum account** (the free Open Plan is enough) the circuits run on a real IBM computer. Four small jobs use roughly 1 to 2 minutes of QPU time in all, depending on the queue's device and the number of randomizations; check your remaining time first.
- **Without an account** they run on `FakeManilaV2`, a 5-qubit model of an older IBM device, simulated in Colab.

The Estimator used here, `qiskit_ibm_runtime.executor_estimator.Estimator` (qiskit-ibm-runtime 0.50), prepares the extra circuits and does the post-processing **on your own computer**, so you can see what each option costs. The older `EstimatorV2` takes the same options but is deprecated."""),

md("""## Step 0: install Qiskit

About a minute. The versions are fixed so that the code below works as written."""),
code("%pip install -q qiskit==2.5.2 qiskit-aer==0.17.2 qiskit-ibm-runtime==0.50.0"),

md("""## Step 1: choose the computer

To use a real IBM computer, store your API key in Colab's **Secrets** (the key icon on the left) under the name `IBM_QUANTUM_TOKEN`, and, if your account has one, the instance CRN under `IBM_QUANTUM_INSTANCE`. Allow this notebook to read them. **Never paste the key into a cell.** Without these secrets the cell picks `FakeManilaV2`."""),
code("""import warnings
import numpy as np
import qiskit, qiskit_ibm_runtime
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.executor_estimator import Estimator
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
warnings.filterwarnings("ignore", category=DeprecationWarning)

token = instance = None
try:
    from google.colab import userdata
    try:
        token = userdata.get("IBM_QUANTUM_TOKEN")
        instance = userdata.get("IBM_QUANTUM_INSTANCE")
    except Exception:
        pass
except ImportError:
    pass

if token:
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance or None)
    backend = service.least_busy(operational=True, simulator=False)
    REAL = True
else:
    backend = FakeManilaV2()
    REAL = False
print("qiskit", qiskit.__version__, "| qiskit-ibm-runtime", qiskit_ibm_runtime.__version__)
print("computer:", backend.name, "(real hardware)" if REAL else "(simulated fake device)")"""),

md("""## Step 2: the experiment

The same circuit as in the lab: H and a CNOT make the Bell state, then both qubits wait 20 µs. The Estimator measures the observable XX itself (it adds the basis change), so there are no H gates or measurements at the end. The transpiler places the two qubits on a connected pair and rewrites the gates in the device's native gates."""),
code("""qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.delay(20, [0, 1], unit="us")

pm = generate_preset_pass_manager(backend=backend, optimization_level=1, seed_transpiler=7)
isa = pm.run(qc)
observable = SparsePauliOp("XX").apply_layout(isa.layout)
print("physical qubits:", isa.layout.final_index_layout()[:2], " gates:", dict(isa.count_ops()))"""),

md("""## Step 3: four sets of options

| run | options | lab step |
|---|---|---|
| raw | `resilience_level = 0` | Step 2 |
| DD | plus `dynamical_decoupling.enable = True`, `sequence_type = "XY4"` | Step 3 |
| DD + readout | plus `resilience.measure_mitigation = True` (TREX) | Step 4 |
| DD + readout + ZNE | plus `resilience.zne_mitigation = True`, noise factors (1, 3, 5), linear extrapolation | Step 5 |

Each run uses 4,000 shots for the main circuit. Readout mitigation and ZNE add circuits of their own (randomized readout, folded gates); that is their cost. *Predict first:* which step changes ⟨XX⟩ most on this computer?"""),
code("""DD = {"enable": True, "sequence_type": "XY4"}
RUNS = {
    "raw": {"resilience_level": 0},
    "DD": {"resilience_level": 0, "dynamical_decoupling": DD},
    "DD + readout": {"resilience_level": 0, "dynamical_decoupling": DD, "resilience": {"measure_mitigation": True}},
    "DD + readout + ZNE": {"resilience_level": 0, "dynamical_decoupling": DD,
                           "resilience": {"measure_mitigation": True, "zne_mitigation": True,
                                          "zne": {"noise_factors": (1, 3, 5), "extrapolator": "linear"}}},
}
results = {}
for name, opts in RUNS.items():
    est = Estimator(mode=backend, options={**opts, "default_shots": 4000})
    job = est.run([(isa, observable)])
    res = job.result()[0]
    results[name] = (float(res.data.evs), float(res.data.stds))
    print(f"{name:20s} <XX> = {results[name][0]:.3f} +/- {results[name][1]:.3f}")"""),

md("""## Step 4: did each option help?

Judge every step against its statistical error, as in the lab's Step 6. The cell prints each step's change and its combined error σ_diff, and applies the course's 3σ rule (Module 6): a change larger than 3σ_diff is significant, because shot noise alone almost never produces it. A smaller change may still be real, but this run cannot show it. The rule assumes the shot noise is roughly bell-shaped, which holds for many shots."""),
code("""names = list(results)
for a, b in zip(names, names[1:]):
    (va, sa), (vb, sb) = results[a], results[b]
    diff, err = vb - va, np.hypot(sa, sb)
    verdict = "significant (more than 3 sigma_diff)" if abs(diff) > 3 * err else "not significant: this run cannot tell"
    print(f"{a:>20s} -> {b:20s} change {diff:+.3f} (error {err:.3f}): {verdict}")"""),

md("""## What to look for

- **Readout mitigation** usually changes ⟨XX⟩ the most on real devices, as in the lab's Step 7.
- **DD** helps on real hardware when the qubits pick up coherent phase or crosstalk while they wait; on a fake device, whose noise model has no coherent idle errors, it changes little, as in the lab.
- **ZNE** helps when gate errors matter; with one CNOT and a 20 µs wait the dephasing dominates, so expect a small change and a larger error bar.
- What remains below 1 is mostly dephasing during the wait, which none of these options removes. Compare with exp(−t/T₂) for the two qubits (`backend.target.qubit_properties`).

Back in Canvas: nothing to submit for this notebook. If you ran on a real computer, the Module 3 discussion of your capstone (Module 6) can use these numbers."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl3m3colab{i:02d}"
OUT.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT / NAME)
print("wrote", OUT / NAME)
