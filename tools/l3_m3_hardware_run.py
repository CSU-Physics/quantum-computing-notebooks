"""For the course team: run the Module 3 circuits on a real IBM quantum computer and save the result in the format of
content/level3/l3_m3_device_run.json (made by tools/make_l3_m3_data.py with the FakePittsburgh noise model).
Run it in Google Colab, cell by cell or as one cell:

    !pip install -q qiskit==2.5.2 qiskit-ibm-runtime==0.50.0
    # In Colab: Secrets (key icon) > add IBM_QUANTUM_TOKEN (your IBM Quantum API key) and, if your account has one,
    # IBM_QUANTUM_INSTANCE (the instance CRN). Never paste the key into the notebook itself.

Eight circuits (raw, DD, DD with the CNOT folded 3 and 5 times, four readout-calibration circuits), 8,000 shots each:
64,000 shots of short circuits, about 20 to 40 seconds of QPU time. Then replace content/level3/l3_m3_device_run.json
with the file this writes, change the lab's Step 7 text ("simulated with Qiskit Aer") and rerun tools/build_l3_m3.py.

Set the environment variable L3_DRY_RUN=1 to test the script on FakeManilaV2 without an account."""
import json
import os
from datetime import date

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2

SHOTS, IDLE_US = 8000, 20
KEYS = ("00", "01", "10", "11")

if os.environ.get("L3_DRY_RUN"):
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    backend = FakeManilaV2()
else:
    from qiskit_ibm_runtime import QiskitRuntimeService
    try:
        from google.colab import userdata                 # Colab Secrets
        token = userdata.get("IBM_QUANTUM_TOKEN")
        try:
            instance = userdata.get("IBM_QUANTUM_INSTANCE")
        except Exception:  # noqa: BLE001
            instance = None
    except ImportError:
        token, instance = None, None                      # elsewhere: a saved account is used
    service = (QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)
               if token else QiskitRuntimeService())
    backend = service.least_busy(operational=True, simulator=False)
t = backend.target
two_q = next(g for g in ("cz", "ecr", "cx") if g in t.operation_names)
print("backend:", backend.name, "two-qubit gate:", two_q)

# a coupled pair whose two-qubit and readout errors are close to the device medians
pairs = [(a, b) for (a, b), p in t[two_q].items() if p is not None and p.error is not None and a < b]
med_2q = float(np.median([t[two_q][p].error for p in pairs]))
med_ro = float(np.median([t["measure"][(q,)].error for q in range(backend.num_qubits)]))


def score(a, b):
    ro = [max(t["measure"][(q,)].error, 1e-4) for q in (a, b)]
    return abs(np.log(t[two_q][(a, b)].error / med_2q)) + sum(abs(np.log(r / med_ro)) for r in ro)


PAIR = list(min(pairs, key=lambda ab: score(*ab)))
print("pair:", PAIR)


def experiment(dd, scale=1):
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    for _ in range(scale):
        qc.cx(0, 1)
        qc.barrier()
    if dd:
        qc.delay(IDLE_US / 2, [0, 1], unit="us")
        qc.x([0, 1])
        qc.delay(IDLE_US / 2, [0, 1], unit="us")
        qc.x([0, 1])
    else:
        qc.delay(IDLE_US, [0, 1], unit="us")
    qc.h([0, 1])
    qc.measure([0, 1], [0, 1])
    return qc


def calibration(key):
    qc = QuantumCircuit(2, 2)
    if key[1] == "1":
        qc.x(0)
    if key[0] == "1":
        qc.x(1)
    qc.measure([0, 1], [0, 1])
    return qc


circuits = {"raw": experiment(False), "dd": experiment(True), "dd_fold3": experiment(True, 3),
            "dd_fold5": experiment(True, 5)}
circuits.update({f"cal_{k}": calibration(k) for k in KEYS})
pm = generate_preset_pass_manager(backend=backend, optimization_level=0, initial_layout=PAIR,
                                  scheduling_method="alap")
isa = {n: pm.run(qc) for n, qc in circuits.items()}
job = SamplerV2(mode=backend).run(list(isa.values()), shots=SHOTS)
print("job:", job.job_id())
res = job.result()
counts = {}
for name, pub in zip(isa, res):
    c = pub.data.c.get_counts()
    counts[name] = {k: int(c.get(k, 0)) for k in KEYS}

props = {}
for q in PAIR:
    qp = t.qubit_properties[q] if t.qubit_properties else None
    props[str(q)] = {"t1_us": round(qp.t1 * 1e6, 1) if qp and qp.t1 else None,
                     "t2_us": round(qp.t2 * 1e6, 1) if qp and qp.t2 else None,
                     "readout_error": round(t["measure"][(q,)].error, 5),
                     "x_error": round(t["x"][(q,)].error, 6) if "x" in t.operation_names else None}
result = {"backend": backend.name,
          "kind": (f"hardware run on {backend.name}, job {job.job_id()}" if not os.environ.get("L3_DRY_RUN")
                   else "dry run on FakeManilaV2 (not for the lab)"),
          "calibration_date": str(date.today()), "source": "qiskit-ibm-runtime SamplerV2",
          "physical_qubits": PAIR, "qubit_properties": props, "cz_error": round(t[two_q][tuple(PAIR)].error, 5),
          "idle_us": IDLE_US, "shots": SHOTS,
          "cz_gates": {n: int(q.count_ops().get(two_q, 0)) for n, q in isa.items()},
          "key_format": "'b1b0': classical bit 1 (qubit 1) on the left, bit 0 (qubit 0) on the right, as Qiskit prints counts",
          "circuits": {"raw": "H(0), CX(0,1), idle 20 us, H on both, measure both: <XX>",
                       "dd": "as raw, with an X on both qubits after 10 us and after 20 us",
                       "dd_fold3": "as dd, with the CX replaced by three CXs", "dd_fold5": "as dd, with five CXs",
                       "cal_b1b0": "prepare the bit string b1b0 with X gates and measure"},
          "counts": counts}
with open("l3_m3_device_run.json", "w") as f:
    json.dump(result, f, indent=1)
par = {n: sum(v * (-1) ** k.count("1") for k, v in c.items()) / SHOTS for n, c in counts.items() if not n.startswith("cal")}
print("wrote l3_m3_device_run.json; raw <XX>:", {k: round(v, 3) for k, v in par.items()})
