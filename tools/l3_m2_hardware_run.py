"""For the course team: run the Module 2 repetition-code circuits on a real IBM quantum computer and save the
result in the format of content/level3/l3_m2_device_run.json (made by tools/make_l3_m2_data.py with the
FakePittsburgh noise model). Run it in Google Colab, cell by cell or as one cell:

    !pip install -q qiskit qiskit-ibm-runtime
    # In Colab: Secrets (key icon) > add IBM_QUANTUM_TOKEN (your IBM Quantum API key) and, if your account has one,
    # IBM_QUANTUM_INSTANCE (the instance CRN). Never paste the key into the notebook itself.

Two circuits of 3 rounds, 4,000 shots each: well under one minute of QPU time on the Open Plan.
Then replace content/level3/l3_m2_device_run.json with the file this writes, change the lab's Step 7 sentence
"It is a simulation with real device noise" and rerun the checks of the lab page and slides (tools/build_l3_m2.py)."""
import json
from datetime import date

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

ROUNDS, SHOTS = 3, 4000

try:
    from google.colab import userdata                     # Colab Secrets
    token = userdata.get("IBM_QUANTUM_TOKEN")
    try:
        instance = userdata.get("IBM_QUANTUM_INSTANCE")
    except Exception:  # noqa: BLE001
        instance = None
except ImportError:
    token, instance = None, None                          # elsewhere: a saved account is used
service = (QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)
           if token else QiskitRuntimeService())
backend = service.least_busy(operational=True, simulator=False, dynamic_circuits=True)
t = backend.target
print("backend:", backend.name)

two_q = "cz" if "cz" in t.operation_names else "ecr"


def err2(a, b):
    p = t[two_q].get((a, b)) or t[two_q].get((b, a))
    return p.error if p is not None else None


med_2q = float(np.median([p.error for p in t[two_q].values() if p is not None and p.error is not None]))
med_ro = float(np.median([t["measure"][(q,)].error for q in range(backend.num_qubits) if t["measure"].get((q,))]))
nbrs = {q: set() for q in range(backend.num_qubits)}
for a, b in t[two_q].keys():
    nbrs[a].add(b)
    nbrs[b].add(a)


def chains(start):
    stack = [[start]]
    while stack:
        path = stack.pop()
        if len(path) == 5:
            yield path
            continue
        for n in nbrs[path[-1]]:
            if n not in path:
                stack.append(path + [n])


best, best_score = None, 1e9                              # a chain with errors close to the device medians
for s in range(backend.num_qubits):
    for path in chains(s):
        errs = [err2(a, b) for a, b in zip(path, path[1:])]
        if None in errs or any(e >= 0.05 for e in errs):
            continue
        ro = [t["measure"][(q,)].error for q in path]
        score = sum(abs(np.log(e / med_2q)) for e in errs) + sum(abs(np.log(max(r, 1e-4) / med_ro)) for r in ro)
        if score < best_score:
            best, best_score = path, score
p0, p1, p2, p3, p4 = best
layout = [p0, p2, p4, p1, p3]
print("chain:", best)


def circuit(logical):
    q = QuantumRegister(5, "q")
    regs = [ClassicalRegister(2, f"round{r + 1}") for r in range(ROUNDS)]
    final = ClassicalRegister(3, "data")
    qc = QuantumCircuit(q, *regs, final)
    if logical:
        qc.x([0, 1, 2])
    for r in range(ROUNDS):
        qc.barrier()
        qc.cx(0, 3); qc.cx(1, 3); qc.cx(1, 4); qc.cx(2, 4)
        qc.measure(3, regs[r][0]); qc.measure(4, regs[r][1])
        qc.reset(3); qc.reset(4)
    qc.barrier()
    qc.measure([0, 1, 2], final)
    return qc


pm = generate_preset_pass_manager(backend=backend, optimization_level=1, initial_layout=layout)
isa = [pm.run(circuit(0)), pm.run(circuit(1))]
job = SamplerV2(mode=backend).run(isa, shots=SHOTS)
print("job:", job.job_id())
res = job.result()

out = {}
for logical, pub in enumerate(res):
    d = pub.data
    rounds = [d[f"round{r + 1}"].get_bitstrings() for r in range(ROUNDS)]
    data = d["data"].get_bitstrings()
    conv = {}
    for i in range(SHOTS):
        k = " ".join([rounds[r][i][::-1] for r in range(ROUNDS)] + [data[i][::-1]])
        conv[k] = conv.get(k, 0) + 1
    out[f"counts_logical_{logical}"] = dict(sorted(conv.items(), key=lambda kv: -kv[1]))
    out[f"two_qubit_gates_logical_{logical}"] = int(isa[logical].count_ops().get(two_q, 0))

result = {"backend": backend.name, "kind": f"hardware run on {backend.name}, job {job.job_id()}",
          "calibration_date": str(date.today()), "source": "qiskit-ibm-runtime SamplerV2",
          "physical_qubits": {"data": [p0, p2, p4], "ancilla": [p1, p3]},
          "cz_error": [round(err2(a, b), 5) for a, b in zip(best, best[1:])],
          "readout_error": {str(q): round(t["measure"][(q,)].error, 5) for q in best},
          "device_median_cz_error": round(med_2q, 5), "device_median_readout_error": round(med_ro, 5),
          "rounds": ROUNDS, "shots": SHOTS,
          "key_format": "r1 r2 r3 data; each round 's1s2' (s1 = parity of data qubits 0 and 1, s2 = parity of data qubits 1 and 2); data 'd0d1d2'",
          **out}
with open("l3_m2_device_run.json", "w") as f:
    json.dump(result, f, indent=1)
print("wrote l3_m2_device_run.json")
