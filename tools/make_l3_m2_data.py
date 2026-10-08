"""Saved syndrome data for the Level 3 Module 2 lab: the three-qubit bit-flip repetition code, three rounds of
syndrome measurement, logical |0> and logical |1>, run with Qiskit Aer and the FakePittsburgh noise model (the
calibration snapshot of the 156-qubit IBM computer ibm_pittsburgh that ships with qiskit-ibm-runtime).
This is a simulation with a real device's noise, not a real hardware run; the course team can replace it with a
hardware run in the same format (tools/l3_m2_hardware_run.py).
Writes content/level3/l3_m2_device_run.json.

Format: counts keyed "r1 r2 r3 data". Each round is two characters "s1s2": s1 = parity of data qubits 0 and 1,
s2 = parity of data qubits 1 and 2. data is "d0d1d2", the final measurement of data qubits 0, 1, 2 (left to right).
No correction is applied during the run: the rounds only record syndromes, as in repetition-code experiments."""
import json
from pathlib import Path

import numpy as np
import qiskit_ibm_runtime
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakePittsburgh

ROOT = Path(__file__).resolve().parents[1]
ROUNDS, SHOTS, SEED = 3, 4000, 2026
backend = FakePittsburgh()
t = backend.target


def cz_err(a, b):
    p = t["cz"].get((a, b)) or t["cz"].get((b, a))
    return p.error if p is not None else None


# a chain p0 - p1 - p2 - p3 - p4 whose two-qubit and readout errors are close to the device medians
med_cz = float(np.median([p.error for p in t["cz"].values() if p is not None and p.error is not None]))
med_ro = float(np.median([t["measure"][(q,)].error for q in range(backend.num_qubits)]))
nbrs = {q: set() for q in range(backend.num_qubits)}
for a, b in t["cz"].keys():
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


best, best_score = None, 1e9
for s in range(backend.num_qubits):
    for path in chains(s):
        errs = [cz_err(a, b) for a, b in zip(path, path[1:])]
        ro = [t["measure"][(q,)].error for q in path]
        t1 = [t.qubit_properties[q].t1 for q in path]
        if None in errs or min(t1) < 100e-6:
            continue
        score = sum(abs(np.log(e / med_cz)) for e in errs) + sum(abs(np.log(r / med_ro)) for r in ro)
        if score < best_score:
            best, best_score = path, score
p0, p1, p2, p3, p4 = best
layout = [p0, p2, p4, p1, p3]          # data 0, 1, 2 on p0, p2, p4; ancilla 3 between data 0 and 1, ancilla 4 between 1 and 2


def circuit(logical):
    q = QuantumRegister(5, "q")
    regs = [ClassicalRegister(2, f"round{r + 1}") for r in range(ROUNDS)]
    final = ClassicalRegister(3, "data")
    qc = QuantumCircuit(q, *regs, final)
    if logical:                        # |1_L> = |111>; a basis state needs no entangling encoder, so no routing
        qc.x([0, 1, 2])
    for r in range(ROUNDS):
        qc.barrier()
        qc.cx(0, 3)
        qc.cx(1, 3)
        qc.cx(1, 4)
        qc.cx(2, 4)
        qc.measure(3, regs[r][0])
        qc.measure(4, regs[r][1])
        qc.reset(3)
        qc.reset(4)
    qc.barrier()
    qc.measure([0, 1, 2], final)
    return qc


sim = AerSimulator.from_backend(backend)
out = {}
for logical in (0, 1):
    tqc = transpile(circuit(logical), backend, initial_layout=layout, optimization_level=1, seed_transpiler=SEED)
    counts = sim.run(tqc, shots=SHOTS, seed_simulator=SEED + logical).result().get_counts()
    conv = {}
    for key, n in counts.items():
        parts = key.split()              # Qiskit: last register first, bit 0 of each register on the right
        data = parts[0][::-1]
        rounds = [p[::-1] for p in reversed(parts[1:])]
        k = " ".join(rounds + [data])
        conv[k] = conv.get(k, 0) + n
    out[f"counts_logical_{logical}"] = dict(sorted(conv.items(), key=lambda kv: -kv[1]))
    out[f"two_qubit_gates_logical_{logical}"] = int(tqc.count_ops().get("cz", 0))

data = {"backend": backend.name, "kind": "simulation with the FakePittsburgh noise model (Qiskit Aer), not a hardware run",
        "calibration_date": str(backend.properties().last_update_date.date()),
        "source": f"qiskit_ibm_runtime {qiskit_ibm_runtime.__version__}, qiskit-aer AerSimulator.from_backend",
        "physical_qubits": {"data": [p0, p2, p4], "ancilla": [p1, p3]},
        "cz_error": [round(cz_err(a, b), 5) for a, b in zip(best, best[1:])],
        "readout_error": {str(q): round(t["measure"][(q,)].error, 5) for q in best},
        "device_median_cz_error": round(med_cz, 5), "device_median_readout_error": round(med_ro, 5),
        "rounds": ROUNDS, "shots": SHOTS,
        "key_format": "r1 r2 r3 data; each round 's1s2' (s1 = parity of data qubits 0 and 1, s2 = parity of data qubits 1 and 2); data 'd0d1d2'",
        **out}
(ROOT / "content/level3/l3_m2_device_run.json").write_text(json.dumps(data, indent=1))
print("chain", best, "cz errors", data["cz_error"], "readout", data["readout_error"], "cz gates", out["two_qubit_gates_logical_0"])
for logical in (0, 1):
    c = out[f"counts_logical_{logical}"]
    r1 = sum(n for k, n in c.items() if k.split()[0] == "00") / SHOTS
    maj = sum(n for k, n in c.items() if (k.split()[-1].count("1") >= 2) != bool(logical)) / SHOTS
    print(f"logical {logical}: {len(c)} distinct keys; round 1 syndrome 00 in {r1:.4f}; majority-vote error {maj:.4f}")
