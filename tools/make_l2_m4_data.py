"""Build content/level2/m4_saved_runs.json for the Level 2 Module 4 lab (running on real hardware).

Run with the versions the Colab notebook pins: qiskit 2.5.2, qiskit-aer 0.17.2, qiskit-ibm-runtime 0.50.0.
The file holds the transpiler and noisy-simulator results of the course's reference Grover circuit on FakePittsburgh,
so that learners without a Google account can do the same analysis in the browser, and so that the Colab and the
browser versions of the lab give the same lab-check answers.
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import qiskit, qiskit_aer, qiskit_ibm_runtime
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakePittsburgh

assert (qiskit.__version__, qiskit_aer.__version__, qiskit_ibm_runtime.__version__) == ("2.5.2", "0.17.2", "0.50.0")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level2" / "m4_saved_runs.json"
SHOTS, TEACH_SEED = 4000, 11
# --teach-only: redo only the (fast) transpiler part of "teach" and keep the saved noisy counts and personal runs
TEACH_ONLY = "--teach-only" in sys.argv
OLD = json.loads(OUT.read_text()) if TEACH_ONLY else None


def reference_grover(marked, iterations):
    """The course's reference Grover circuit (the same gates as Module 2's reference, written with Qiskit)."""
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


def instructions(circ):
    return [[i.operation.name, [circ.find_bit(q).index for q in i.qubits]] for i in circ.data]


def summary(isa):
    return {"depth": isa.depth(), "size": isa.size(),
            "two_qubit": sum(1 for i in isa.data if i.operation.num_qubits == 2 and i.operation.name != "barrier"),
            "ops": dict(isa.count_ops()), "layout": list(isa.layout.final_index_layout())}


backend = FakePittsburgh()
noisy = AerSimulator.from_backend(backend)
target = backend.target
t0 = time.time()

cz_err = [p.error for p in target["cz"].values() if p is not None and p.error is not None]
sx_err = [p.error for p in target["sx"].values() if p is not None and p.error is not None]
ro_err = [p.error for p in target["measure"].values() if p is not None and p.error is not None]
t1 = [target.qubit_properties[q].t1 for q in range(backend.num_qubits) if target.qubit_properties[q].t1]
t2 = [target.qubit_properties[q].t2 for q in range(backend.num_qubits) if target.qubit_properties[q].t2]
edges = sorted({tuple(sorted(e)) for e in target["cz"].keys()})
device = {"name": backend.name, "num_qubits": backend.num_qubits, "basis_gates": sorted(backend.operation_names),
          "two_qubit_gate": "cz", "coupled_pairs": len(edges),
          "median_cz_error": statistics.median(cz_err), "median_sx_error": statistics.median(sx_err),
          "median_readout_error": statistics.median(ro_err),
          "median_t1_us": statistics.median(t1) * 1e6, "median_t2_us": statistics.median(t2) * 1e6,
          "neighbours_of_0": sorted({b for a, b in edges if a == 0} | {a for a, b in edges if b == 0}),
          "edges": [list(e) for e in edges]}

marked = "101"
teach = {"marked": marked, "seed": TEACH_SEED, "shots": SHOTS, "ideal": {}, "runs": {}, "instructions": {}}
for t in range(4):
    qc = reference_grover(marked, t)
    c = qc.copy(); c.remove_final_measurements()
    teach["ideal"][str(t)] = float(Statevector(c).probabilities_dict().get(marked, 0.0))
    teach["runs"][str(t)] = {}
    for lvl in range(4):
        pm = generate_preset_pass_manager(backend=backend, optimization_level=lvl, seed_transpiler=TEACH_SEED)
        isa = pm.run(qc)
        s = summary(isa)
        ops = instructions(isa)
        s["cz_errors"] = [target["cz"][tuple(q)].error for name, q in ops if name == "cz"]
        s["readout_errors"] = [target["measure"][(q[0],)].error for name, q in ops if name == "measure"]
        if TEACH_ONLY:
            s["counts"] = OLD["teach"]["runs"][str(t)][str(lvl)]["counts"]
        else:
            s["counts"] = noisy.run(isa, shots=SHOTS, seed_simulator=TEACH_SEED).result().get_counts()
        teach["runs"][str(t)][str(lvl)] = s
        if t == 2 and lvl in (0, 3):
            teach["instructions"][str(lvl)] = instructions(isa)
    print("teach t =", t, round(time.time() - t0, 1), "s", flush=True)
logical = reference_grover(marked, 2)
teach["logical_two_qubit_level_none"] = {"ops": dict(logical.count_ops())}

personal = OLD["personal"] if TEACH_ONLY else []
rng = np.random.default_rng(20261008)
seen = set()
while len(personal) < 200:
    m, seed = int(rng.integers(0, 8)), int(rng.integers(100, 1000))
    if (m, seed) in seen:
        continue
    seen.add((m, seed))
    w = format(m, "03b")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3, seed_transpiler=seed)
    isa = pm.run(reference_grover(w, 2))
    counts = noisy.run(isa, shots=SHOTS, seed_simulator=seed).result().get_counts()
    personal.append({"marked": m, "seed": seed, "two_qubit": summary(isa)["two_qubit"], "layout": summary(isa)["layout"],
                     "counts": counts})
    if len(personal) % 25 == 0:
        print("personal", len(personal), round(time.time() - t0, 1), "s", flush=True)

data = {"about": "Saved results for the Quantum Computing Intermediate Module 4 lab: the course's reference Grover circuit "
                 "transpiled for FakePittsburgh (a calibration snapshot of ibm_pittsburgh) and run on its noise model.",
        "versions": {"qiskit": qiskit.__version__, "qiskit-aer": qiskit_aer.__version__,
                     "qiskit-ibm-runtime": qiskit_ibm_runtime.__version__},
        "device": device, "teach": teach, "personal": personal}
OUT.write_text(json.dumps(data, separators=(",", ":")))
print("wrote", OUT, OUT.stat().st_size, "bytes in", round(time.time() - t0, 1), "s")
