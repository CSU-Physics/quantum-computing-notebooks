"""Saved device data for the Level 3 Module 3 lab: one expectation value, <XX> of a Bell state after a 20 us idle
period, measured raw, with dynamical decoupling (an X echo), and with the CNOT folded 3 and 5 times for zero-noise
extrapolation, plus the four readout-calibration circuits. Run with Qiskit Aer and the FakePittsburgh noise model
(the calibration snapshot of the 156-qubit IBM computer ibm_pittsburgh that ships with qiskit-ibm-runtime) on two
coupled qubits. This is a simulation with a real device's noise, not a real hardware run; the course team can replace
it with a hardware run in the same format (tools/l3_m3_hardware_run.py).
Writes content/level3/l3_m3_device_run.json.

Counts are keyed as Qiskit prints them: two characters, classical bit 1 on the left, bit 0 (qubit 0) on the right."""
import json
from pathlib import Path

import numpy as np
import qiskit_ibm_runtime
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakePittsburgh

ROOT = Path(__file__).resolve().parents[1]
SHOTS, SEED, IDLE_US = 40000, 2027, 20
PAIR = [97, 107]                        # qubit 0 and qubit 1 of the lab; a coupled pair close to the device medians
backend = FakePittsburgh()
t = backend.target
KEYS = ("00", "01", "10", "11")


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
sim = AerSimulator.from_backend(backend)
counts, ops = {}, {}
for i, (name, qc) in enumerate(circuits.items()):
    tqc = transpile(qc, backend, initial_layout=PAIR, optimization_level=0, scheduling_method="alap", seed_transpiler=SEED)
    c = sim.run(tqc, shots=SHOTS, seed_simulator=SEED + i).result().get_counts()
    counts[name] = {k: int(c.get(k, 0)) for k in KEYS}
    ops[name] = int(tqc.count_ops().get("cz", 0))


def cz_err(a, b):
    p = t["cz"].get((a, b)) or t["cz"].get((b, a))
    return p.error


props = {str(q): {"t1_us": round(t.qubit_properties[q].t1 * 1e6, 1), "t2_us": round(t.qubit_properties[q].t2 * 1e6, 1),
                  "readout_error": round(t["measure"][(q,)].error, 5), "x_error": round(t["x"][(q,)].error, 6)} for q in PAIR}
data = {"backend": backend.name, "kind": "simulation with the FakePittsburgh noise model (Qiskit Aer), not a hardware run",
        "calibration_date": str(backend.properties().last_update_date.date()),
        "source": f"qiskit_ibm_runtime {qiskit_ibm_runtime.__version__}, qiskit-aer AerSimulator.from_backend",
        "physical_qubits": PAIR, "qubit_properties": props, "cz_error": round(cz_err(*PAIR), 5),
        "idle_us": IDLE_US, "shots": SHOTS, "cz_gates": ops,
        "key_format": "'b1b0': classical bit 1 (qubit 1) on the left, bit 0 (qubit 0) on the right, as Qiskit prints counts",
        "circuits": {"raw": "H(0), CX(0,1), idle 20 us, H on both, measure both: <XX>",
                     "dd": "as raw, with an X on both qubits after 10 us and after 20 us",
                     "dd_fold3": "as dd, with the CX replaced by three CXs", "dd_fold5": "as dd, with five CXs",
                     "cal_b1b0": "prepare the bit string b1b0 with X gates and measure"},
        "counts": counts}
(ROOT / "content/level3/l3_m3_device_run.json").write_text(json.dumps(data, indent=1))


def parity(c):
    return sum(n * (-1) ** k.count("1") for k, n in c.items()) / SHOTS


A = np.array([[counts[f"cal_{j}"][i] / SHOTS for j in KEYS] for i in KEYS])
mit = []
for name in ("dd", "dd_fold3", "dd_fold5"):
    v = np.linalg.solve(A, np.array([counts[name][k] / SHOTS for k in KEYS]))
    mit.append(float(sum(x * (-1) ** k.count("1") for x, k in zip(v, KEYS))))
lin = float(np.polyfit([1, 3, 5], mit, 1)[-1])
print("pair", PAIR, props, "cz", data["cz_error"], "cz gates", ops)
print(f"raw {parity(counts['raw']):.4f}  dd {parity(counts['dd']):.4f}  dd+readout {mit[0]:.4f}  zne points "
      f"{[round(m, 4) for m in mit]}  linear {lin:.4f}")
