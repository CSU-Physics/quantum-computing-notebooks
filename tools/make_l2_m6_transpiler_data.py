"""Build content/level2/m6_transpiler_runs.json for the Level 2 Module 6 transpiler project.

Run with the versions the Colab notebook pins: qiskit 2.5.2, qiskit-aer 0.17.2, qiskit-ibm-runtime 0.50.0.
The file holds the transpiler and noisy-simulator results of the course's mirror circuits (a QFT, a barrier, and the
inverse QFT, on a basis state) on FakePittsburgh, so that learners without a Google account can do the same analysis
in the browser, and so that the Colab and browser versions of the project give the same quiz answers.
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import qiskit, qiskit_aer, qiskit_ibm_runtime
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakePittsburgh

assert (qiskit.__version__, qiskit_aer.__version__, qiskit_ibm_runtime.__version__) == ("2.5.2", "0.17.2", "0.50.0")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "content" / "level2" / "m6_transpiler_runs.json"
SOLUTIONS = ROOT / "l2_m6_transpiler_solutions.json"      # kept out of the repository (.gitignore: *solution*)
SHOTS = 4000
WIDTHS = {3: "101", 4: "1011", 5: "10110", 6: "101101"}
LEVELS = [0, 1, 2, 3]
SEEDS = list(range(1, 11))


def qft(qc, qubits):
    """The quantum Fourier transform on the given qubits (Qiskit's convention, with the final swaps)."""
    qubits = list(qubits)
    n = len(qubits)
    for j in reversed(range(n)):
        qc.h(qubits[j])
        for k in reversed(range(j)):
            qc.cp(np.pi / 2 ** (j - k), qubits[k], qubits[j])
    for j in range(n // 2):
        qc.swap(qubits[j], qubits[n - 1 - j])


def inverse_qft(qc, qubits):
    """The inverse of qft(): the same gates in reverse order with negative angles."""
    qubits = list(qubits)
    n = len(qubits)
    for j in range(n // 2):
        qc.swap(qubits[j], qubits[n - 1 - j])
    for j in range(n):
        for k in range(j):
            qc.cp(-np.pi / 2 ** (j - k), qubits[k], qubits[j])
        qc.h(qubits[j])


def mirror_circuit(bits):
    """The course's reference mirror circuit: prepare |bits>, QFT, barrier, inverse QFT, measure qubit q into bit q."""
    n = len(bits)
    qc = QuantumCircuit(n, n)
    ones = [q for q in range(n) if bits[n - 1 - q] == "1"]
    if ones:
        qc.x(ones)
    qft(qc, range(n))
    qc.barrier()
    inverse_qft(qc, range(n))
    qc.measure(range(n), range(n))
    return qc


def run(bits, level, seed, backend, noisy):
    pm = generate_preset_pass_manager(backend=backend, optimization_level=level, seed_transpiler=seed)
    isa = pm.run(mirror_circuit(bits))
    counts = noisy.run(isa, shots=SHOTS, seed_simulator=seed).result().get_counts()
    two_q = sum(1 for i in isa.data if i.operation.num_qubits == 2 and i.operation.name != "barrier")
    return {"level": level, "seed": seed, "success": counts.get(bits, 0), "two_qubit": two_q, "depth": isa.depth(),
            "layout": list(isa.layout.final_index_layout()), "counts": dict(sorted(counts.items()))}


def mirror_without_barrier(bits):
    """The mirror circuit without its barrier, to show what the transpiler does to it."""
    n = len(bits)
    qc = QuantumCircuit(n, n)
    ones = [q for q in range(n) if bits[n - 1 - q] == "1"]
    if ones:
        qc.x(ones)
    qft(qc, range(n))
    inverse_qft(qc, range(n))
    qc.measure(range(n), range(n))
    return qc


def no_barrier_runs(backend, noisy):
    out = []
    for level in LEVELS:
        pm = generate_preset_pass_manager(backend=backend, optimization_level=level, seed_transpiler=1)
        isa = pm.run(mirror_without_barrier("1011"))
        counts = noisy.run(isa, shots=SHOTS, seed_simulator=1).result().get_counts()
        out.append({"level": level, "two_qubit": sum(1 for i in isa.data if i.operation.num_qubits == 2 and i.operation.name != "barrier"),
                    "depth": isa.depth(), "ops": dict(isa.count_ops()), "success": counts.get("1011", 0)})
    return out


def exact_success(isa, bits, backend):
    """The noise model's exact probability that the transpiled circuit records `bits` (no shots).

    A density-matrix run of the circuit without its final measurements gives the probabilities of the measured
    physical qubits; the backend noise model's readout error of each measured qubit is then applied to the recorded
    bits. (This noise model has no quantum error on 'measure' itself, only readout errors.)
    """
    nm = AerSimulator.from_backend(backend).options.noise_model
    dm = AerSimulator.from_backend(backend, method="density_matrix")
    meas = sorted((isa.find_bit(i.clbits[0]).index, isa.find_bit(i.qubits[0]).index) for i in isa.data if i.operation.name == "measure")
    phys = [q for _, q in meas]
    c = isa.copy()
    c.remove_final_measurements()
    c.save_probabilities(qubits=phys)
    pr = np.asarray(dm.run(c).result().data()["probabilities"])
    m = len(phys)
    for j, q in enumerate(phys):
        conf = np.asarray(nm._local_readout_errors[(q,)].probabilities)
        pr = np.einsum("aib,ij->ajb", pr.reshape(2 ** (m - 1 - j), 2, 2 ** j), conf).reshape(-1)
    return float(pr[int(bits, 2)])


if __name__ == "__main__" and "--exact-only" in sys.argv:
    # add the noise model's exact success probability to every saved run (transpiling again with the same seeds)
    backend = FakePittsburgh()
    data = json.loads(OUT.read_text())
    t0 = time.time()
    for r in data["runs"]:
        pm = generate_preset_pass_manager(backend=backend, optimization_level=r["level"], seed_transpiler=r["seed"])
        isa = pm.run(mirror_circuit(r["bits"]))
        assert list(isa.layout.final_index_layout()) == r["layout"] and isa.depth() == r["depth"]
        r["p_exact"] = round(exact_success(isa, r["bits"], backend), 6)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"added p_exact to {len(data['runs'])} runs in {time.time() - t0:.0f} s")
elif __name__ == "__main__" and "--extra-only" in sys.argv:
    # add the no-barrier runs to an existing file without redoing the rest
    backend = FakePittsburgh()
    data = json.loads(OUT.read_text())
    data["no_barrier"] = no_barrier_runs(backend, AerSimulator.from_backend(backend))
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print("added no_barrier:", data["no_barrier"])
elif __name__ == "__main__":
    backend = FakePittsburgh()
    noisy = AerSimulator.from_backend(backend)
    t0 = time.time()
    logical = {}
    for n, bits in WIDTHS.items():
        c = mirror_circuit(bits)
        logical[str(n)] = {"bits": bits, "two_qubit": sum(1 for i in c.data if i.operation.num_qubits == 2 and i.operation.name != "barrier"),
                           "ops": dict(c.count_ops())}
    runs = []
    for n, bits in WIDTHS.items():
        for level in LEVELS:
            for seed in SEEDS:
                r = run(bits, level, seed, backend, noisy)
                r.update({"n": n, "bits": bits})
                pm = generate_preset_pass_manager(backend=backend, optimization_level=level, seed_transpiler=seed)
                r["p_exact"] = round(exact_success(pm.run(mirror_circuit(bits)), bits, backend), 6)
                runs.append(r)
        print(f"n={n} done after {time.time() - t0:.0f} s", flush=True)
    data = {"versions": {"qiskit": qiskit.__version__, "qiskit-aer": qiskit_aer.__version__,
                         "qiskit-ibm-runtime": qiskit_ibm_runtime.__version__},
            "backend": backend.name, "shots": SHOTS, "widths": {str(k): v for k, v in WIDTHS.items()},
            "levels": LEVELS, "seeds": SEEDS, "logical": logical, "runs": runs}
    # personal runs: BITS 0 to 15 (4 qubits), SEED 100 to 999; level 3, seed_transpiler = seed_simulator = SEED
    rnd = random.Random(20261006)
    sets, seen = [], set()
    while len(sets) < 200:
        b, s = rnd.randrange(16), rnd.randrange(100, 1000)
        if (b, s) in seen:
            continue
        seen.add((b, s))
        bits = format(b, "04b")
        r = run(bits, 3, s, backend, noisy)
        del r["counts"]
        sets.append({"bits": b, "seed": s, "value": r["success"], "two_qubit": r["two_qubit"], "layout": r["layout"]})
    print(f"personal sets done after {time.time() - t0:.0f} s", flush=True)
    data["personal"] = [{"bits": d["bits"], "seed": d["seed"], "success": d["value"]} for d in sets]
    data["no_barrier"] = no_barrier_runs(backend, noisy)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    SOLUTIONS.write_text(json.dumps(sets, indent=1))
    print("wrote", OUT, OUT.stat().st_size, "bytes and", SOLUTIONS)
