"""qsim's noise models against Qiskit Aer (qiskit_aer.noise), for the Module 5 lab.

1. Density matrices: random 1- to 3-qubit circuits with random noise models (depolarizing,
   Pauli, amplitude damping, phase damping and thermal relaxation errors, all-qubit and
   qubit-specific) give the same final density matrix as Aer's density-matrix method.
2. Counts: circuits with final measurements, quantum errors on 'measure' and readout errors
   give the same result distribution as Aer (qsim's exact distribution against 200,000 Aer shots).
3. Mid-circuit measurements and resets with noise: qsim's shot-by-shot simulation against Aer.

Run: PYTHONPATH=content/level1 python tests/test_noise_vs_aer.py
"""
import math

import numpy as np
from qiskit import QuantumCircuit as QQC
from qiskit_aer import AerSimulator as AAer
from qiskit_aer import noise as an

import qsim

rng = np.random.default_rng(2026)
ONE = ["h", "x", "sx", "s", "t", "ry", "rz"]
TWO = ["cx", "cz"]


def rand_error(n, kind):
    if kind == "dep":
        return ("dep", float(rng.uniform(0, 0.4)), n)
    if kind == "pauli":
        import itertools
        labs = ["".join(p) for p in itertools.product("IXYZ", repeat=n)]
        pick = list(rng.choice(labs, size=3, replace=False))
        w = rng.dirichlet([1, 1, 1])
        return ("pauli", list(zip(pick, map(float, w))))
    if kind == "amp":
        return ("amp", float(rng.uniform(0, 0.5)))
    if kind == "phase":
        return ("phase", float(rng.uniform(0, 0.5)))
    t1 = float(rng.uniform(50, 300))
    t2 = float(rng.uniform(0.2, 2.0) * t1)
    return ("thermal", t1, t2, float(rng.uniform(0.05, 60)))


def build(spec, mod):
    k = spec[0]
    if k == "dep":
        return mod.depolarizing_error(spec[1], spec[2])
    if k == "pauli":
        return mod.pauli_error(spec[1])
    if k == "amp":
        return mod.amplitude_damping_error(spec[1])
    if k == "phase":
        return mod.phase_damping_error(spec[1])
    return mod.thermal_relaxation_error(spec[1], spec[2], spec[3])


def rand_model(n):
    """A random noise model as a list of ('all' | 'local', spec, instructions, qubits) entries."""
    entries = []
    for g in ONE:
        if rng.random() < 0.5:
            entries.append(("all", rand_error(1, str(rng.choice(["dep", "pauli", "amp", "phase", "thermal"]))), [g], None))
    if n >= 2:
        for g in TWO:
            if rng.random() < 0.7:
                entries.append(("all", rand_error(2, str(rng.choice(["dep", "pauli"]))), [g], None))
        if rng.random() < 0.5:
            a, b = (int(x) for x in rng.choice(n, size=2, replace=False))
            entries.append(("local", rand_error(2, "dep"), ["cx"], [a, b]))
    if rng.random() < 0.5:
        q = int(rng.integers(n))
        entries.append(("local", rand_error(1, "amp"), ["h"], [q]))
    return entries


def make_models(entries, readout=None, meas_err=None):
    qm, am = qsim.NoiseModel(), an.NoiseModel()
    for scope, spec, ins, qs in entries:
        if scope == "all":
            qm.add_all_qubit_quantum_error(build(spec, qsim), ins)
            am.add_all_qubit_quantum_error(build(spec, an), ins)
        else:
            qm.add_quantum_error(build(spec, qsim), ins, qs)
            am.add_quantum_error(build(spec, an), ins, qs)
    if meas_err is not None:
        qm.add_all_qubit_quantum_error(build(meas_err, qsim), ["measure"])
        am.add_all_qubit_quantum_error(build(meas_err, an), ["measure"])
    if readout is not None:
        for q, probs in readout.items():
            if q == "all":
                qm.add_all_qubit_readout_error(qsim.ReadoutError(probs))
                am.add_all_qubit_readout_error(an.ReadoutError(probs))
            else:
                qm.add_readout_error(qsim.ReadoutError(probs), [q])
                am.add_readout_error(an.ReadoutError(probs), [q])
    return qm, am


def rand_circuit(n, depth):
    ops = []
    for _ in range(depth):
        if n >= 2 and rng.random() < 0.35:
            a, b = (int(x) for x in rng.choice(n, size=2, replace=False))
            ops.append((str(rng.choice(TWO)), [a, b], None))
        else:
            g = str(rng.choice(ONE))
            ops.append((g, [int(rng.integers(n))], float(rng.uniform(-3, 3)) if g in ("ry", "rz") else None))
    return ops


def to(ops, n, cls, nc=0):
    qc = cls(n, nc) if nc else cls(n)
    for g, qs, p in ops:
        getattr(qc, g)(*([p] if p is not None else []), *qs)
    return qc


def rand_readout(n):
    out = {}
    if rng.random() < 0.5:
        e0, e1 = rng.uniform(0, 0.15, size=2)
        out["all"] = [[1 - e0, e0], [e1, 1 - e1]]
    for q in range(n):
        if rng.random() < 0.5:
            e0, e1 = rng.uniform(0, 0.15, size=2)
            out[q] = [[1 - e0, e0], [e1, 1 - e1]]
    return out


worst_rho = 0.0
for trial in range(200):
    n = int(rng.integers(1, 4))
    ops = rand_circuit(n, int(rng.integers(3, 12)))
    entries = rand_model(n)
    qm, am = make_models(entries)
    rq = qsim.simulate_density_matrix(to(ops, n, qsim.QuantumCircuit), noise_model=qm).data
    aqc = to(ops, n, QQC)
    aqc.save_density_matrix()
    ra = np.asarray(AAer(method="density_matrix", noise_model=am).run(aqc).result().data()["density_matrix"])
    worst_rho = max(worst_rho, float(np.max(np.abs(rq - ra))))
print(f"1. density matrices, 200 random noisy circuits: largest difference {worst_rho:.1e}")
assert worst_rho < 1e-10

SHOTS = 200_000
worst_tvd, limit = 0.0, 0.0
for trial in range(60):
    n = int(rng.integers(1, 4))
    ops = rand_circuit(n, int(rng.integers(3, 10)))
    entries = rand_model(n)
    ro = rand_readout(n)
    me = rand_error(1, "dep") if rng.random() < 0.4 else None
    qm, am = make_models(entries, ro, me)
    qq = to(ops, n, qsim.QuantumCircuit, n)
    qq.measure(list(range(n)), list(range(n)))
    aq = to(ops, n, QQC, n)
    aq.measure(list(range(n)), list(range(n)))
    dist, meas = qsim._noisy_distribution(qq, qm)
    pq = {format(sum(((o >> j) & 1) << c for j, (_, c) in enumerate(meas)), f"0{n}b"): p for o, p in enumerate(dist)}
    ca = AAer(noise_model=am, seed_simulator=trial).run(aq, shots=SHOTS).result().get_counts()
    keys = set(pq) | set(ca)
    tvd = 0.5 * sum(abs(pq.get(k, 0) - ca.get(k, 0) / SHOTS) for k in keys)
    lim = 0.5 * sum(math.sqrt(pq.get(k, 0) * (1 - pq.get(k, 0)) / SHOTS) for k in keys) * 5 + 1e-4
    worst_tvd, limit = max(worst_tvd, tvd), max(limit, lim)
    assert tvd < lim, (trial, tvd, lim)
print(f"2. counts with readout and measure errors, 60 circuits x {SHOTS:,} Aer shots: largest total variation "
      f"distance {worst_tvd:.4f} (each within its 5-sigma shot-noise bound)")

# qsim's own sampling matches its exact distribution
qm, _ = make_models([("all", ("dep", 0.2, 2), ["cx"], None)], {"all": [[0.97, 0.03], [0.05, 0.95]]})
qc = qsim.QuantumCircuit(2, 2)
qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])
c = qsim.AerSimulator(seed_simulator=1, noise_model=qm).run(qc, shots=SHOTS).result().get_counts()
dist, _ = qsim._noisy_distribution(qc, qm)
tv = 0.5 * sum(abs(dist[int(k[::-1], 2)] - c.get(k, 0) / SHOTS) for k in ["00", "01", "10", "11"])
print(f"   qsim sampling against its own exact distribution: total variation distance {tv:.4f}")
assert tv < 0.005

worst, aer_retries = 0.0, 0
for trial in range(20):
    n = 2
    qm, am = make_models([e for e in rand_model(n) if e[0] == "all"], rand_readout(n), rand_error(1, "dep"))
    for lib, Q in (("q", qsim.QuantumCircuit), ("a", QQC)):
        qc = Q(2, 3)
        qc.h(0); qc.cx(0, 1); qc.measure(0, 0); qc.ry(0.7, 0); qc.reset(1); qc.cx(0, 1); qc.measure([0, 1], [1, 2])
        if lib == "q":
            cq = qsim.AerSimulator(seed_simulator=trial, noise_model=qm).run(qc, shots=20000).result().get_counts()
        else:
            ca = None
            for opts in ({}, {"fusion_enable": False}, {"method": "density_matrix"}):
                try:
                    ca = AAer(noise_model=am, seed_simulator=trial, **opts).run(qc, shots=200000).result().get_counts()
                    break
                except RuntimeError:
                    aer_retries += 1
    keys = set(cq) | set(ca)
    tvd = 0.5 * sum(abs(cq.get(k, 0) / 20000 - ca.get(k, 0) / 200000) for k in keys)
    worst = max(worst, tvd)
    assert tvd < 0.03, (trial, tvd)
print(f"3. mid-circuit measurement and reset with noise, 20 models: largest total variation distance {worst:.4f}"
      f" (Aer needed a second setting {aer_retries} time(s))")

# (Part 3: Aer 0.17.2 sometimes stops with an internal eigen-solver error ("heevx ... Check that
# input matrix is really hermitian") for noisy circuits with mid-circuit measurement, even for
# errors on gates the circuit does not use. The test then reruns Aer with gate fusion off, or with
# the density-matrix method, and counts the retries. Part 3 uses all-qubit errors; parts 1 and 2
# test qubit-specific errors.)

# the noiseless path is unchanged: Module 4's seeded counts
qc = qsim.QuantumCircuit(2, 2)
qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])
assert qsim.AerSimulator(seed_simulator=7).run(qc, shots=1000).result().get_counts() == {"00": 502, "11": 498}
assert qsim.AerSimulator(seed_simulator=7, noise_model=qsim.NoiseModel()).run(qc, shots=1000).result().get_counts() == {"00": 502, "11": 498}
print("4. without noise (or with an empty NoiseModel) the seeded counts are unchanged")
print("ALL PASSED")
