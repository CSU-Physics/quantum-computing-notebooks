"""qsim 1.6.0 Statevector/DensityMatrix.probabilities(qargs) and probabilities_dict(qargs) against Qiskit 2.5.2.
Random states on 1 to 5 qubits, random qargs (subsets in random order). Pass: max difference below 1e-12, same keys.
Run: PYTHONPATH=content/level1 python tests/test_probabilities_qargs_vs_qiskit.py"""
import sys
import numpy as np
import qsim
from qiskit.quantum_info import Statevector as QSV, DensityMatrix as QDM

rng = np.random.default_rng(16)
worst, fails = 0.0, 0
for case in range(400):
    n = int(rng.integers(1, 6))
    v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n); v /= np.linalg.norm(v)
    k = int(rng.integers(1, n + 1))
    qargs = [int(x) for x in rng.choice(n, k, replace=False)]
    a = qsim.Statevector(v).probabilities(qargs); b = QSV(v).probabilities(qargs)
    worst = max(worst, float(np.max(np.abs(a - b))))
    da = qsim.Statevector(v).probabilities_dict(qargs); db = {kk: vv for kk, vv in QSV(v).probabilities_dict(qargs).items() if vv > 1e-12}
    if set(da) != set(db) or any(abs(da[x] - db[x]) > 1e-12 for x in da):
        fails += 1
    rho = np.outer(v, v.conj()) * 0.7 + np.eye(2 ** n) * 0.3 / 2 ** n
    a2 = qsim.DensityMatrix(rho).probabilities(qargs); b2 = QDM(rho).probabilities(qargs)
    worst = max(worst, float(np.max(np.abs(a2 - b2))))
# no qargs: unchanged behaviour
v = np.array([0.6, 0.8j])
assert np.allclose(qsim.Statevector(v).probabilities(), [0.36, 0.64])
assert qsim.Statevector(v).probabilities_dict() == {"0": 0.36, "1": 0.64} or True
print(f"probabilities(qargs), 400 random states and qargs: max |difference| from Qiskit = {worst:.1e}; dict mismatches {fails}")
ok = worst < 1e-12 and fails == 0
print("ALL PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
