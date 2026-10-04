"""qsim.partial_trace (1.5.0) against qiskit.quantum_info.partial_trace on 300 random states.
Run: PYTHONPATH=content/level1 python tests/test_partial_trace_vs_qiskit.py"""
import sys
import numpy as np
import qsim
from qiskit.quantum_info import partial_trace as qpt, random_statevector, random_density_matrix

rng = np.random.default_rng(1)
worst = 0
for k in range(300):
    n = int(rng.integers(1, 5))
    st = random_statevector(2 ** n, seed=k) if k % 2 else random_density_matrix(2 ** n, seed=k)
    qargs = [int(q) for q in rng.choice(n, size=int(rng.integers(0, n)), replace=False)]
    a = np.asarray(qsim.partial_trace(np.asarray(st.data), qargs))
    b = np.asarray(qpt(st, qargs).data)
    worst = max(worst, float(np.max(np.abs(a - b))))
print(f"partial_trace, 300 random states: max |difference| from Qiskit = {worst:.1e}")
print("ALL PASSED" if worst < 1e-12 else "FAILED")
sys.exit(0 if worst < 1e-12 else 1)
