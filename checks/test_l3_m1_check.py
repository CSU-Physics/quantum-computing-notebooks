"""Tests for checks/l3_m1_check.py: correct and wrong versions of the four functions, the closed form of the
verification value, and agreement with Qiskit Aer's thermal_relaxation_error and with qsim.
Run: PYTHONPATH=content/level3:checks python checks/test_l3_m1_check.py"""
import numpy as np
import l3_m1_check as chk
from l3_m1_check import check_l3_module1, amplitude_damping_kraus, phase_damping_kraus

X, Y, Z = chk.X, chk.Y, chk.Z


# ---------------------------------------------------------------- correct versions
def dm_ok(psi):
    psi = np.asarray(psi, dtype=complex)
    return np.outer(psi, psi.conj())


def dm_ok2(psi):                        # column vector times its dagger
    v = np.asarray(psi, dtype=complex).reshape(-1, 1)
    return v @ v.conj().T


def bloch_ok(rho):
    return np.array([np.trace(rho @ P).real for P in (X, Y, Z)])


def bloch_ok2(rho):                     # from the matrix elements
    rho = np.asarray(rho)
    return [2 * rho[0, 1].real, -2 * rho[0, 1].imag, (rho[0, 0] - rho[1, 1]).real]


def ch_ok(rho, kraus):
    return sum(K @ rho @ K.conj().T for K in kraus)


def ch_ok2(rho, kraus):
    out = np.zeros_like(np.asarray(rho, dtype=complex))
    for K in kraus:
        out = out + np.asarray(K) @ np.asarray(rho) @ np.asarray(K).conj().T
    return out


def relax_ok(rho, t, T1, T2):
    rho = ch_ok(rho, amplitude_damping_kraus(1 - np.exp(-t / T1)))
    return ch_ok(rho, phase_damping_kraus(1 - np.exp(-2 * t / T2 + t / T1)))


def relax_ok2(rho, t, T1, T2):          # phase damping first: the two channels commute
    rho = ch_ok(rho, phase_damping_kraus(1 - np.exp(-2 * t / T2 + t / T1)))
    return ch_ok(rho, amplitude_damping_kraus(1 - np.exp(-t / T1)))


def relax_ok3(rho, t, T1, T2):          # by the Bloch-vector formulas
    x, y, z = bloch_ok(rho)
    c = np.exp(-t / T2)
    x, y = x * c, y * c
    z = 1 - (1 - z) * np.exp(-t / T1)
    return 0.5 * (np.eye(2) + x * X + y * Y + z * Z)


# ---------------------------------------------------------------- wrong versions
def dm_noconj(psi): psi = np.asarray(psi, dtype=complex); return np.outer(psi, psi)
def dm_probs(psi): psi = np.asarray(psi, dtype=complex); return np.diag(np.abs(psi) ** 2)
def bloch_ysign(rho): return np.array([np.trace(rho @ X).real, np.trace(rho @ Y.T).real, np.trace(rho @ Z).real])
def bloch_half(rho): return bloch_ok(rho) / 2
def bloch_complex(rho):                  # complex numbers with zero imaginary parts are fine
    return np.array([np.trace(rho @ P) for P in (X, Y, Z)])
def ch_rev(rho, kraus): return sum(K.conj().T @ rho @ K for K in kraus)
def ch_first(rho, kraus): return kraus[0] @ rho @ kraus[0].conj().T
def ch_avg(rho, kraus): return sum(K @ rho @ K.conj().T for K in kraus) / len(kraus)


def relax_wronglam(rho, t, T1, T2):
    rho = ch_ok(rho, amplitude_damping_kraus(1 - np.exp(-t / T1)))
    return ch_ok(rho, phase_damping_kraus(1 - np.exp(-t / T2)))


def relax_adonly(rho, t, T1, T2): return ch_ok(rho, amplitude_damping_kraus(1 - np.exp(-t / T1)))
def notyet(*a): raise NotImplementedError


ARGS = (1.25, 200, 150, 60)
good = [(dm_ok, bloch_complex, ch_ok, relax_ok), (dm_ok, bloch_ok, ch_ok, relax_ok), (dm_ok2, bloch_ok2, ch_ok2, relax_ok2), (dm_ok, bloch_ok2, ch_ok, relax_ok3)]
bad = [(dm_noconj, bloch_ok, ch_ok, relax_ok), (dm_probs, bloch_ok, ch_ok, relax_ok), (dm_ok, bloch_ysign, ch_ok, relax_ok),
       (dm_ok, bloch_half, ch_ok, relax_ok), (dm_ok, bloch_ok, ch_rev, relax_ok),
       (dm_ok, bloch_ok, ch_first, relax_ok), (dm_ok, bloch_ok, ch_avg, relax_ok), (dm_ok, bloch_ok, ch_ok, relax_wronglam),
       (dm_ok, bloch_ok, ch_ok, relax_adonly), (notyet, bloch_ok, ch_ok, relax_ok), (dm_ok, bloch_ok, ch_ok, notyet)]
wrong = 0
for fns in good:
    ok, msgs, v = check_l3_module1(*fns, *ARGS)
    if not ok:
        wrong += 1
        print("GOOD FAILED", [f.__name__ for f in fns], msgs[-1])
for fns in bad:
    ok, msgs, v = check_l3_module1(*fns, *ARGS)
    if ok:
        wrong += 1
        print("BAD PASSED", [f.__name__ for f in fns])
    else:
        print("  bad ->", next(m for m in msgs if "passed" not in m)[:160])
ok, msgs, v = check_l3_module1(dm_ok, bloch_ok, ch_ok, relax_ok, "1.25", 200, 150, 60)
print("value for", ARGS, "=", v)
for a in [(0.2, 200, 150, 60), (1.25, 200, 450, 60), (1.255, 200, 150, 60), (1.25, 200, 150, 300)]:
    ok, msgs, _ = check_l3_module1(dm_ok, bloch_ok, ch_ok, relax_ok, *a)
    assert not ok, a

# closed form
rng = np.random.default_rng(1)
for _ in range(500):
    th = rng.uniform(0.3, 2.8); T1 = rng.uniform(100, 400); T2 = rng.uniform(50, 2 * T1); tm = rng.uniform(20, 200)
    f = chk.personal_fidelity(th, T1, T2, tm)
    g = (1 + np.sin(th) ** 2 * np.exp(-tm / T2) + np.cos(th) * (1 - (1 - np.cos(th)) * np.exp(-tm / T1))) / 2
    assert abs(f - g) < 1e-12
print("closed form agrees (500 random cases)")

# Qiskit Aer
try:
    from qiskit_aer.noise import thermal_relaxation_error
    from qiskit.quantum_info import DensityMatrix, Kraus
    worst = 0
    for _ in range(300):
        T1 = rng.uniform(50, 400); T2 = rng.uniform(10, 2 * T1); t = rng.uniform(0, 300)
        rho = chk._rand_rho(rng, 1)
        aer = DensityMatrix(rho).evolve(Kraus(thermal_relaxation_error(T1, T2, t).to_quantumchannel())).data
        worst = max(worst, np.max(np.abs(aer - chk.reference_relax(rho, t, T1, T2))))
    print("reference_relax vs Aer thermal_relaxation_error: worst difference", worst)
    assert worst < 1e-10
except ImportError:
    print("Qiskit Aer not installed; skipped")

# qsim (as in the browser)
import qsim
worst = 0
for _ in range(300):
    T1 = rng.uniform(50, 400); T2 = rng.uniform(10, 2 * T1); t = rng.uniform(0, 300)
    rho = chk._rand_rho(rng, 1)
    q = chk.reference_apply_channel(rho, qsim.thermal_relaxation_error(T1, T2, t).kraus)
    worst = max(worst, np.max(np.abs(q - chk.reference_relax(rho, t, T1, T2))))
print("reference_relax vs qsim thermal_relaxation_error: worst difference", worst)
assert worst < 1e-10
print("wrong verdicts:", wrong)
assert wrong == 0
