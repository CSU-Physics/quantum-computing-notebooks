"""Check cell logic for the Level 2 Module 2 lab (Grover's search), on qsim 1.7.0.

check_l2_module2(oracle, diffuser, grover, MARKED, SEED)
returns (passed, messages, verification_value).

The learner writes three functions:
- oracle(marked): a phase oracle on n = len(marked) qubits that multiplies the amplitude of the basis state
  |marked> by -1 and leaves every other amplitude alone (Qiskit bit order: the bit for qubit q is
  marked[n - 1 - q]);
- diffuser(n): the reflection about the mean, 2|s><s| - I with |s> the equal superposition; an overall sign
  of -1 is allowed, because no measurement can detect it;
- grover(marked, iterations): H on every qubit, then `iterations` rounds of oracle and diffuser, then qubit q
  measured into classical bit q.

The verification value is the total number of shots, out of 5 x 1,000, in which the course's reference Grover
circuit on 4 qubits returns the 4-bit string of MARKED when it runs with 1, 2, 3, 4 and 5 iterations (1,000 shots
each on qsim's AerSimulator, seeds SEED + 1 to SEED + 5). It is printed only when every test passes, and it
depends on the seeded draws, so it cannot be worked out by hand.
"""
import math

import numpy as np
import qsim
from qsim import AerSimulator, Operator, QuantumCircuit


# ---------------------------------------------------------------- reference circuits
def _mcz(qc, qubits):
    """Multi-controlled Z on the given qubits (symmetric: it flips the sign of the all-ones state)."""
    qubits = list(qubits)
    if len(qubits) == 1:
        qc.z(qubits[0])
    elif len(qubits) == 2:
        qc.cz(qubits[0], qubits[1])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])


def reference_oracle(marked):
    n = len(marked)
    qc = QuantumCircuit(n)
    zeros = [q for q in range(n) if marked[n - 1 - q] == "0"]
    if zeros:
        qc.x(zeros)
    _mcz(qc, range(n))
    if zeros:
        qc.x(zeros)
    return qc


def reference_diffuser(n):
    qc = QuantumCircuit(n)
    qc.h(list(range(n)))
    qc.x(list(range(n)))
    _mcz(qc, range(n))
    qc.x(list(range(n)))
    qc.h(list(range(n)))
    return qc


def reference_grover(marked, iterations):
    n = len(marked)
    qc = QuantumCircuit(n, n)
    qc.h(list(range(n)))
    for _ in range(iterations):
        qc.compose(reference_oracle(marked), inplace=True)
        qc.compose(reference_diffuser(n), inplace=True)
    qc.measure(list(range(n)), list(range(n)))
    return qc


def oracle_matrix(marked):
    n = len(marked)
    d = np.ones(2 ** n, dtype=complex)
    d[int(marked, 2)] = -1
    return np.diag(d)


def diffuser_matrix(n):
    N = 2 ** n
    s = np.ones((N, 1)) / math.sqrt(N)
    return 2 * (s @ s.T) - np.eye(N)


def exact_distribution(qc):
    """The exact probability of each result string of a circuit (qsim's branch simulator)."""
    out = {}
    for rho, bits in qsim._branch_states(qc, None):
        w = float(np.real(np.trace(rho)))
        if w > 1e-12:
            key = qsim._format_key(list(bits), qc._cregs)
            out[key] = out.get(key, 0.0) + w
    return out


def personal_value(marked_int, seed):
    marked = format(int(marked_int), "04b")
    total = 0
    for t in range(1, 6):
        counts = AerSimulator(seed_simulator=int(seed) + t).run(reference_grover(marked, t), shots=1000).result().get_counts()
        total += int(counts.get(marked, 0))
    return total


# ---------------------------------------------------------------- tests
def _gates_only(qc, n, name):
    if not isinstance(qc, QuantumCircuit) or qc.num_qubits != n:
        return f"{name} should return a QuantumCircuit with {n} qubits."
    if any(i.name in ("measure", "reset", "if_else") for i in qc.data):
        return f"{name} should contain gates only, no measurements."
    return None


def _diagnose_oracle(m, marked):
    n = len(marked)
    if not np.allclose(m, np.diag(np.diag(m)), atol=1e-8):
        return ("it changes more than signs. An oracle only multiplies amplitudes by -1: check that every X gate "
                "before the multi-controlled Z is undone by the same X gate after it.")
    d = np.real(np.diag(m))
    flipped = [format(i, f"0{n}b") for i in range(2 ** n) if d[i] < 0]
    if len(flipped) == 1:
        f = flipped[0]
        if f == marked[::-1]:
            return (f"it marks {f}, the string in reverse order. In Qiskit, qubit 0 is the rightmost bit, so the bit "
                    f"for qubit q is marked[n - 1 - q].")
        if f == "".join("1" if b == "0" else "0" for b in marked):
            return f"it marks {f}, every bit flipped: the X gates belong on the qubits whose bit is 0."
        return f"it marks {f} instead of {marked}: check which qubits get X gates."
    if not flipped:
        return "no amplitude changes sign: the multi-controlled Z on all the qubits is missing."
    return f"it changes the sign of {len(flipped)} states; it should change only |{marked}>."


ORACLE_STRINGS = [format(i, f"0{n}b") for n in (2, 3, 4) for i in range(2 ** n)]


def _test_oracle(oracle):
    note = False
    for marked in ORACLE_STRINGS:
        try:
            qc = oracle(marked)
        except NotImplementedError:
            return False, ["oracle() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"oracle('{marked}') raised {type(e).__name__}: {e}"]
        bad = _gates_only(qc, len(marked), f"oracle('{marked}')")
        if bad:
            return False, [bad]
        m = Operator(qc).data
        want = oracle_matrix(marked)
        if np.allclose(m, want, atol=1e-8):
            continue
        if np.allclose(m, -want, atol=1e-8):
            note = True
            continue
        return False, [f"oracle('{marked}') is not the phase oracle for {marked}: " + _diagnose_oracle(m, marked)]
    msg = f"oracle(): passed (all {len(ORACLE_STRINGS)} strings on 2, 3 and 4 qubits)."
    if note:
        msg += " Your oracle flips every sign except the marked one; that differs only by an overall sign, which Grover's search cannot see."
    return True, [msg]


def _test_diffuser(diffuser):
    for n in (2, 3, 4, 5):
        try:
            qc = diffuser(n)
        except NotImplementedError:
            return False, ["diffuser() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"diffuser({n}) raised {type(e).__name__}: {e}"]
        bad = _gates_only(qc, n, f"diffuser({n})")
        if bad:
            return False, [bad]
        m = Operator(qc).data
        want = diffuser_matrix(n)
        if np.allclose(m, want, atol=1e-8) or np.allclose(m, -want, atol=1e-8):
            continue
        N = 2 ** n
        allones = np.zeros((N, 1)); allones[-1] = 1
        Hn = reference_hadamards(n)
        no_x = Hn @ (np.eye(N) - 2 * allones @ allones.T) @ Hn
        zero = np.zeros((N, 1)); zero[0] = 1
        no_h = np.eye(N) - 2 * zero @ zero.T
        if np.allclose(m, no_x, atol=1e-8) or np.allclose(m, -no_x, atol=1e-8):
            hint = "put X gates on every qubit just before and just after the multi-controlled Z (inside the H layers)."
        elif np.allclose(m, no_h, atol=1e-8) or np.allclose(m, -no_h, atol=1e-8):
            hint = "start and end with an H gate on every qubit."
        else:
            hint = "the pattern is H on all, X on all, multi-controlled Z, X on all, H on all."
        return False, [f"diffuser({n}) is not the reflection about the mean: {hint}"]
    return True, ["diffuser(): passed (n = 2 to 5, the reflection about the mean up to an overall sign)."]


def reference_hadamards(n):
    h = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    out = np.array([[1.0]])
    for _ in range(n):
        out = np.kron(out, h)
    return out


GROVER_CASES = [("101", 2), ("0110", 3), ("010", 3), ("111", 1), ("0011", 2), ("1001", 0), ("11", 1), ("10", 1)]


def _variant(marked, its, start_h=True, diffuse=True):
    n = len(marked)
    qc = QuantumCircuit(n, n)
    if start_h:
        qc.h(list(range(n)))
    for _ in range(its):
        qc.compose(reference_oracle(marked), inplace=True)
        if diffuse:
            qc.compose(reference_diffuser(n), inplace=True)
    qc.measure(list(range(n)), list(range(n)))
    return qc


def _tvd(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)


def _test_grover(grover):
    for marked, its in GROVER_CASES:
        n = len(marked)
        try:
            qc = grover(marked, its)
        except NotImplementedError:
            return False, ["grover() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"grover('{marked}', {its}) raised {type(e).__name__}: {e}"]
        if not isinstance(qc, QuantumCircuit) or qc.num_qubits != n or qc.num_clbits != n:
            return False, [f"grover('{marked}', {its}) should return a circuit with {n} qubits and {n} classical bits."]
        meas = [i for i in qc.data if i.name == "measure"]
        if sorted(i.qubits[0] for i in meas) != list(range(n)) or any(i.qubits[0] != i.clbits[0] for i in meas):
            return False, ["Measure each qubit q into classical bit q, once, at the end."]
        got = exact_distribution(qc)
        ref = exact_distribution(reference_grover(marked, its))
        if _tvd(got, ref) > 1e-6:
            hint = ""
            candidates = [(_variant(marked, its, start_h=False), " Start with an H gate on every qubit."),
                          (_variant(marked, its, diffuse=False), " Each iteration needs the diffuser after the oracle."),
                          (_variant(marked, its - 1) if its > 0 else None,
                           " It looks like one iteration too few: run the oracle and the diffuser `iterations` times."),
                          (_variant(marked, its + 1), " It looks like one iteration too many.")]
            for cand, h in candidates:
                if cand is not None and _tvd(got, exact_distribution(cand)) < 1e-6:
                    hint = h
                    break
            best = max(got, key=got.get) if got else ""
            if not hint and best == marked[::-1] != marked:
                hint = (f" The most likely result is {best}, the marked string reversed: in Qiskit the bit for qubit q "
                        f"is marked[n - 1 - q].")
            return False, [f"grover('{marked}', {its}) gives the marked string with probability {got.get(marked, 0.0):.4f}; "
                           f"it should be {ref.get(marked, 0.0):.4f}.{hint}"]
    return True, [f"grover(): passed ({len(GROVER_CASES)} cases on 2 to 4 qubits, 0 to 3 iterations; each distribution exact)."]


def exact_success(n, t):
    th = math.asin(math.sqrt(1 / 2 ** n))
    return math.sin((2 * t + 1) * th) ** 2


def check_l2_module2(oracle, diffuser, grover, MARKED, SEED):
    try:
        marked, seed = int(MARKED), int(SEED)
        if float(MARKED) != marked or float(SEED) != seed:
            raise ValueError
    except (TypeError, ValueError):
        return False, ["MARKED and SEED must be the whole numbers shown in your Canvas lab check."], None
    if not (0 <= marked <= 15) or not (100 <= seed <= 999):
        return False, ["MARKED and SEED are outside the range Canvas uses. Copy them again from the lab check."], None
    msgs, ok = [], True
    for test, fn in ((_test_oracle, oracle), (_test_diffuser, diffuser), (_test_grover, grover)):
        good, m = test(fn)
        msgs += m
        ok = ok and good
    if not ok:
        return False, msgs, None
    return True, msgs, personal_value(marked, seed)
