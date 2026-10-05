"""Check cell logic for the Level 2 Module 4 lab (running on real hardware), for both versions of the lab:
the Google Colab version (real Qiskit) and the browser version (qsim with saved transpiler results).

check_l2_module4(grover_circuit, two_qubit_count, success_rate, MARKED, SEED, saved)
returns (passed, messages, verification_value).

The learner writes three functions:
- grover_circuit(marked, iterations): Grover's search from Module 2 as one circuit (H on every qubit, the rounds of
  oracle and diffuser, qubit q measured into classical bit q);
- two_qubit_count(ops): the number of two-qubit gates in a list of instructions [(name, [qubits]), ...], not counting
  barriers (a barrier can span two qubits but is not a gate);
- success_rate(counts, marked): the fraction of shots that gave the marked string.

`saved` is the content of m4_saved_runs.json (results of the course's reference circuit on FakePittsburgh, made with
qiskit 2.5.2, qiskit-aer 0.17.2 and qiskit-ibm-runtime 0.50.0). The verification value is the number of shots, out of
4,000, in which the reference Grover circuit for the 3-bit string of MARKED with 2 iterations, transpiled at
optimization level 3 with seed SEED, found MARKED on the FakePittsburgh noise model with seed SEED. Both versions of
the lab read it from the same saved run, so they give the same value.
"""
try:                                    # Google Colab: real Qiskit
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
except ImportError:                     # browser: the course simulator
    from qsim import QuantumCircuit, Statevector


def _ideal(qc):
    c = qc.copy()
    c.remove_final_measurements()
    return {k: float(v) for k, v in Statevector(c).probabilities_dict().items() if v > 1e-12}


def reference_grover(marked, iterations):
    n = len(marked)
    qc = QuantumCircuit(n, n)
    qc.h(list(range(n)))
    for _ in range(iterations):
        zeros = [q for q in range(n) if marked[n - 1 - q] == "0"]
        if zeros:
            qc.x(zeros)
        qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
        if zeros:
            qc.x(zeros)
        qc.h(list(range(n))); qc.x(list(range(n)))
        qc.h(n - 1); qc.mcx(list(range(n - 1)), n - 1); qc.h(n - 1)
        qc.x(list(range(n))); qc.h(list(range(n)))
    qc.measure(list(range(n)), list(range(n)))
    return qc


def _num_qubits_of(inst):
    try:
        return inst.operation.num_qubits, inst.qubits      # Qiskit CircuitInstruction
    except AttributeError:
        return len(inst.qubits), inst.qubits                # qsim Instruction


GROVER_CASES = [("101", 2), ("011", 1), ("110", 3), ("000", 2), ("11", 1), ("1001", 3), ("0110", 0)]


def _test_grover(grover_circuit):
    for marked, its in GROVER_CASES:
        n = len(marked)
        try:
            qc = grover_circuit(marked, its)
        except NotImplementedError:
            return False, ["grover_circuit() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"grover_circuit('{marked}', {its}) raised {type(e).__name__}: {e}"]
        if not isinstance(qc, QuantumCircuit) or qc.num_qubits != n or qc.num_clbits != n:
            return False, [f"grover_circuit('{marked}', {its}) should return a circuit with {n} qubits and {n} classical bits."]
        meas = [i for i in qc.data if getattr(getattr(i, "operation", i), "name", "") == "measure"]
        if len(meas) != n:
            return False, ["Measure each qubit once, at the end: qc.measure(range(n), range(n))."]
        got, ref = _ideal(qc), _ideal(reference_grover(marked, its))
        keys = set(got) | set(ref)
        if 0.5 * sum(abs(got.get(k, 0) - ref.get(k, 0)) for k in keys) > 1e-6:
            return False, [f"grover_circuit('{marked}', {its}) gives the marked string with probability "
                           f"{got.get(marked, 0):.4f}; it should be {ref.get(marked, 0):.4f}. Use your Module 2 oracle and diffuser: "
                           "H on every qubit, then `iterations` rounds of oracle and diffuser, then measure."]
    return True, [f"grover_circuit(): passed ({len(GROVER_CASES)} cases on 2 to 4 qubits, 0 to 3 iterations; each distribution exact)."]


TQ_CASES = [
    ([("cz", [0, 1]), ("rz", [1]), ("sx", [0]), ("cz", [1, 2]), ("measure", [0])], 2),
    ([("barrier", [0, 1]), ("cz", [3, 4]), ("barrier", [0, 1, 2]), ("x", [2])], 1),
    ([("swap", [0, 5]), ("ecr", [1, 2]), ("cx", [2, 3]), ("ccx", [0, 1, 2])], 3),
    ([("rz", [0]), ("sx", [0]), ("measure", [0])], 0),
    ([], 0),
]


def _test_two_qubit(two_qubit_count):
    for ops, want in TQ_CASES:
        try:
            got = two_qubit_count([(n, list(q)) for n, q in ops])
        except NotImplementedError:
            return False, ["two_qubit_count() is not written yet."]
        except Exception as e:  # noqa: BLE001
            return False, [f"two_qubit_count() raised {type(e).__name__}: {e}"]
        if got != want:
            names = [n for n, _ in ops]
            hint = ""
            if "barrier" in names:
                hint = " A barrier can span two qubits, but it is not a gate: leave barriers out."
            elif isinstance(got, (int, float)) and got > want and "ccx" in names:
                hint = " Count only instructions on exactly two qubits (a ccx acts on three)."
            elif any(n in ("swap", "ecr", "cx") for n in names):
                hint = " Count every instruction on exactly two qubits, not only cz: count by the number of qubits, not by name."
            return False, [f"two_qubit_count() returned {got!r} for {names}; it should be {want}.{hint}"]
    return True, [f"two_qubit_count(): passed ({len(TQ_CASES)} instruction lists, with barriers, swaps and a three-qubit gate)."]


SR_CASES = [({"101": 3568, "001": 120, "111": 312}, "101", 0.892), ({"00": 10, "11": 30}, "01", 0.0),
            ({"0110": 1}, "0110", 1.0), ({"10": 250, "00": 250, "01": 250, "11": 250}, "10", 0.25)]


def _test_success(success_rate):
    for counts, marked, want in SR_CASES:
        try:
            got = success_rate(dict(counts), marked)
        except NotImplementedError:
            return False, ["success_rate() is not written yet."]
        except KeyError:
            return False, [f"success_rate() stopped with a KeyError for '{marked}': use counts.get(marked, 0) for a result that never appeared."]
        except Exception as e:  # noqa: BLE001
            return False, [f"success_rate() raised {type(e).__name__}: {e}"]
        try:
            ok = abs(float(got) - want) < 1e-9
        except (TypeError, ValueError):
            ok = False
        if not ok:
            hint = " Divide by the total number of shots, sum(counts.values()), not by a fixed number." if counts and sum(counts.values()) != 4000 else ""
            return False, [f"success_rate() returned {got!r} for marked '{marked}'; it should be {want}.{hint}"]
    return True, [f"success_rate(): passed ({len(SR_CASES)} cases, including a string that never appeared)."]


def personal_value(saved, marked_int, seed):
    for p in saved["personal"]:
        if p["marked"] == int(marked_int) and p["seed"] == int(seed):
            return int(p["counts"].get(format(int(marked_int), "03b"), 0))
    return None


def check_l2_module4(grover_circuit, two_qubit_count, success_rate, MARKED, SEED, saved):
    try:
        marked, seed = int(MARKED), int(SEED)
        if float(MARKED) != marked or float(SEED) != seed:
            raise ValueError
    except (TypeError, ValueError):
        return False, ["MARKED and SEED must be the whole numbers shown in your Canvas lab check."], None
    if not (0 <= marked <= 7) or not (100 <= seed <= 999):
        return False, ["MARKED must be 0 to 7 and SEED 100 to 999. Copy them again from the lab check."], None
    msgs, ok = [], True
    for test, fn in ((_test_grover, grover_circuit), (_test_two_qubit, two_qubit_count), (_test_success, success_rate)):
        good, m = test(fn)
        msgs += m
        ok = ok and good
    if not ok:
        return False, msgs, None
    value = personal_value(saved, marked, seed)
    if value is None:
        return False, msgs + ["These two numbers are not in the saved runs. Copy MARKED and SEED again from the lab check."], None
    return True, msgs, value
