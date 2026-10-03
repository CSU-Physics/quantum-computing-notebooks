"""Check cell logic for the Module 6 lab (Bernstein-Vazirani), on qsim.

check_module6(bv_oracle, bv_circuit, classical_bv, KEY, DEP_PCT)
returns (passed, messages, verification_value).
Hidden strings are written like Qiskit results: the rightmost character is the bit of qubit 0.
The learner writes the three functions. The verification value is the number of shots, out of 1,000,
in which the Bernstein-Vazirani circuit for the learner's personal 6-bit hidden string (made from KEY)
returns that string, on qsim's AerSimulator with seed KEY and a depolarizing error of DEP_PCT % on
every CNOT. It is printed only when every test passes, and it depends on the seeded draw, so it cannot
be worked out by hand.
"""
import numpy as np
from qsim import AerSimulator, NoiseModel, Operator, QuantumCircuit, depolarizing_error

N_PERSONAL = 6


def reference_oracle(s):
    """|x>|y> -> |x>|y XOR s.x>: a CNOT from input qubit i to the answer qubit n for every 1 in s."""
    n = len(s)
    qc = QuantumCircuit(n + 1)
    for i in range(n):
        if s[n - 1 - i] == "1":
            qc.cx(i, n)
    return qc


def reference_circuit(oracle, n):
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)
    qc.h(range(n + 1))
    qc = qc.compose(oracle)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


def personal_string(key):
    return format((int(key) * 47) % 63 + 1, f"0{N_PERSONAL}b")


def personal_value(key, dep_pct):
    s = personal_string(key)
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(float(dep_pct) / 100, 2), ["cx"])
    qc = reference_circuit(reference_oracle(s), N_PERSONAL)
    counts = AerSimulator(seed_simulator=int(key), noise_model=nm).run(qc, shots=1000).result().get_counts()
    return int(counts.get(s, 0))


def _marked_oracle(s, variant):
    """A hidden oracle for testing bv_circuit(): it starts with an 'id' marker on the answer qubit."""
    n = len(s)
    qc = QuantumCircuit(n + 1)
    qc.id(n)
    ones = [i for i in range(n) if s[n - 1 - i] == "1"]
    if variant == "cz" and ones:                 # the same oracle written with CZ gates
        qc.h(n)
        for i in ones:
            qc.cz(i, n)
        qc.h(n)
    else:
        for i in (reversed(ones) if variant == "rev" else ones):
            qc.cx(i, n)
    return qc


def check_module6(bv_oracle, bv_circuit, classical_bv, KEY, DEP_PCT):
    msgs = []

    # Test 1: bv_oracle(s) is the oracle |x>|y> -> |x>|y XOR s.x>, with the answer qubit n.
    for s in ["1", "0", "10", "01", "101", "1101", "0110", "11111", "10010", "000"]:
        n = len(s)
        try:
            qc = bv_oracle(s)
        except Exception as exc:  # noqa: BLE001
            return False, [f"bv_oracle('{s}') stopped with an error: {exc}"], None
        if not isinstance(qc, QuantumCircuit) or qc.num_qubits != n + 1:
            return False, [f"bv_oracle('{s}') must return a QuantumCircuit with {n + 1} qubits: "
                           f"{n} input qubits (0 to {n - 1}) and the answer qubit {n}."], None
        if any(i.name in ("measure", "reset") for i in qc.data):
            return False, [f"bv_oracle('{s}') must not measure or reset: it is only the oracle."], None
        if not Operator(qc).equiv(Operator(reference_oracle(s))):
            hint = ("For every '1' in s, add a CNOT from that input qubit to the answer qubit n. The rightmost "
                    "character of s belongs to qubit 0, so character s[n - 1 - i] belongs to qubit i.")
            if Operator(qc).equiv(Operator(reference_oracle(s[::-1]))):
                hint = "The bits are in the wrong order. " + hint
            return False, [f"bv_oracle('{s}') is not the right oracle. {hint}"], None
    msgs.append("Test 1, bv_oracle(): passed for 10 hidden strings")

    # Test 2: bv_circuit(oracle, n) finds a hidden string it is not told, with one use of the oracle.
    rng = np.random.default_rng(606)
    cases = [("1011", "cx"), ("0000", "cx"), ("111", "cz"), ("100101", "rev")]
    for n in (2, 3, 5, 6):
        cases.append(("".join(str(b) for b in rng.integers(0, 2, size=n)), "cx"))
    for s, variant in cases:
        n = len(s)
        oracle = _marked_oracle(s, variant)
        try:
            qc = bv_circuit(oracle.copy(), n)
        except Exception as exc:  # noqa: BLE001
            return False, [f"bv_circuit(oracle, {n}) stopped with an error: {exc}"], None
        if not isinstance(qc, QuantumCircuit) or qc.num_qubits != n + 1:
            return False, [f"bv_circuit(oracle, {n}) must return a QuantumCircuit with {n + 1} qubits."], None
        uses = sum(1 for i in qc.data if i.name == "id")
        if uses != 1:
            return False, [f"bv_circuit() used the oracle {uses} times. Bernstein-Vazirani needs it exactly once: "
                           "add it with qc.compose(oracle), and do not build your own oracle inside bv_circuit()."], None
        if qc.num_clbits != n or sum(1 for i in qc.data if i.name == "measure") != n:
            return False, [f"bv_circuit(oracle, {n}) must have {n} classical bits and measure the {n} input qubits "
                           f"(0 to {n - 1}) into them, and not the answer qubit."], None
        try:
            counts = AerSimulator(seed_simulator=1).run(qc, shots=200).result().get_counts()
        except Exception as exc:  # noqa: BLE001
            return False, [f"Running bv_circuit(oracle, {n}) stopped with an error: {exc}"], None
        if counts != {s: 200}:
            hint = ("Prepare the answer qubit in |-> (X, then H), put H on every input qubit, add the oracle, put H "
                    "on the input qubits again, then measure input qubit i into classical bit i.")
            if counts == {s[::-1]: 200}:
                hint = "The result is reversed: measure input qubit i into classical bit i. " + hint
            elif len(counts) > 1:
                hint = "The result is random, so the interference is missing. " + hint
            return False, [f"For a hidden oracle, bv_circuit() gave {dict(sorted(counts.items())[:4])}, but every shot "
                           f"should give the hidden string. {hint}"], None
    msgs.append("Test 2, bv_circuit(): passed, finding 8 hidden strings with one use of the oracle each")

    # Test 3: classical_bv(query, n) finds s with n queries.
    for s in ["1", "01", "110", "1011", "100110", "01111110"]:
        n = len(s)
        calls = []

        def query(x, s=s, n=n, calls=calls):
            x = str(x)
            if len(x) != n or set(x) - set("01"):
                raise ValueError(f"query() needs a string of {n} characters, each 0 or 1, such as {'0' * (n - 1) + '1'!r}")
            calls.append(x)
            return sum(int(a) * int(b) for a, b in zip(s, x)) % 2

        try:
            got = classical_bv(query, n)
        except Exception as exc:  # noqa: BLE001
            return False, [f"classical_bv(query, {n}) stopped with an error: {exc}"], None
        if str(got) != s:
            hint = ("Ask about one bit at a time: the input with a single 1 in position j returns s[j]. Join the "
                    "answers into a string in the same order.")
            if str(got) == s[::-1]:
                hint = "The answer is reversed. " + hint
            return False, [f"classical_bv(query, {n}) returned {got!r}, but the hidden string is {s!r}. {hint}"], None
        if len(calls) > n:
            return False, [f"classical_bv(query, {n}) used {len(calls)} queries. Use exactly {n}: one per bit."], None
    msgs.append("Test 3, classical_bv(): passed, each with n queries")

    if KEY is None or DEP_PCT is None:
        return False, msgs + ["Enter KEY and DEP_PCT from Canvas in Step 8, then run this cell again."], None
    value = personal_value(KEY, DEP_PCT)
    msgs.append(f"Your personal oracle has a 6-bit hidden string. The Bernstein-Vazirani circuit for it was run "
                f"1,000 times with seed {KEY} and a {DEP_PCT}% depolarizing error on each CNOT.")
    return True, msgs, value
