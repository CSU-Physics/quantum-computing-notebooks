"""Tests for checks/l3_m2_check.py: correct and wrong versions of the three functions, the closed form of the
verification value against qsim's exact simulation, and qsim against Qiskit Aer (sampling).
Run: PYTHONPATH=content/level3:checks python checks/test_l3_m2_check.py"""
import numpy as np
import l3_m2_check as chk
from l3_m2_check import (check_l3_module2, reference_encode as enc, reference_measure_syndrome as syn,
                         reference_correct as cor)


# ---------------------------------------------------------------- correct versions
def enc2(qc):
    qc.cx(0, 2)
    qc.cx(0, 1)


def syn2(qc):                         # other gate order, reset before nothing else
    qc.cx(1, 4); qc.cx(2, 4); qc.cx(0, 3); qc.cx(1, 3)
    qc.measure([3, 4], [0, 1])
    qc.reset([3, 4])


def cor_else(qc):                     # with else blocks
    with qc.if_test((0, 1)) as else_:
        with qc.if_test((1, 1)) as e2:
            qc.x(1)
        with e2:
            qc.x(0)
    with else_:
        with qc.if_test((1, 1)):
            qc.x(2)


def cor_b(qc):                        # outer test on bit 1
    with qc.if_test((1, 1)):
        with qc.if_test((0, 1)):
            qc.x(1)
        with qc.if_test((0, 0)):
            qc.x(2)
    with qc.if_test((1, 0)):
        with qc.if_test((0, 1)):
            qc.x(0)


# ---------------------------------------------------------------- wrong versions
def enc_one(qc): qc.cx(0, 1)
def enc_rev(qc): qc.cx(1, 0); qc.cx(2, 0)
def enc_h(qc): qc.cx(0, 1); qc.cx(0, 2); qc.h(0); qc.h(1); qc.h(2)
def enc_anc(qc): qc.cx(0, 1); qc.cx(0, 2); qc.cx(0, 3)
def syn_swap(qc): qc.cx(0, 3); qc.cx(1, 3); qc.cx(1, 4); qc.cx(2, 4); qc.measure(3, 1); qc.measure(4, 0); qc.reset([3, 4])
def syn_noreset(qc): qc.cx(0, 3); qc.cx(1, 3); qc.cx(1, 4); qc.cx(2, 4); qc.measure(3, 0); qc.measure(4, 1)
def syn_data(qc): qc.measure(0, 0); qc.measure(1, 1)
def syn_rev(qc): qc.cx(3, 0); qc.cx(3, 1); qc.cx(4, 1); qc.cx(4, 2); qc.measure(3, 0); qc.measure(4, 1); qc.reset([3, 4])
def syn_one(qc): qc.cx(0, 3); qc.cx(1, 4); qc.measure(3, 0); qc.measure(4, 1); qc.reset([3, 4])


def cor_none(qc): qc.barrier()
def cor_always(qc): qc.x(0)


def cor_wrong(qc):                    # (1,0) -> qubit 1, (1,1) -> qubit 0
    with qc.if_test((0, 1)):
        with qc.if_test((1, 1)):
            qc.x(0)
        with qc.if_test((1, 0)):
            qc.x(1)
    with qc.if_test((0, 0)):
        with qc.if_test((1, 1)):
            qc.x(2)


def cor_flat(qc):                     # one bit at a time: wrong for (1, 1)
    with qc.if_test((0, 1)):
        qc.x(0)
    with qc.if_test((1, 1)):
        qc.x(2)


def notyet(qc): raise NotImplementedError


ARGS = (1.25, 0.10, 0.05)
good = [(enc, syn, cor), (enc2, syn2, cor_else), (enc, syn2, cor_b), (enc2, syn, cor_b)]
bad = [(enc_one, syn, cor), (enc_rev, syn, cor), (enc_h, syn, cor), (enc_anc, syn, cor), (enc, syn_swap, cor),
       (enc, syn_noreset, cor), (enc, syn_data, cor), (enc, syn_rev, cor), (enc, syn_one, cor), (enc, syn, cor_none),
       (enc, syn, cor_always), (enc, syn, cor_wrong), (enc, syn, cor_flat), (notyet, syn, cor), (enc, notyet, cor),
       (enc, syn, notyet)]
wrong = 0
for fns in good:
    ok, msgs, v = check_l3_module2(*fns, *ARGS)
    if not ok:
        wrong += 1
        print("GOOD FAILED", [f.__name__ for f in fns], msgs[-1])
for fns in bad:
    ok, msgs, v = check_l3_module2(*fns, *ARGS)
    if ok:
        wrong += 1
        print("BAD PASSED", [f.__name__ for f in fns])
    else:
        print("  bad", [f.__name__ for f in fns], "->", msgs[-1][:150])
ok, msgs, v = check_l3_module2(enc, syn, cor, *ARGS)
print("value for", ARGS, "=", v)
for a in [(0.2, 0.1, 0.05), (1.25, 0.35, 0.05), (1.25, 0.1, 0.2), (1.255, 0.1, 0.05)]:
    assert not check_l3_module2(enc, syn, cor, *a)[0], a

# closed form against qsim's exact simulation of the noisy circuit
rng = np.random.default_rng(2)
worst = 0
for _ in range(40):
    th, p, q = rng.uniform(0.3, 2.8), rng.uniform(0.01, 0.3), rng.uniform(0, 0.1)
    qc = chk.one_round(th)
    f = chk.data_fidelity(qc, th, chk.noise_model(p, q))
    worst = max(worst, abs(f - chk.personal_fidelity(th, p, q)))
print("closed form vs qsim exact simulation (40 cases): worst difference", worst)
assert worst < 1e-10

# qsim and Qiskit Aer: logical failure frequency of the full round, sampled
try:
    from qiskit import QuantumCircuit as QC
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, pauli_error, ReadoutError

    def aer_round(th):
        qc = QC(5, 5)
        qc.ry(th, 0); qc.cx(0, 1); qc.cx(0, 2)
        for d in (0, 1, 2):
            qc.id(d)
        syn(qc)
        cor(qc)
        qc.cx(0, 2); qc.cx(0, 1); qc.ry(-th, 0)        # undo the encoding: success if qubits 0, 1, 2 read 000
        qc.measure([0, 1, 2], [2, 3, 4])
        return qc
    p, q, th, shots = 0.12, 0.06, 1.1, 200000
    nm = NoiseModel()
    for d in (0, 1, 2):
        nm.add_quantum_error(pauli_error([("X", p), ("I", 1 - p)]), ["id"], [d])
    for a in (3, 4):
        nm.add_readout_error(ReadoutError([[1 - q, q], [q, 1 - q]]), [a])
    c = AerSimulator(noise_model=nm, seed_simulator=5).run(aer_round(th), shots=shots).result().get_counts()
    ok_aer = sum(v for k, v in c.items() if k.split()[0][:3] == "000") / shots if " " in next(iter(c)) else \
        sum(v for k, v in c.items() if k[:3] == "000") / shots
    want = chk.personal_fidelity(th, p, q)
    print(f"Aer: P(back to 000) = {ok_aer:.4f}; closed-form fidelity {want:.4f}; difference {abs(ok_aer - want):.4f} "
          f"(3 sigma = {3 * np.sqrt(want * (1 - want) / shots):.4f})")
    assert abs(ok_aer - want) < 4 * np.sqrt(want * (1 - want) / shots)
except ImportError:
    print("Qiskit Aer not installed; skipped")

# qsim sampling of the same circuit
import qsim
qc = qsim.QuantumCircuit(5, 5)
qc.ry(1.1, 0); enc(qc)
for d in (0, 1, 2):
    qc.id(d)
syn(qc); cor(qc)
qc.cx(0, 2); qc.cx(0, 1); qc.ry(-1.1, 0)
qc.measure([0, 1, 2], [2, 3, 4])
c = qsim.AerSimulator(noise_model=chk.noise_model(0.12, 0.06), seed_simulator=3).run(qc, shots=20000).result().get_counts()
f = sum(v for k, v in c.items() if k[:3] == "000") / 20000
print(f"qsim sampling: P(back to 000) = {f:.4f} (exact {chk.personal_fidelity(1.1, 0.12, 0.06):.4f})")
print("wrong verdicts:", wrong)
assert wrong == 0
