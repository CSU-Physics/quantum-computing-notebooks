"""Tests for checks/l2_m0_check.py: correct versions must pass, wrong ones must fail.
Run: PYTHONPATH=content/level1:checks python checks/test_l2_m0_check.py"""
import sys
from l2_m0_check import check_l2_module0, personal_value

# ---- active_reset versions
def ar_ok(qc, q, c):
    qc.measure(q, c)
    with qc.if_test((c, 1)):
        qc.x(q)

def ar_ok_else(qc, q, c):
    qc.measure(q, c)
    with qc.if_test((c, 0)) as else_:
        qc.id(q)
    with else_:
        qc.x(q)

def ar_ok_return(qc, q, c):
    qc.measure(q, c)
    with qc.if_test((c, 1)):
        qc.x(q)
    return qc

def ar_reset(qc, q, c): qc.reset(q)
def ar_uncond(qc, q, c): qc.measure(q, c); qc.x(q)
def ar_value0(qc, q, c):
    qc.measure(q, c)
    with qc.if_test((c, 0)):
        qc.x(q)
def ar_nomeasure(qc, q, c):
    with qc.if_test((c, 1)):
        qc.x(q)
def ar_noif(qc, q, c): qc.measure(q, c)
def ar_todo(qc, q, c): raise NotImplementedError

# ---- bob_corrections versions
def bc_ok(qc):
    with qc.if_test((1, 1)):
        qc.x(2)
    with qc.if_test((0, 1)):
        qc.z(2)

def bc_ok_zfirst(qc):
    with qc.if_test((0, 1)):
        qc.z(2)
    with qc.if_test((1, 1)):
        qc.x(2)

def bc_ok_nested(qc):
    with qc.if_test((1, 1)) as else_:
        with qc.if_test((0, 1)) as e2:
            qc.y(2)
        with e2:
            qc.x(2)
    with else_:
        with qc.if_test((0, 1)):
            qc.z(2)

def bc_swapped(qc):
    with qc.if_test((0, 1)):
        qc.x(2)
    with qc.if_test((1, 1)):
        qc.z(2)
def bc_noz(qc):
    with qc.if_test((1, 1)):
        qc.x(2)
def bc_nox(qc):
    with qc.if_test((0, 1)):
        qc.z(2)
def bc_uncond(qc):
    qc.x(2); qc.z(2)
def bc_wrongqubit(qc):
    with qc.if_test((1, 1)):
        qc.x(1)
    with qc.if_test((0, 1)):
        qc.z(1)
def bc_value0(qc):
    with qc.if_test((1, 0)):
        qc.x(2)
    with qc.if_test((0, 0)):
        qc.z(2)
def bc_measure(qc):
    qc.measure(2, 1)
    with qc.if_test((1, 1)):
        qc.x(2)
def bc_todo(qc): raise NotImplementedError

PARAMS = [(1.25, 512), (0.31, 101), (2.87, 999)]
good_ar = [ar_ok, ar_ok_else, ar_ok_return]
bad_ar = [ar_reset, ar_uncond, ar_value0, ar_nomeasure, ar_noif, ar_todo]
good_bc = [bc_ok, bc_ok_zfirst, bc_ok_nested]
bad_bc = [bc_swapped, bc_noz, bc_nox, bc_uncond, bc_wrongqubit, bc_value0, bc_measure, bc_todo]
wrong = 0
for th, sd in PARAMS:
    ref = personal_value(th, sd)
    for a in good_ar:
        for b in good_bc:
            ok, msgs, val = check_l2_module0(a, b, th, sd)
            if not ok or val != ref:
                wrong += 1; print("WRONG (should pass):", a.__name__, b.__name__, msgs)
    for a in bad_ar:
        ok, msgs, val = check_l2_module0(a, bc_ok, th, sd)
        if ok:
            wrong += 1; print("WRONG (should fail):", a.__name__)
    for b in bad_bc:
        ok, msgs, val = check_l2_module0(ar_ok, b, th, sd)
        if ok:
            wrong += 1; print("WRONG (should fail):", b.__name__)
ok, msgs, val = check_l2_module0(ar_ok, bc_ok, 5.0, 512)
if ok:
    wrong += 1; print("WRONG: out-of-range THETA accepted")
print(f"{len(good_ar) * len(good_bc)} correct combinations and {len(bad_ar) + len(bad_bc)} wrong versions "
      f"at {len(PARAMS)} parameter sets: {wrong} wrong verdicts")
print("values:", [personal_value(t, s) for t, s in PARAMS])
sys.exit(1 if wrong else 0)
