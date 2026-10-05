"""Tests for checks/l2_m4_check.py, with real Qiskit (as in Colab) and with qsim (as in the browser).
Run: PYTHONPATH=content/level1:checks python checks/test_l2_m4_check.py          (Qiskit if installed)
     PYTHONPATH=content/level1:checks python checks/test_l2_m4_check.py qsim     (force qsim)"""
import json
import sys
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] == "qsim":
    sys.modules["qiskit"] = None                    # make "import qiskit" fail, as in the browser
import l2_m4_check as chk
from l2_m4_check import check_l2_module4, QuantumCircuit

SAVED = json.loads((Path(__file__).resolve().parents[1] / "content/level2/m4_saved_runs.json").read_text())
print("checker uses", QuantumCircuit.__module__)


def mcz(qc, qs):
    qs = list(qs)
    if len(qs) == 2:
        qc.cz(qs[0], qs[1])
    else:
        qc.h(qs[-1]); qc.mcx(qs[:-1], qs[-1]); qc.h(qs[-1])


def make_grover(start_h=True, extra=0, reverse=False, measure_all=False, no_diffuser=False):
    def g(marked, iterations):
        n = len(marked)
        w = marked[::-1] if reverse else marked
        qc = QuantumCircuit(n) if measure_all else QuantumCircuit(n, n)
        if start_h:
            for q in range(n):
                qc.h(q)
        for _ in range(iterations + extra):
            zeros = [q for q in range(n) if w[n - 1 - q] == "0"]
            for q in zeros: qc.x(q)
            mcz(qc, range(n))
            for q in zeros: qc.x(q)
            if not no_diffuser:
                for q in range(n): qc.h(q); qc.x(q)
                mcz(qc, range(n))
                for q in range(n): qc.x(q); qc.h(q)
        if measure_all:
            qc.measure_all()
        else:
            for q in range(n):
                qc.measure(q, q)
        return qc
    return g


def gr_todo(marked, iterations):
    raise NotImplementedError


def tq_ok(ops):
    return sum(1 for name, qs in ops if len(qs) == 2 and name != "barrier")


def tq_ok_set(ops):
    return len([1 for name, qs in ops if name not in ("barrier", "measure") and len(set(qs)) == 2])


def tq_counts_barrier(ops):
    return sum(1 for name, qs in ops if len(qs) == 2)


def tq_only_cz(ops):
    return sum(1 for name, qs in ops if name == "cz")


def tq_two_or_more(ops):
    return sum(1 for name, qs in ops if len(qs) >= 2 and name != "barrier")


def tq_todo(ops):
    raise NotImplementedError


def sr_ok(counts, marked):
    return counts.get(marked, 0) / sum(counts.values())


def sr_keyerror(counts, marked):
    return counts[marked] / sum(counts.values())


def sr_fixed_total(counts, marked):
    return counts.get(marked, 0) / 4000


def sr_percent(counts, marked):
    return 100 * counts.get(marked, 0) / sum(counts.values())


def sr_todo(counts, marked):
    raise NotImplementedError


good_g, good_t, good_s = [make_grover(), make_grover(measure_all=True)], [tq_ok, tq_ok_set], [sr_ok]
bad_g = [make_grover(start_h=False), make_grover(extra=1), make_grover(reverse=True), make_grover(no_diffuser=True), gr_todo]
bad_t = [tq_counts_barrier, tq_only_cz, tq_two_or_more, tq_todo]
bad_s = [sr_keyerror, sr_fixed_total, sr_percent, sr_todo]
p = SAVED["personal"]
PARAMS = [(p[0]["marked"], p[0]["seed"]), (p[1]["marked"], p[1]["seed"]), (p[2]["marked"], p[2]["seed"])]

wrong, values = 0, []
for mk, seed in PARAMS:
    for g in good_g:
        for t in good_t:
            for s in good_s:
                ok, msgs, v = check_l2_module4(g, t, s, mk, seed, SAVED)
                if not ok:
                    wrong += 1; print("WRONG (should pass):", msgs)
    values.append(check_l2_module4(good_g[0], tq_ok, sr_ok, mk, seed, SAVED)[2])
    for i, b in enumerate(bad_g):
        ok, msgs, _ = check_l2_module4(b, tq_ok, sr_ok, mk, seed, SAVED)
        if ok: wrong += 1; print("WRONG (should fail): grover", i)
        elif (mk, seed) == PARAMS[0]: print("   g", i, "->", [m for m in msgs if "passed" not in m][0][:140])
    for b in bad_t:
        ok, msgs, _ = check_l2_module4(good_g[0], b, sr_ok, mk, seed, SAVED)
        if ok: wrong += 1; print("WRONG (should fail):", b.__name__)
        elif (mk, seed) == PARAMS[0]: print("  ", b.__name__, "->", [m for m in msgs if "passed" not in m][0][:140])
    for b in bad_s:
        ok, msgs, _ = check_l2_module4(good_g[0], tq_ok, b, mk, seed, SAVED)
        if ok: wrong += 1; print("WRONG (should fail):", b.__name__)
        elif (mk, seed) == PARAMS[0]: print("  ", b.__name__, "->", [m for m in msgs if "passed" not in m][0][:140])
for bad in [(8, 512), (3, 99), (2.5, 512), (3, 1000)]:
    if check_l2_module4(good_g[0], tq_ok, sr_ok, *bad, SAVED)[0]:
        wrong += 1; print("WRONG: out-of-range parameters accepted", bad)
# the reference instruction lists in the saved file
for lvl in ("0", "3"):
    ops = SAVED["teach"]["instructions"][lvl]
    assert tq_ok(ops) == SAVED["teach"]["runs"]["2"][lvl]["two_qubit"], lvl
n_good = len(good_g) * len(good_t) * len(good_s)
print(f"{n_good} correct combinations and {len(bad_g) + len(bad_t) + len(bad_s)} wrong versions at {len(PARAMS)} "
      f"parameter sets: {wrong} wrong verdicts")
print("values:", values, "expected:", [chk.personal_value(SAVED, m, s) for m, s in PARAMS])
sys.exit(1 if wrong else 0)
