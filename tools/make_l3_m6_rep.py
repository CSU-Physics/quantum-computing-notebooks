"""Build the Level 3 Module 6 capstone notebook "Repetition code: syndrome statistics and the logical error".
Numbers quoted in the text are computed here from checks/l3_m6_rep_check.py (exact, course noise model)."""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "content" / "level3"), str(ROOT / "checks"), str(ROOT / "tools")]
import l3_m6_rep_check as R  # noqa: E402
from l3_hidden import hidden_check  # noqa: E402
from l3_m6_text import META, check_cell, code, ending, intro, md, personal_step, step0, step1_statistics, step2_noise  # noqa: E402

NAME = "QC-L3-M6-capstone-repetition-code.ipynb"
P2, RO = 0.01, 0.02

# ---------------------------------------------------------------- numbers for the text
enc = {r: R.logical_error(R.record_distribution(R.reference_memory_circuit(r), P2, RO, P2), r) for r in range(1, 6)}
bare = {r: R.record_distribution(R.bare_circuit(r), P2, RO, P2).get("1", 0.0) for r in range(1, 6)}
d3 = R.record_distribution(R.reference_memory_circuit(3), P2, RO, P2)
rates = [0.0] * 4
kept = kept_err = 0.0
for k, p in d3.items():
    ev = R.reference_detection_events(k, 3)
    for i, e in enumerate(ev):
        if e != (0, 0):
            rates[i] += p
    if all(e == (0, 0) for e in ev):
        kept += p
        kept_err += p * (R.reference_decode(k, 3) != 0)
scan = [round(0.005 * i, 3) for i in range(1, 17)]
cross = None
for p in [round(0.001 * i, 3) for i in range(5, 81)]:
    e = R.logical_error(R.record_distribution(R.reference_memory_circuit(3), p, RO, p), 3)
    b = R.record_distribution(R.bare_circuit(3), p, RO, p).get("1", 0.0)
    if e > b:
        cross = p
        break
assert cross is not None
NUM = dict(e3=enc[3], b3=bare[3], r=rates, kept=kept, kerr=kept_err / kept, cross=cross)
print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in NUM.items()})

cells = [
intro("Repetition code: syndrome statistics and the logical error",
      "Does the three-qubit repetition code protect a stored bit better than a single qubit that waits as long, how does its "
      "logical error grow with the number of syndrome rounds, and at which noise level does it stop helping?",
      """1. write the statistics functions every capstone uses (Step 1) and meet the capstone noise model (Step 2);
2. build the **memory experiment**: encode, several rounds of syndrome measurement, a final measurement of the data (Step 3);
3. read every shot: write `detection_events()` and `decode()` and measure the **logical error** (Step 4);
4. compare the code with an unencoded qubit for 1 to 5 rounds (Step 5);
5. look at the **syndrome statistics**: how often each round signals an error (Step 6);
6. use the code to **detect** instead of correct: discard shots that signalled an error (Step 7);
7. find the **break-even** noise level, where the code stops beating a single qubit (Step 8);
8. get your verification value (Step 9).""",
      "Question 1 gives you three personal numbers: ROUNDS, P2 and READOUT."),
*step0(),
*step1_statistics(),
*step2_noise("""

This capstone measures in the middle of the circuit, so it adds two more prepared tools in Step 3: `record_distribution(qc, p2, readout, idle)`, the exact probability of every recorded key of a circuit with mid-circuit measurements (each measurement splits the density matrix into branches, as in Module 2), and `logical_error()`."""),

md("""## Step 3: the memory experiment

A **memory experiment** stores one bit and checks how well it survives. It is how IBM, Google and others test their codes. Your circuit, `memory_circuit(rounds, bit)`, uses the layout of Module 2: data qubits 0, 1 and 2, ancillas 3 and 4, and 2 · rounds + 3 classical bits.

1. **Prepare and encode.** If `bit` is 1, apply X to qubit 0. Then CX(0, 1) and CX(0, 2): the logical 0 is |000⟩, the logical 1 is |111⟩.
2. **Each round r = 0, 1, ...:**
   - an `id` gate on each data qubit. The data qubits wait while the ancillas are measured, and the noise model puts the **idle error** (here equal to P2) on these id gates;
   - CX(0, 3), CX(1, 3), CX(1, 4), CX(2, 4), in this order: ancilla 3 collects the parity of qubits 0 and 1, ancilla 4 that of qubits 1 and 2 (without noise the order does not matter; with noise it changes the result slightly, and the check uses this order);
   - measure ancilla 3 into classical bit 2r and ancilla 4 into classical bit 2r + 1;
   - reset both ancillas, so the next round starts again from |0⟩.
3. **Finally** measure data qubit d into classical bit 2 · rounds + d.

There is no correction during the circuit: the decoder reads everything afterwards. (Correcting with `if_test` in every round, as in Module 2, gives the same logical error for this code, because a correction only flips a data qubit the final majority vote would flip anyway.)

**Your task:** write `memory_circuit(rounds, bit=0)`. The prepared cell after it defines `bare_circuit(rounds, bit)`, a single qubit that waits as long (one id gate per round), and the exact tools."""),
code("""def memory_circuit(rounds, bit=0):
    \"\"\"The memory experiment: encode `bit`, `rounds` rounds of syndrome measurement, then measure the data.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete memory_circuit() first.")"""),
code("""def bare_circuit(rounds, bit=0):
    \"\"\"Prepared: an unencoded qubit that waits as long (one id gate per round), then a measurement.\"\"\"
    qc = QuantumCircuit(1, 1)
    if bit:
        qc.x(0)
    for _ in range(rounds):
        qc.id(0)
    qc.measure(0, 0)
    return qc


def record_distribution(qc, p2=P2_COURSE, readout=READOUT_COURSE, idle=None):
    \"\"\"Prepared: the exact probability of every recorded key of a circuit with mid-circuit measurements.
    idle defaults to p2. Each measurement in the middle splits the state into branches (the course simulator's
    branch simulation, as for if_test in Module 2); the final measurements are read from each branch.\"\"\"
    idle = p2 if idle is None else idle
    final = qc._final_measure_indices()
    meas = [qc.data[k] for k in sorted(final)]
    body = qc.copy_empty_like()
    body.data = [inst for k, inst in enumerate(qc.data) if k not in final]
    nm = capstone_noise_model(p2, readout, idle)
    branches = qsim._branch_states(body, nm if not nm.is_ideal() else None)
    qubits, clbits = [m.qubits[0] for m in meas], [m.clbits[0] for m in meas]
    out = {}
    for r, bits in branches:
        w = float(np.real(np.trace(r)))
        if w < 1e-15:
            continue
        probs = np.asarray(qsim.DensityMatrix(r / w).probabilities(qargs=qubits), dtype=float) * w
        probs = apply_readout(probs, readout)
        for o, p in enumerate(probs):
            if p < 1e-16:
                continue
            b = list(bits)
            for j, c in enumerate(clbits):
                b[c] = (o >> j) & 1
            key = "".join(str(x) for x in reversed(b))
            out[key] = out.get(key, 0.0) + float(p)
    return out


qc = memory_circuit(2)
print(qc.draw())
ops = [i.name for i in qc.data]
print("2 rounds:", ops.count("cx"), "CX,", ops.count("id"), "id,", ops.count("measure"), "measurements,", ops.count("reset"), "resets")
ideal = record_distribution(memory_circuit(3, 0), p2=0, readout=0)
print("Without noise, logical 0 after 3 rounds gives:", ideal, "  (expected {'000000000': 1.0})")"""),

md("""## Step 4: read every shot

Each shot gives a key of 2 · rounds + 3 bits, in Qiskit's order: classical bit 0 is the **rightmost** character. For 2 rounds, the key `"010" + "00" + "01"` means: round 0 syndrome (s₁, s₂) = (1, 0) (bits 0 and 1), round 1 syndrome (0, 0) (bits 2 and 3), final data bits d₀ d₁ d₂ = 0, 1, 0 (bits 4, 5, 6).

**Detection events.** A syndrome of (1, 0) can mean a data error that happened in this round, or one from an earlier round that is still there. What signals a *new* error is a **change**: the syndrome of round r XOR the syndrome of round r − 1 (before the first round, the syndrome counts as (0, 0)). A change is a **detection event**, the quantity real decoders work with (you met it in Module 2's saved device run). After the last round, the final data bits give one more syndrome, (d₀ XOR d₁, d₁ XOR d₂), compared with the last measured one.

**Decoding.** The simplest decoder is the majority of the three final data bits: it corrects any single flipped data qubit.

**Your task:**

- `detection_events(key, rounds)`: a list of rounds + 1 pairs (e₁, e₂), the changes of the syndrome, the last one from the final data bits.
- `decode(key, rounds)`: 0 or 1, the majority of the final data bits.

The prepared `logical_error(probs, rounds, bit)` then adds up the probability (or the fraction of shots) where `decode` disagrees with the stored bit."""),
code("""def detection_events(key, rounds):
    \"\"\"rounds + 1 pairs: each round's syndrome XOR the previous one, then the syndrome of the final data bits.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete detection_events() first.")


def decode(key, rounds):
    \"\"\"The logical bit: the majority of the three final data bits.\"\"\"
    # YOUR CODE HERE
    raise NotImplementedError("Complete decode() first.")"""),
code("""def logical_error(probs, rounds, bit=0):
    \"\"\"Prepared: the probability (or fraction of shots) in which decode() disagrees with the stored bit.\"\"\"
    total = sum(probs.values())
    return sum(p for k, p in probs.items() if decode(k, rounds) != bit) / total


print(detection_events("010" + "00" + "01", 2), "  (expected [(1, 0), (1, 0), (1, 1)])")
print(decode("010" + "00" + "01", 2), "  (expected 0)")

SHOTS, ROUNDS_DEMO = 4000, 3
exact3 = record_distribution(memory_circuit(ROUNDS_DEMO))
counts = sample_counts(exact3, SHOTS, seed=1)
p_exact = logical_error(exact3, ROUNDS_DEMO)
f = logical_error(counts, ROUNDS_DEMO)
s = fraction_sigma(p_exact, SHOTS)
print(f"Logical 0, {ROUNDS_DEMO} rounds, P2 = {P2_COURSE}, READOUT = {READOUT_COURSE}:")
print(f"   exact logical error {p_exact:.4f};  your run of {SHOTS} shots: {f:.4f} +- {s:.4f} ({(f - p_exact) / s:+.1f} sigma)")"""),
md(f"""**What to notice.** The exact logical error after 3 rounds is **{NUM['e3']:.4f}**. A run of 4,000 shots lands within a few σ of it (σ ≈ {(NUM['e3'] * (1 - NUM['e3']) / 4000) ** 0.5:.4f}); the 3σ rule says whether a measured value agrees with the noise model. A small error rate needs many shots to measure precisely: with p ≈ 0.015, σ / p is about 13% at 4,000 shots."""),

md("""## Step 5: does the code beat a single qubit?

An unencoded qubit stores a bit too. Over the same number of rounds it collects idle errors, and its readout error is not corrected. The code costs four noisy CNOTs per round and more qubits that can fail, but it corrects any single flipped data qubit.

Run the cell: it compares the exact logical error of the code with the exact error of a bare qubit (`bare_circuit`) for 1 to 5 rounds, and uses `combined_sigma` to say whether 4,000 shots of each could tell them apart."""),
code("""print(f"{'rounds':>7s}{'code':>10s}{'bare qubit':>12s}{'difference':>12s}{'sigma_diff':>12s}{'significant?':>14s}")
enc, bare = [], []
for r in range(1, 6):
    e = logical_error(record_distribution(memory_circuit(r)), r)
    b = record_distribution(bare_circuit(r)).get("1", 0.0)
    sd = combined_sigma([1, -1], [fraction_sigma(e, SHOTS), fraction_sigma(b, SHOTS)])
    enc.append(e); bare.append(b)
    print(f"{r:7d}{e:10.4f}{b:12.4f}{b - e:12.4f}{sd:12.4f}{('yes' if abs(b - e) > 3 * sd else 'no'):>14s}")

plt.figure(figsize=(6, 3.6))
plt.plot(range(1, 6), enc, "o-", label="repetition code")
plt.plot(range(1, 6), bare, "s--", label="bare qubit")
plt.xlabel("rounds"); plt.ylabel("logical error"); plt.legend(); plt.title("Stored 0, P2 = 0.01, READOUT = 0.02")
plt.tight_layout(); plt.show()"""),
md(f"""**What to notice.** The code's logical error ({enc[1]:.4f} after one round, {NUM['e3']:.4f} after three) is less than half the bare qubit's ({bare[1]:.4f} and {NUM['b3']:.4f}), and with 4,000 shots of each the difference is significant at every number of rounds. Both errors grow with the rounds, because errors keep arriving; the code's grows because two flips in the same block, at any time before the final measurement, beat the majority vote. Part of the code's advantage comes from the readout: the majority vote also corrects one misread data bit."""),

md("""## Step 6: syndrome statistics

How often does a round signal an error? Run the cell: for 3 rounds it gives the exact fraction of shots with a detection event in each round (either of the two syndrome bits changed), and the same from your 4,000-shot run with its σ."""),
code("""exact_rate = [0.0] * 4
run_rate = [0] * 4
for k, p in exact3.items():
    for i, e in enumerate(detection_events(k, 3)):
        if tuple(e) != (0, 0):
            exact_rate[i] += p
for k, c in counts.items():
    for i, e in enumerate(detection_events(k, 3)):
        if tuple(e) != (0, 0):
            run_rate[i] += c
labels = ["round 1", "round 2", "round 3", "final data"]
for lab, pe, k in zip(labels, exact_rate, run_rate):
    f = k / SHOTS
    print(f"{lab:>11s}: exact {pe:.4f}   your run {f:.4f} +- {fraction_sigma(pe, SHOTS):.4f}")"""),
md(f"""**What to notice.** About {NUM['r'][1]:.0%} of shots show a detection event in round 2 (exact **{NUM['r'][1]:.4f}**), far more than the logical error: most detected errors are corrected or come from the ancillas and their readout, not from the stored bit. Round 1 is lower ({NUM['r'][0]:.4f}) because its syndrome is compared with a perfect (0, 0), while every later round compares two noisy syndromes, so a single wrong syndrome makes two events, one when it appears and one when it goes away. The comparison with the final data ({NUM['r'][3]:.4f}) involves the data readout instead of the ancilla readout. In a real device, the pattern of detection events in space and time is what a decoder such as minimum-weight matching works with."""),

md("""## Step 7: detect instead of correct

A code that corrects one error can **detect** more. Instead of trusting the majority vote, you can throw away every shot that showed any detection event and keep only the "quiet" ones. This is **post-selection**. It lowers the error of the shots you keep, at the price of the shots you discard; it works for a memory test, but a long computation cannot simply be restarted every time an error appears.

Run the cell: it compares the logical error of all shots with that of the kept shots, exact and from your run."""),
code("""def quiet(k, rounds):
    return all(tuple(e) == (0, 0) for e in detection_events(k, rounds))

kept_p = sum(p for k, p in exact3.items() if quiet(k, 3))
kept_err = sum(p for k, p in exact3.items() if quiet(k, 3) and decode(k, 3) != 0) / kept_p
kept_shots = sum(c for k, c in counts.items() if quiet(k, 3))
kept_wrong = sum(c for k, c in counts.items() if quiet(k, 3) and decode(k, 3) != 0)
print(f"exact: all shots {logical_error(exact3, 3):.4f};  kept {kept_p:.1%} of shots, their logical error {kept_err:.5f}")
print(f"your run: kept {kept_shots} of {SHOTS} shots, {kept_wrong} of them wrong "
      f"({kept_wrong / kept_shots:.5f} +- {fraction_sigma(kept_err, kept_shots):.5f})")"""),
md(f"""**What to notice.** Post-selection keeps about **{NUM['kept']:.0%}** of the shots, and among them the logical error falls to about **{NUM['kerr']:.4f}**, roughly six times lower. With 4,000 shots only a handful of kept shots are wrong, so the measured value has a large relative uncertainty: small error rates need many shots. Detection is cheap and powerful for experiments that can be repeated, which is why early demonstrations of codes often report post-selected results; correction is what a long computation needs."""),

md("""## Step 8: break-even

The code helps only if its extra gates add fewer errors than it removes. Run the cell: it scans P2 (and the idle error with it) from 0.005 to 0.08 for 3 rounds, with READOUT = 0.02, and finds the first P2 where the code is worse than a bare qubit."""),
code("""p2_values = [round(0.005 * i, 3) for i in range(1, 17)]
code_err = [logical_error(record_distribution(memory_circuit(3), p2=p), 3) for p in p2_values]
bare_err = [record_distribution(bare_circuit(3), p2=p).get("1", 0.0) for p in p2_values]
for p, e, b in zip(p2_values, code_err, bare_err):
    print(f"P2 = {p:.3f}:  code {e:.4f}   bare {b:.4f}   {'code better' if e < b else 'bare better'}")

fine = [round(0.001 * i, 3) for i in range(5, 81)]
cross = next(p for p in fine if logical_error(record_distribution(memory_circuit(3), p2=p), 3)
             > record_distribution(bare_circuit(3), p2=p).get("1", 0.0))
print("\\nBreak-even (first P2 where the code is worse, steps of 0.001):", cross)
plt.figure(figsize=(6, 3.6))
plt.plot(p2_values, code_err, "o-", label="repetition code, 3 rounds")
plt.plot(p2_values, bare_err, "s--", label="bare qubit")
plt.axvline(cross, color="grey", ls=":")
plt.xlabel("P2 (and idle error)"); plt.ylabel("logical error"); plt.legend(); plt.tight_layout(); plt.show()"""),
md(f"""**What to notice.** The code wins below P2 ≈ **{NUM['cross']:.3f}** and loses above it. This is the repetition code's version of a **threshold**: below it, encoding helps, and a longer code (distance 5, 7, ...) would help more; above it, more qubits and more gates only add errors. Real devices are well below this break-even for the bit-flip code (IBM's median CZ error is about 0.0013 to 0.003), which is why the repetition code works on hardware; the hard part is the surface code, which must correct phase flips as well."""),

*personal_step(9, ["ROUNDS", "P2", "READOUT"], "ROUNDS (1 to 5), P2 and READOUT (0.005 to 0.030)",
               "ROUNDS, P2, READOUT = 3, 0.01, 0.02 gives " + str(R.personal_value(3, 0.01, 0.02)), None, None,
               "100,000 times the exact logical error of a stored 0 after ROUNDS rounds with your noise model (idle error equal to "
               "P2), decoded by majority vote, rounded to a whole number."),
check_cell(hidden_check, "l3_m6_rep_check", "check_l3_m6_rep",
           "fraction_sigma, expectation_sigma, combined_sigma, memory_circuit, detection_events, decode,\n"
           "                                          ROUNDS, P2, READOUT"),
*ending("""- **Only one kind of error is tested.** A stored 0 or 1 suffers only from bit flips. The noise model also makes phase flips, which this code cannot see; a superposition such as (|000⟩ + |111⟩)/√2 would lose its phase, as you saw in Module 2. A real memory must protect both, which needs a larger code such as the surface code.
- **The decoder is simple.** The majority vote ignores the syndrome history. A decoder that uses the detection events in time (for example minimum-weight matching) can tell a measurement error from a data error and does better with more rounds.
- **The noise model is simple.** Errors are independent and the same for every qubit and gate; real devices have crosstalk, leakage, drifting calibrations and errors that hit several qubits at once (and readout that disturbs neighbouring qubits).
- **Exact values, sampled runs.** Your 4,000-shot run measures each rate with an uncertainty; small rates, such as the post-selected error, need far more shots for a precise value.
- **No connectivity.** Here any qubit can share a CX with any other. On a heavy-hex chip the ancillas must sit next to the data qubits, which IBM's layouts allow for this code, but larger codes need careful layouts.""",
       "the **Capstone Quiz — Repetition Code**"),
]

nb = nbf.v4.new_notebook(cells=cells, metadata=META)
for i, c in enumerate(nb.cells):
    c["id"] = f"qcl3m6r{i:02d}"
out = ROOT / "content" / "level3" / NAME
nbf.write(nb, out)
print("wrote", out)
