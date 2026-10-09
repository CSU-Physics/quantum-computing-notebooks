"""Level 3 Module 5 data: published hardware figures and two IBM calibration snapshots.
Writes content/level3/l3_m5_data.json. Run from the repository root with qiskit-ibm-runtime 0.50.0 installed.

platforms: one row per device or experiment, each figure exactly as published, with its source, its date and what kind of
number it is (mean, median, average from randomized benchmarking, bound). Missing figures are null: no source gave them.
devices: per-qubit and per-pair calibration of ibm_boston (Heron r3) and ibm_miami (Nighthawk r1), from the snapshots
IBM ships in qiskit-ibm-runtime 0.50.0 (fake_provider FakeBoston and FakeMiami, both 17 April 2026)."""
import json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
from qiskit_ibm_runtime import fake_provider as fp

ROOT = Path(__file__).resolve().parents[1]


def sig(x, n=4):
    return float(f"{x:.{n}g}")


PLATFORMS = [
    dict(name="IBM Heron r3 (ibm_boston)", platform="superconducting", qubits=156,
         e1=None, e2=None, e_ro=None, t1_us=None, t2_us=None, t2q_us=None,   # filled below from the calibration snapshot
         kind="median over qubits and pairs, calibration of 17 Apr 2026",
         source="IBM calibration data shipped in qiskit-ibm-runtime 0.50.0 (FakeBoston)", year=2026),
    dict(name="IBM Nighthawk r1 (ibm_miami)", platform="superconducting", qubits=120,
         e1=None, e2=None, e_ro=None, t1_us=None, t2_us=None, t2q_us=None,
         kind="median over qubits and pairs, calibration of 17 Apr 2026",
         source="IBM calibration data shipped in qiskit-ibm-runtime 0.50.0 (FakeMiami)", year=2026),
    dict(name="Google Willow", platform="superconducting", qubits=105,
         e1=0.00035, e2=0.0033, e_ro=0.0077, t1_us=68, t2_us=None, t2q_us=None,
         kind="mean, all qubits operated simultaneously (chip 1, error correction)",
         source="Google Quantum AI, Willow spec sheet (2024)", year=2024),
    dict(name="Quantinuum Helios", platform="trapped ions (barium-137)", qubits=98,
         e1=0.000025, e2=0.00079, e_ro=0.00048, t1_us=None, t2_us=None, t2q_us=70,
         kind="average from randomized benchmarking; SPAM error as readout",
         source="Quantinuum, 'Helios: a 98-qubit trapped-ion quantum computer', arXiv:2511.05465 (2025)", year=2025),
    dict(name="IonQ Aria 1", platform="trapped ions (ytterbium-171)", qubits=25,
         e1=0.0005, e2=0.004, e_ro=0.0039, t1_us=None, t2_us=1_000_000, t2q_us=600,
         kind="as listed by the cloud provider; two-qubit value not SPAM corrected",
         source="Microsoft Azure Quantum documentation, IonQ provider page (2026)", year=2026),
    dict(name="Harvard / MIT / QuEra array", platform="neutral atoms (rubidium-87)", qubits=60,
         e1=0.0003, e2=0.0048, e_ro=None, t1_us=None, t2_us=None, t2q_us=0.27,
         kind="CZ fidelity 99.52% from repeated gates; up to 60 atoms in parallel",
         source="Evered et al., Nature 622, 268 (2023)", year=2023),
    dict(name="Caltech 6,100-atom array", platform="neutral atoms (caesium-133)", qubits=6100,
         e1=None, e2=None, e_ro=None, t1_us=None, t2_us=12_600_000, t2q_us=None,
         kind="trapping and coherence only, no two-qubit gates; coherence 12.6 s",
         source="Manetsch et al., Nature 647, 60 (2025)", year=2025),
    dict(name="Diraq / imec silicon unit cells", platform="silicon spin qubits", qubits=2,
         e1=0.01, e2=0.01, e_ro=0.001, t1_us=None, t2_us=None, t2q_us=None,
         kind="bounds: one- and two-qubit fidelities above 99% on all four devices; SPAM up to 99.9%",
         source="Steinacker et al., Nature 646, 81 (2025)", year=2025),
    dict(name="Intel Tunnel Falls", platform="silicon spin qubits", qubits=12,
         e1=None, e2=None, e_ro=None, t1_us=None, t2_us=112, t2q_us=None,
         kind="Hahn-echo T2 across the 12-dot array; fidelities not reported",
         source="George et al., arXiv:2410.16583 (2024)", year=2024),
    dict(name="PsiQuantum Omega", platform="photons (silicon photonics)", qubits=None,
         e1=None, e2=0.0078, e_ro=0.0002, t1_us=None, t2_us=None, t2q_us=None,
         kind="two-qubit fusion 99.22%, single-qubit SPAM 99.98%, not counting photon loss",
         source="PsiQuantum, Nature (2025), doi:10.1038/s41586-025-08820-7", year=2025),
    dict(name="Xanadu Aurora", platform="photons (squeezed light)", qubits=12,
         e1=None, e2=None, e_ro=None, t1_us=None, t2_us=None, t2q_us=None,
         kind="12 qubit modes per clock cycle, 35 chips; about 14 dB end-to-end loss",
         source="Aghaee Rad et al., Nature 638, 912 (2025)", year=2025),
]


def device(b):
    t, p = b.target, b.properties()
    n = b.num_qubits
    q = dict(t1_us=[round(p.t1(i) * 1e6, 1) for i in range(n)], t2_us=[round(p.t2(i) * 1e6, 1) for i in range(n)],
             e1=[sig(t["sx"][(i,)].error) for i in range(n)], e_ro=[sig(p.readout_error(i)) for i in range(n)])
    pairs = sorted((a, c) for (a, c) in t["cz"] if a < c)
    return dict(name=b.backend_name.replace("fake_", "ibm_"), qubits=n, calibrated=str(p.last_update_date)[:10],
                qubit_data=q, pairs=[[a, c] for a, c in pairs], e2=[sig(t["cz"][(a, c)].error) for a, c in pairs],
                t2q_ns=[round(t["cz"][(a, c)].duration * 1e9, 1) for a, c in pairs])


if __name__ == "__main__":
    import numpy as np
    devs = {d["name"]: d for d in (device(fp.FakeBoston()), device(fp.FakeMiami()))}
    for row, key in ((PLATFORMS[0], "ibm_boston"), (PLATFORMS[1], "ibm_miami")):
        d = devs[key]
        ok = lambda v: [x for x in v if x < 1]  # noqa: E731  error 1.0 marks a qubit or pair IBM reports as unusable
        row.update(e1=sig(float(np.median(ok(d["qubit_data"]["e1"])))), e2=sig(float(np.median(ok(d["e2"])))),
                   e_ro=sig(float(np.median(ok(d["qubit_data"]["e_ro"])))),
                   t1_us=round(float(np.median(d["qubit_data"]["t1_us"])), 1), t2_us=round(float(np.median(d["qubit_data"]["t2_us"])), 1),
                   t2q_us=round(float(np.median(d["t2q_ns"])) / 1000, 3))
    out = dict(note="Published figures as stated by each source; see each row's kind and source. Missing figures are null.",
               platforms=PLATFORMS, devices=devs)
    (ROOT / "content/level3/l3_m5_data.json").write_text(json.dumps(out, separators=(",", ":")) + "\n")
    for r in PLATFORMS[:2]:
        print(r)
    print({k: (v["qubits"], len(v["pairs"]), v["calibrated"]) for k, v in devs.items()})
