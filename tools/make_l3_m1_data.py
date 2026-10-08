"""Save the T1, T2 and readout errors of FakePittsburgh (a model of the 156-qubit IBM computer ibm_pittsburgh,
from a calibration snapshot that ships with qiskit-ibm-runtime 0.50.0) for the Level 3 Module 1 lab.
Writes content/level3/l3_m1_device_data.json."""
import json
from pathlib import Path
from qiskit_ibm_runtime.fake_provider import FakePittsburgh
import qiskit_ibm_runtime

ROOT = Path(__file__).resolve().parents[1]
b = FakePittsburgh()
t = b.target
rows = []
for q in range(b.num_qubits):
    p = t.qubit_properties[q]
    rows.append({"qubit": q, "t1_us": round(p.t1 * 1e6, 1), "t2_us": round(p.t2 * 1e6, 1),
                 "readout_error": round(t["measure"][(q,)].error, 5)})
data = {"backend": b.name, "num_qubits": b.num_qubits,
        "calibration_date": str(b.properties().last_update_date.date()),
        "source": f"qiskit_ibm_runtime {qiskit_ibm_runtime.__version__}, fake_provider.FakePittsburgh",
        "sx_duration_ns": round(t["sx"][(0,)].duration * 1e9, 1), "qubits": rows}
(ROOT / "content/level3/l3_m1_device_data.json").write_text(json.dumps(data, indent=1))
print("wrote", len(rows), "qubits;", data["calibration_date"], data["sx_duration_ns"], "ns")
