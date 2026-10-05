"""Build the Level 2 Module 6 project notebooks (the mini-project) from checks/l2_m6_*_check.py.

Browser notebooks (content/level2):  Grover with noise, phase estimation of 1/3 with noise, QAOA with noise,
                                      and the browser version of the transpiler comparison (saved runs).
Colab notebooks (colab/level2):       the transpiler comparison with real Qiskit, and the optional real-hardware run.
"""
import shutil
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from l2_m6_text import (VERSION, md, code, common_intro, step0_browser, step1_statistics, step1_test, step2_noise,  # noqa: E402
                        limits_and_summary, check_source)
import l2_m6_grover_nb as grover  # noqa: E402
import l2_m6_qpe_nb as qpe  # noqa: E402
import l2_m6_qaoa_nb as qaoa  # noqa: E402
import l2_m6_transpiler_nb as tr  # noqa: E402
import l2_m6_hardware_nb as hw  # noqa: E402

META_BROWSER = {"kernelspec": {"name": "python", "display_name": "Python (Pyodide)", "language": "python"},
                "language_info": {"name": "python"}}
META_COLAB = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
              "colab": {"provenance": []}}


def write(cells, path, meta, prefix):
    nb = nbf.v4.new_notebook(cells=cells, metadata=meta)
    for i, c in enumerate(nb.cells):
        c["id"] = f"{prefix}{i:02d}"
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print("wrote", path.relative_to(ROOT), len(cells), "cells")


OUT = ROOT / "content" / "level2"
COLAB = ROOT / "colab" / "level2"
write(grover.cells(), OUT / "QC-L2-M6-project-grover-noise.ipynb", META_BROWSER, "qcl2m6g")
write(qpe.cells(), OUT / "QC-L2-M6-project-phase-estimation-noise.ipynb", META_BROWSER, "qcl2m6p")
write(qaoa.cells(), OUT / "QC-L2-M6-project-qaoa-noise.ipynb", META_BROWSER, "qcl2m6q")
write(tr.cells(colab=False), OUT / "QC-L2-M6-project-transpiler-browser.ipynb", META_BROWSER, "qcl2m6t")
write(tr.cells(colab=True), COLAB / "QC-L2-M6-project-transpiler-colab.ipynb", META_COLAB, "qcl2m6c")
write(hw.cells(), COLAB / "QC-L2-M6-hardware-colab.ipynb", META_COLAB, "qcl2m6h")
shutil.copyfile(ROOT / "content" / "level1" / "qsim.py", OUT / "qsim.py")
print("synced qsim.py")
