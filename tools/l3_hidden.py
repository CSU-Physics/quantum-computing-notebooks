"""The check cell of a Level 3 notebook, with the check code kept in its own file (6 Oct 2026).

Until 6 Oct 2026 every check cell contained the whole check file, including the reference solutions of the functions
learners write. A review found that this lets a learner copy them, so the verification value showed working code,
not code the learner wrote. Now the check code stays in content/level3/<module>.py (served next to the notebook in
JupyterLite, and downloaded from GitHub Pages in Colab) and the cell only imports the check function.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES_FILES = "https://csu-physics.github.io/quantum-computing-notebooks/files/level3/"

NOTE = ("# The course's check code is in {files}{where}.\n"
        "# It is kept out of the notebook so that the course's reference solutions are not on screen.\n"
        "# You may open the file to see how your functions are tested, but write your own functions first.\n")


def hidden_check(modules, func, call, colab=False):
    """modules: the check files (without .py) the function needs, the one that defines func last."""
    for m in modules:                       # publish the check file next to the notebooks
        shutil.copyfile(ROOT / "checks" / f"{m}.py", ROOT / "content" / "level3" / f"{m}.py")
    files = " and ".join(f"{m}.py" for m in modules)
    where = " (this cell downloads it from the course website)" if colab else ", next to this notebook"
    src = NOTE.format(files=files, where=where)
    if colab:
        src += "import urllib.request\n"
        for m in modules:
            src += f'urllib.request.urlretrieve("{PAGES_FILES}{m}.py", "{m}.py")\n'
    src += f"from {modules[-1]} import {func}\n\n" + call.strip()
    return src
