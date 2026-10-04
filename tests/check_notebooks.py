#!/usr/bin/env python3
"""Execute every tutorial notebook with the Octave kernel and fail on errors.

A notebook passes when no code cell produces an ``error`` output and no
stream output contains an Octave error line (``error: ...`` at line start).
Printed numerical accuracies such as ``Max error: 5.8e-14`` are not errors.

Requirements: jupyter nbclient/nbformat and octave_kernel
(``pip install nbclient nbformat octave_kernel``).

Usage: python3 tests/check_notebooks.py [notebook ...]
"""
import pathlib
import re
import sys

import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT = sorted(ROOT.glob("notebooks/*.ipynb")) + [
    ROOT / "flagship_project" / "project_notebook.ipynb"
]
OCTAVE_ERROR = re.compile(r"^error: ", re.MULTILINE)


def check(path: pathlib.Path) -> list[str]:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        kernel_name="octave",
        timeout=900,
        allow_errors=True,
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
    problems = []
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    for i, cell in enumerate(code_cells):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                problems.append(f"cell {i}: {out.get('ename')}: {out.get('evalue')}")
            text = "".join(out.get("text", ""))
            for m in OCTAVE_ERROR.finditer(text):
                line = text[m.start():].splitlines()[0]
                problems.append(f"cell {i}: {line}")
    return problems


def main(argv: list[str]) -> int:
    paths = [pathlib.Path(a).resolve() for a in argv] or DEFAULT
    failed = 0
    for p in paths:
        problems = check(p)
        status = "PASS" if not problems else "FAIL"
        print(f"{status}  {p.relative_to(ROOT)}")
        for msg in problems:
            print(f"      {msg}")
        failed += bool(problems)
    print(f"\n{len(paths) - failed}/{len(paths)} notebooks executed without errors")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
