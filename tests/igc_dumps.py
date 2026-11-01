"""What in IGC's dumps of a binary shows a miscompile that ``xpu_transforms.IGC_OPTIONS``
avoids, for the tests of ``tests/test_xpu_igc_workarounds.py``. The compile does not
check this: IGC versions differ. ``agent-tools/igc_binary_check.py`` in
helion-mlir-dev reports the same and more for any dump.

- A vector compare and select in IGC's final LLVM IR, or in its vISA a min/max of
  execution size 1 into a multi-element variable: element 0 is written and the rest
  read stale registers.
- A value used before its def in a block of IGC's optimized LLVM IR, the signature of
  ``VectorizerAllowFPTRUNC``'s bug: the conversion reads registers an earlier kernel
  left.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_DECL = re.compile(r"\s*\.decl (V\d+) .*?num_elts=(\d+)(?:.*?alias=<(V\d+),)?")
_NARROW = re.compile(r"\s*(?:max|min)(?:\.\w+)?\s*\([^,)]*,?\s*1\)\s*(V\d+)\(")
_VECTOR_COMPARE = re.compile(r"\bfcmp \w+ <\d+ x float>")
_DEF = re.compile(r"^\s*(%[\w.$-]+) = (.*)$")
_VALUE = re.compile(r"%[\w.$-]+")
_BLOCK_START = re.compile(r"^[\w.$-]+:")


def narrow_minmax(visaasm: Path) -> list[str]:
    """Size-1 min/max instructions into a variable of several elements: how IGC
    emits a vector compare and select it takes for a min/max."""
    elements: dict[str, int] = {}
    alias: dict[str, str] = {}
    found: list[str] = []
    lines = visaasm.read_text(errors="replace").splitlines()
    for line in lines:
        if match := _DECL.match(line):
            elements[match[1]] = int(match[2])
            if match[3]:
                alias[match[1]] = match[3]
    for line in lines:
        if not (match := _NARROW.match(line.split("///")[0])):
            continue
        variable = match[1]
        while variable in alias:
            variable = alias[variable]
        if elements.get(variable, 1) > 1:
            found.append(line.strip())
    return found


def use_before_def(optimized_ll: Path) -> list[str]:
    """Instructions of an LLVM IR file that use a value defined later in the same
    block (phis excepted)."""
    found: list[str] = []
    block: list[str] = []

    def flush() -> None:
        defs: dict[str, int] = {}
        for index, line in enumerate(block):
            if match := _DEF.match(line):
                defs[match[1]] = index
        for index, line in enumerate(block):
            match = _DEF.match(line)
            right = match[2] if match else line
            if right.lstrip().startswith("phi "):
                continue
            if any(defs.get(value, -1) > index for value in _VALUE.findall(right)):
                found.append(line.strip())

    for line in optimized_ll.read_text(errors="replace").splitlines():
        if line.startswith(("define ", "}")) or _BLOCK_START.match(line):
            flush()
            block = []
        elif line.startswith("  "):
            block.append(line.rstrip())
    flush()
    return found


def igc_dump_problems(files: list[Path]) -> list[str]:
    """What shows a miscompile in IGC's dump ``files``; one entry saying so if they
    hold no vISA, as then nothing could be checked."""
    problems: list[str] = []
    visas = [path for path in files if path.suffix == ".visaasm"]
    if not visas:
        problems.append("IGC wrote no vISA to check")
    for visa in visas:
        problems += [f"{visa.name}: {line}" for line in narrow_minmax(visa)]
    for path in files:
        if path.name.endswith("_codegen.ll"):
            text = path.read_text(errors="replace")
            if match := _VECTOR_COMPARE.search(text):
                problems.append(f"{path.name}: {match[0]}")
        elif path.name.endswith("_optimized.ll"):
            problems += [f"{path.name}: {line}" for line in use_before_def(path)]
    return problems
