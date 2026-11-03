"""What in IGC's dumps of a binary shows a miscompile that ``xpu_transforms.IGC_OPTIONS``
and ``xevm_lane_conversions`` avoid, for the tests of ``tests/test_xpu_igc_workarounds.py``. The compile does not
check this: IGC versions differ. ``agent-tools/igc_binary_check.py`` in
helion-mlir-dev reports the same and more for any dump.

- A vector compare and select in IGC's final LLVM IR, or in its vISA a min/max of
  execution size 1 into a multi-element variable: element 0 is written and the rest
  read stale registers.
- A vector ``llvm.*`` call left in IGC's final LLVM IR, other than what its codegen handles per
  lane: ``llvm.fabs.v16f32`` and ``llvm.abs.v16i32`` write one row of the vector
  (``xevm_lane_compares``).
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
_VECTOR_COMPARE = re.compile(r"\b[fi]cmp \w+ <\d+ x \w+>")
_LLVM_CALL = re.compile(r"\bcall\b[^@]*@llvm\.([\w.]+)\(")
_VECTOR_INTRINSICS_OK = frozenset(
    {"fshl", "smin", "smax", "umin", "umax", "exp2", "maxnum"}
)
"""Written whole, row by row: ``IGCVectorizer``'s ``exp2`` and ``maxnum`` are 16 ``exp`` or ``max``
of size 16 in the vISA on 21 and 97, with nothing read unwritten (B50, IGC 2.41)."""
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


def vector_intrinsics(codegen_ll: Path) -> list[str]:
    """Vector ``llvm.*`` calls in IGC's final LLVM IR that its codegen doesn't write whole, so
    far as known: anything but :data:`_VECTOR_INTRINSICS_OK` and ``llvm.genx.*``."""
    found: set[str] = set()
    for name in _LLVM_CALL.findall(codegen_ll.read_text(errors="replace")):
        base, *types = name.split(".")
        if (
            base != "genx"
            and base not in _VECTOR_INTRINSICS_OK
            and any(re.fullmatch(r"v\d+\w+", part) for part in types)
        ):
            found.add(name)
    return sorted(found)


_TYPE_BYTES = {
    "q": 8,
    "uq": 8,
    "df": 8,
    "hf": 2,
    "bf": 2,
    "w": 2,
    "uw": 2,
    "b": 1,
    "ub": 1,
}
_VISA_DECL = re.compile(r"\s*\.decl (\w+) v_type=\w type=(\w+) num_elts=(\d+)(.*)")
_VISA_ALIAS = re.compile(r"alias=<(\w+), *(\d+)>")
_VISA_OP = re.compile(
    r"^\s*(\(.*?\)\s*)?(?P<op>[a-z_.0-9]+)\s*\((?:M\d+(?:_NM)?|NM)?,?\s*(?P<size>\d+)\)\s*(?P<rest>.*)$"
)
_VISA_OPERAND = re.compile(
    r"(\(-\)|\(abs\)|\(sat\))?(?P<v>V\d+)\((?P<r>\d+),(?P<c>\d+)\)(?:<(?P<region>[^>]*)>)?"
)
_VISA_OPAQUE = ("raw_send", "lsc_", "send", "dpas", "fcvt", "intrinsic")
_GRF_BYTES = 64


def elements_read_never_written(visaasm: Path) -> list[str]:
    """vISA variables some of whose elements are read and written nowhere in the kernel, with
    a line reading them, as in ``igc-vector-minmax/visa_uninit.py``. A variable an opaque op
    (a send) touches counts as fully written."""
    lines = visaasm.read_text(errors="replace").splitlines()
    sizes: dict[str, int] = {}
    alias: dict[str, tuple[str, int]] = {}
    machine_registers: set[str] = set()
    for line in lines:
        if match := _VISA_DECL.match(line):
            sizes[match[1]] = _TYPE_BYTES.get(match[2], 4)
            # vISA's copy of r0, the thread's payload: a low-numbered eight dwords.
            if match[2] == "d" and match[3] == "8" and int(match[1][1:]) < 60:
                machine_registers.add(match[1])
            if found := _VISA_ALIAS.search(match[4]):
                alias[match[1]] = (found[1], int(found[2]))

    def root(name: str) -> tuple[str, int]:
        offset = 0
        while name in alias:
            name, more = alias[name]
            offset += more
        return name, offset

    written: dict[str, set[int]] = {}
    read: dict[str, dict[int, str]] = {}
    opaque: set[str] = set()
    for line in lines:
        code = line.split("///")[0]
        match = _VISA_OP.match(code)
        if code.lstrip().startswith("."):
            continue
        if not match or match["op"].startswith(_VISA_OPAQUE):
            opaque.update(root(v)[0] for v in re.findall(r"\bV\d+\b", code))
            continue
        size = int(match["size"])
        operands = list(_VISA_OPERAND.finditer(match["rest"]))
        # A compare's destination is a predicate, or a variable of 0s and 1s.
        compare = match["op"].startswith("cmp") and not match[
            "rest"
        ].lstrip().startswith("V")
        if operands and not compare:
            first = operands[0]
            variable, offset = root(first["v"])
            width = sizes[first["v"]]
            step = int(first["region"] or "1") * width
            base = offset + int(first["r"]) * _GRF_BYTES + int(first["c"]) * width
            for lane in range(size):
                written.setdefault(variable, set()).update(
                    range(base + lane * step, base + lane * step + width)
                )
            operands = operands[1:]
        for operand in operands:
            variable, offset = root(operand["v"])
            width = sizes[operand["v"]]
            parts = re.split("[;,]", operand["region"] or "0;1,0")
            vertical, row, horizontal = (
                (int(parts[0]), int(parts[1]), int(parts[2]))
                if len(parts) == 3
                else (0, 1, 0)
            )
            base = offset + int(operand["r"]) * _GRF_BYTES + int(operand["c"]) * width
            for lane in range(size):
                place = (
                    base
                    + (lane // max(row, 1)) * vertical * width
                    + (lane % max(row, 1)) * horizontal * width
                )
                for byte in range(place, place + width):
                    read.setdefault(variable, {}).setdefault(byte, code.strip())
    return [
        f"{variable}: {lines_read[min(missing)]}"
        for variable, lines_read in read.items()
        if variable not in opaque and variable not in machine_registers
        if (missing := set(lines_read) - written.get(variable, set()))
    ]


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
            problems += [f"{path.name}: {name}" for name in vector_intrinsics(path)]
        elif path.name.endswith("_optimized.ll"):
            problems += [f"{path.name}: {line}" for line in use_before_def(path)]
    return problems
