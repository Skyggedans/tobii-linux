#!/usr/bin/env python3
"""Recover the Tobii Stream Engine ABI from tobii_stream_engine.dll (x64).

No public 4.x header exists, so the export list, argument count, register
widths and float positions of every entry point are read off the DLL's own
machine code and cross-checked against the headers we ship. Stdlib only;
shells out to `objdump` (binutils reads PE fine).

    dll_abi.py exports [--check crates/tobii-ffi/abi-symbols.txt]
    dll_abi.py args [--out fixtures/abi/args.tsv]
    dll_abi.py headers [--args TSV] [--include DIR] [-v]
    dll_abi.py dis NAME [--len 0x200]
    dll_abi.py callsites 0xOFFSET

`args` is a heuristic: it records the first READ of each argument register
(rcx/rdx/r8/r9, xmm0-3, stack slots from [rsp+0x28]) before anything writes
it. An argument the callee never touches is therefore missed, which makes the
result a LOWER bound on arity; false positives are rare. `headers` only fails
when the DLL reads more arguments than a prototype declares, or when a float
sits where the prototype has an integer or pointer.

The DLL is not part of the build. Point at it with --dll or TOBII_DLL.
"""

import argparse
import os
import re
import struct
import subprocess
import sys
from collections import OrderedDict

DEFAULT_DLL = os.environ.get(
    "TOBII_DLL", os.path.expanduser("~/Work/tobii-sys/tobii_stream_engine.dll")
)


# --------------------------------------------------------------- PE exports

def pe_exports(path):
    """Return (image_base, OrderedDict name -> virtual address)."""
    data = open(path, "rb").read()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        sys.exit(f"{path}: not a PE file")
    section_count = struct.unpack_from("<H", data, pe + 6)[0]
    opt = pe + 24
    if struct.unpack_from("<H", data, opt)[0] != 0x20B:
        sys.exit(f"{path}: not PE32+ (expected an x64 DLL)")
    image_base = struct.unpack_from("<Q", data, opt + 24)[0]
    export_rva = struct.unpack_from("<I", data, opt + 112)[0]
    headers = opt + struct.unpack_from("<H", data, pe + 20)[0]
    sections = []
    for i in range(section_count):
        off = headers + 40 * i
        vsize, vaddr, rsize, raw = struct.unpack_from("<IIII", data, off + 8)
        sections.append((vaddr, max(vsize, rsize), raw))

    def offset_of(rva):
        for vaddr, size, raw in sections:
            if vaddr <= rva < vaddr + size:
                return raw + (rva - vaddr)
        raise ValueError(f"rva {rva:#x} is outside every section")

    table = offset_of(export_rva)
    name_count = struct.unpack_from("<I", data, table + 24)[0]
    func_rva, name_rva, ord_rva = struct.unpack_from("<III", data, table + 28)
    funcs, names, ordinals = offset_of(func_rva), offset_of(name_rva), offset_of(ord_rva)
    exports = {}
    for i in range(name_count):
        ptr = offset_of(struct.unpack_from("<I", data, names + 4 * i)[0])
        name = data[ptr:data.index(b"\0", ptr)].decode()
        index = struct.unpack_from("<H", data, ordinals + 2 * i)[0]
        exports[name] = image_base + struct.unpack_from("<I", data, funcs + 4 * index)[0]
    return image_base, OrderedDict(sorted(exports.items()))


# ------------------------------------------------------------- disassembly

_LINE = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$")


def disassemble(path, start, stop):
    """[(address, mnemonic, operands)] for [start, stop) in Intel syntax."""
    result = subprocess.run(
        ["objdump", "-d", "-M", "intel", "--no-show-raw-insn",
         f"--start-address={start:#x}", f"--stop-address={stop:#x}", path],
        capture_output=True, text=True, check=True,
    )
    out = []
    for line in result.stdout.splitlines():
        match = _LINE.match(line)
        if match:
            mnemonic = match.group(2)
            operands = match.group(3).split("#")[0].strip()
            # MSVC pads hot-patchable entry points with a redundant REX
            # prefix (`40 57` = push rdi); objdump prints it as its own token.
            if mnemonic.startswith("rex") and operands:
                mnemonic, _, operands = operands.partition(" ")
                operands = operands.strip()
            out.append((int(match.group(1), 16), mnemonic, operands))
    return out


ARG_ALIASES = {
    "rcx": ("rcx", "ecx", "cx", "cl", "ch"),
    "rdx": ("rdx", "edx", "dx", "dl", "dh"),
    "r8": ("r8", "r8d", "r8w", "r8b"),
    "r9": ("r9", "r9d", "r9w", "r9b"),
    "xmm0": ("xmm0",), "xmm1": ("xmm1",), "xmm2": ("xmm2",), "xmm3": ("xmm3",),
}
CANONICAL = {alias: reg for reg, aliases in ARG_ALIASES.items() for alias in aliases}
ARG_INDEX = {"rcx": 0, "rdx": 1, "r8": 2, "r9": 3,
             "xmm0": 0, "xmm1": 1, "xmm2": 2, "xmm3": 3}


def width_of(alias):
    if alias.startswith("xmm"):
        return 128
    if alias in ("rcx", "rdx", "r8", "r9"):
        return 64
    if alias in ("ecx", "edx") or alias.endswith("d"):
        return 32
    if alias in ("cx", "dx") or alias.endswith("w"):
        return 16
    return 8


# Mnemonics whose first operand is written without being read.
PURE_WRITE = {
    "mov", "movzx", "movsx", "movsxd", "movabs", "lea", "movss", "movsd",
    "movaps", "movups", "movd", "movq", "movdqa", "movdqu", "cvtsi2ss",
    "cvtsi2sd", "cvtss2sd", "cvtsd2ss", "cvttss2si", "cvttsd2si", "pop",
}
# Mnemonics that read every operand and write none of the tracked registers.
PURE_READ = {"cmp", "test", "comiss", "ucomiss", "comisd", "ucomisd", "push",
             "call", "jmp"}
REGISTER = re.compile(r"\b(r8[dwb]?|r9[dwb]?|[re]?cx|[re]?dx|c[lh]|d[lh]|xmm[0-3])\b")
STACK_SLOT = re.compile(r"\[rsp\+0x([0-9a-f]+)\]")
# The callee's frame at entry: [rsp] return address, [rsp+8..0x20] home slots
# for rcx/rdx/r8/r9, first stack argument at [rsp+0x28].
FIRST_STACK_ARG = 0x28


def is_pure_write(mnemonic):
    return (mnemonic in PURE_WRITE
            or mnemonic.startswith("set")
            or mnemonic.startswith("cmov"))


def registers_in(text):
    return {CANONICAL[name] for name in REGISTER.findall(text)}


# Idioms that zero a register: a write, not a read of the old value.
ZEROING = {"xor", "xorps", "xorpd", "pxor", "sub"}
# Stack arguments past this index are implausible for this API; anything
# larger is a local the frame tracker failed to resolve.
MAX_STACK_ARG = 12


def jump_target(operands):
    token = operands.split()[0] if operands else ""
    return int(token, 16) if token.startswith("0x") else None


def is_chkstk(instructions, index, eax):
    """`mov eax,N; call __chkstk; sub rsp,rax`: MSVC's large-frame probe."""
    if eax is None or index + 1 >= len(instructions):
        return False
    _, mnemonic, operands = instructions[index + 1]
    return mnemonic == "sub" and operands.replace(" ", "") == "rsp,rax"


def analyze(code, entry, exports_by_address, follow=2, limit=400):
    """Argument reads and constant returns of the function at `entry`.

    `code` is (instructions, index_by_address) for the whole text section.
    The walk is linear from the entry and ends at a `ret`, `int3` or
    unconditional `jmp` that no earlier conditional branch jumps past, so it
    stays inside the function instead of running into the next one. A thunk
    (a `jmp` before any argument is read) is followed up to `follow` times.
    """
    instructions, index_by_address = code
    written, first_read, stack_args, returns = set(), {}, set(), set()
    frame, reach = 0, entry
    epilogue_from = None
    chkstk_size = None
    pending_eax = None
    tail_call = ""
    position = index_by_address.get(entry)
    if position is None:
        return None

    for count in range(limit):
        if position + count >= len(instructions):
            break
        address, mnemonic, operands = instructions[position + count]
        if mnemonic == "int3" and address >= reach:
            break
        parts = [p.strip() for p in operands.split(",")] if operands else []

        # MSVC emits an epilogue (add rsp / pop / ret or jmp) on every early
        # exit, with more body code after it that still runs on the full
        # frame: remember the frame the epilogue started from and restore it
        # once the exit instruction has been passed.
        is_epilogue_step = mnemonic == "pop" or (
            mnemonic == "add" and len(parts) == 2 and parts[0] == "rsp")
        if is_epilogue_step and epilogue_from is None:
            epilogue_from = frame
        elif not is_epilogue_step and mnemonic not in ("ret", "jmp"):
            epilogue_from = None
        if mnemonic == "push":
            frame += 8
        elif mnemonic == "pop":
            frame -= 8
        elif mnemonic in ("sub", "add") and len(parts) == 2 and parts[0] == "rsp":
            if parts[1] == "rax" and chkstk_size is not None:
                delta = chkstk_size
            else:
                try:
                    delta = int(parts[1], 16)
                except ValueError:
                    delta = 0
            frame += delta if mnemonic == "sub" else -delta

        target = jump_target(operands) if mnemonic.startswith("j") else None
        if mnemonic == "jmp":
            if target is not None and not first_read and count < 6:
                if target in exports_by_address:
                    tail_call = exports_by_address[target]
                    break
                if follow > 0:  # thunk into an internal function
                    inner = analyze(code, target, exports_by_address, follow - 1, limit)
                    if inner is not None:
                        inner["tail"] = inner["tail"] or f"{target:#x}"
                        return inner
            if target is not None and entry < target < entry + 0x4000 and target > address:
                reach = max(reach, target)
            if address >= reach:
                break
            if epilogue_from is not None:
                frame, epilogue_from = epilogue_from, None
            continue
        if mnemonic.startswith("j") and target is not None and target > address:
            reach = max(reach, target)

        destination, sources = None, []
        if mnemonic in PURE_READ or mnemonic.startswith("j"):
            sources = parts
        elif mnemonic in ZEROING and len(parts) == 2 and parts[0] == parts[1]:
            destination = parts[0]
        elif is_pure_write(mnemonic) and parts:
            destination, sources = parts[0], parts[1:]
        elif parts:
            destination, sources = parts[0], list(parts)  # read-modify-write
        if destination is not None and "[" in destination:
            sources.append(destination)  # memory destination: its base is read
            destination = None

        for source in sources:
            for register in registers_in(source):
                if register not in written and register not in first_read:
                    alias = next(a for a in REGISTER.findall(source)
                                 if CANONICAL[a] == register)
                    first_read[register] = alias
            slot = STACK_SLOT.search(source)
            if slot and mnemonic != "call":
                offset = int(slot.group(1), 16) - frame
                if offset >= FIRST_STACK_ARG and (offset - FIRST_STACK_ARG) % 8 == 0:
                    index = 4 + (offset - FIRST_STACK_ARG) // 8
                    if index <= MAX_STACK_ARG:
                        stack_args.add(index)

        if destination is not None:
            for register in registers_in(destination):
                if register not in first_read:
                    written.add(register)

        if mnemonic == "mov" and len(parts) == 2 and parts[0] == "eax" \
                and parts[1].startswith("0x"):
            pending_eax = int(parts[1], 16)
        elif mnemonic == "xor" and parts == ["eax", "eax"]:
            pending_eax = 0
        elif mnemonic == "call":
            if is_chkstk(instructions, position + count, pending_eax):
                chkstk_size = pending_eax  # probes the stack, preserves args
            else:
                written |= set(ARG_ALIASES)  # volatile across a call
            pending_eax = None
        elif mnemonic == "ret":
            if pending_eax is not None:
                returns.add(pending_eax)
            pending_eax = None
            if address >= reach:
                break
            if epilogue_from is not None:
                frame, epilogue_from = epilogue_from, None

    integers = sorted((ARG_INDEX[r], r, a) for r, a in first_read.items()
                      if not r.startswith("xmm"))
    floats = sorted((ARG_INDEX[r], r) for r in first_read if r.startswith("xmm"))
    return {
        "int_args": ",".join(f"{r}:{width_of(a)}" for _, r, a in integers),
        "float_args": ",".join(r for _, r in floats),
        "stack_args": ",".join(str(i) for i in sorted(stack_args)),
        "returns": ",".join(str(r) for r in sorted(returns)),
        "tail": tail_call,
    }


def analyze_all(dll):
    """name -> analyze() result, from a single disassembly of the image."""
    _, exports = pe_exports(dll)
    by_address = {address: name for name, address in exports.items()}
    addresses = sorted(by_address)
    instructions = disassemble(dll, addresses[0], addresses[-1] + 0x2000)
    index_by_address = {address: i for i, (address, _, _) in enumerate(instructions)}
    code = (instructions, index_by_address)
    empty = {"int_args": "?", "float_args": "", "stack_args": "", "returns": "", "tail": ""}
    rows = {name: analyze(code, address, by_address) or dict(empty)
            for name, address in exports.items()}
    return exports, rows


# -------------------------------------------------------------- subcommands

def read_symbol_list(path):
    """(expected, extensions) from a '# comment'-annotated symbol list."""
    expected, extensions = set(), set()
    for line in open(path):
        name = line.split("#")[0].strip()
        if not name:
            continue
        expected.add(name)
        if "# extension" in line:
            extensions.add(name)
    return expected, extensions


def command_exports(args):
    _, exports = pe_exports(args.dll)
    if not args.check:
        for name in exports:
            print(name)
        print(f"# {len(exports)} exports", file=sys.stderr)
        return 0
    expected, extensions = read_symbol_list(args.check)
    from_dll = expected - extensions
    actual = set(exports)
    missing, extra = sorted(from_dll - actual), sorted(actual - from_dll)
    for name in missing:
        print(f"listed but not exported by the DLL: {name}")
    for name in extra:
        print(f"exported by the DLL but not listed: {name}")
    print(f"dll {len(actual)}, listed {len(from_dll)} "
          f"(+{len(extensions)} of our own)")
    return 1 if missing or extra else 0


def command_args(args):
    exports, rows = analyze_all(args.dll)
    out = open(args.out, "w") if args.out else sys.stdout
    print("name\tint_args\tfloat_args\tstack_args\treturns\ttail", file=out)
    for name in exports:
        row = rows[name]
        print("\t".join([name, row["int_args"], row["float_args"],
                         row["stack_args"], row["returns"], row["tail"]]), file=out)
    if args.out:
        out.close()
        print(f"wrote {args.out} ({len(exports)} exports)", file=sys.stderr)
    return 0


PROTOTYPE = re.compile(
    r"TOBII_API\s+[\w\s*]+?\s*TOBII_CALL\s+(tobii_\w+)\s*\((.*?)\)\s*;", re.S)
FLOATING = re.compile(r"\b(float|double)\b")


def parse_parameters(text):
    """Split a parameter list, keeping function-pointer parameters whole."""
    parameters, depth, current = [], 0, ""
    for piece in text.split(","):
        current = piece if not current else current + "," + piece
        depth += piece.count("(") - piece.count(")")
        if depth == 0:
            parameters.append(current.strip())
            current = ""
    return [p for p in parameters if p and p != "void"]


def parse_headers(include_dir):
    """name -> (header file, parameter count, indices of float parameters)."""
    prototypes = {}
    for filename in sorted(os.listdir(include_dir)):
        if not filename.endswith(".h"):
            continue
        text = re.sub(r"/\*.*?\*/", "", open(os.path.join(include_dir, filename)).read(),
                      flags=re.S)
        for name, parameters in PROTOTYPE.findall(text):
            listed = parse_parameters(parameters)
            floats = [i for i, p in enumerate(listed)
                      if FLOATING.search(p) and "*" not in p]
            prototypes[name] = (filename, len(listed), floats)
    return prototypes


def command_headers(args):
    prototypes = parse_headers(args.include)
    if args.args:
        rows = {}
        for line in open(args.args):
            fields = line.rstrip("\n").split("\t")
            if fields[0] != "name":
                rows[fields[0]] = dict(zip(
                    ("int_args", "float_args", "stack_args", "returns", "tail"),
                    fields[1:] + [""] * 5))
    else:
        _, rows = analyze_all(args.dll)
    _, extensions = read_symbol_list(args.extensions) if args.extensions else (set(), set())

    errors = notes = 0
    for name, (filename, declared, floats) in sorted(prototypes.items()):
        if name not in rows:
            if name not in extensions:
                print(f"{filename}: {name} is declared but the DLL does not export it")
                errors += 1
            continue
        row = rows[name]
        if row["tail"]:
            continue  # a thunk; its target carries the argument reads
        integers = [ARG_INDEX[f.split(":")[0]] for f in row["int_args"].split(",") if f]
        floating = [ARG_INDEX[f] for f in row["float_args"].split(",") if f]
        stack = [int(f) for f in row["stack_args"].split(",") if f]
        used = set(integers) | set(floating) | set(stack)
        lower_bound = max(used) + 1 if used else 0
        if lower_bound > declared:
            print(f"{filename}: {name}: prototype takes {declared}, "
                  f"the DLL reads argument #{lower_bound}")
            errors += 1
        for index in floating:
            if index not in floats:
                print(f"{filename}: {name}: the DLL reads a float in argument "
                      f"#{index + 1}, the prototype does not declare one")
                errors += 1
        for index in floats:
            if index in integers:
                print(f"{filename}: {name}: the prototype has a float at "
                      f"#{index + 1}, the DLL reads an integer register")
                errors += 1
        if lower_bound < declared and args.verbose:
            print(f"{filename}: {name}: prototype {declared}, DLL reads "
                  f">= {lower_bound} (trailing arguments are never touched)")
            notes += 1
    print(f"{len(prototypes)} prototypes checked: {errors} errors, {notes} notes")
    return 1 if errors else 0


def command_dis(args):
    _, exports = pe_exports(args.dll)
    if args.name not in exports:
        sys.exit(f"{args.name}: not exported by {args.dll}")
    address = exports[args.name]
    print(f"# {args.name} @ {address:#x}")
    for at, mnemonic, operands in disassemble(args.dll, address, address + args.len):
        print(f"{at:x}:\t{mnemonic}\t{operands}")
    return 0


def command_callsites(args):
    """Every instruction touching [reg+OFFSET] — finds where a stored
    callback slot is read back and invoked."""
    text = subprocess.run(
        ["objdump", "-d", "-M", "intel", "--no-show-raw-insn", args.dll],
        capture_output=True, text=True, check=True).stdout
    needle = f"+{args.offset:#x}]"
    for line in text.splitlines():
        if needle in line:
            print(line.strip())
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dll", default=DEFAULT_DLL)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("exports", help="list or verify the export table")
    p.add_argument("--check", metavar="SYMBOL_LIST")
    p.set_defaults(run=command_exports)

    p = sub.add_parser("args", help="derive argument reads per export")
    p.add_argument("--out", metavar="TSV")
    p.set_defaults(run=command_args)

    p = sub.add_parser("headers", help="cross-check our prototypes against the DLL")
    p.add_argument("--include", default="crates/tobii-ffi/include/tobii")
    p.add_argument("--args", metavar="TSV", help="reuse the output of `args`")
    p.add_argument("--extensions", metavar="SYMBOL_LIST",
                   default="crates/tobii-ffi/abi-symbols.txt")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(run=command_headers)

    p = sub.add_parser("dis", help="disassemble one export")
    p.add_argument("name")
    p.add_argument("--len", type=lambda v: int(v, 0), default=0x200)
    p.set_defaults(run=command_dis)

    p = sub.add_parser("callsites", help="find uses of a struct/object offset")
    p.add_argument("offset", type=lambda v: int(v, 0))
    p.set_defaults(run=command_callsites)

    args = parser.parse_args()
    if not os.path.exists(args.dll):
        sys.exit(f"DLL not found: {args.dll} (pass --dll or set TOBII_DLL)")
    sys.exit(args.run(args))


if __name__ == "__main__":
    main()
