#!/usr/bin/env python3
"""Print an operation histogram for a serialized MIGraphX program."""

from __future__ import annotations

import argparse
from collections import Counter
import re

import migraphx


def operation_name(instruction) -> str:
    """Handle the Python API variants shipped by different DTK releases."""
    if hasattr(instruction, "name"):
        value = instruction.name()
        if value:
            return str(value)
    if hasattr(instruction, "get_operator"):
        operator = instruction.get_operator()
        if hasattr(operator, "name"):
            return str(operator.name())
        return str(operator)
    return str(instruction).split("(", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine")
    parser.add_argument("--contains")
    args = parser.parse_args()

    program = migraphx.load(args.engine, format="msgpack")
    module = program.get_main_module()
    if not hasattr(module, "get_instructions"):
        text = str(program)
        if args.contains:
            for line in text.splitlines():
                if args.contains in line:
                    print(line)
            return
        counts = Counter()
        kernel_counts = Counter()
        gemm_counts = Counter()
        for line in text.splitlines():
            match = re.match(r"^\S+\s+=\s+([^\[(\s]+)", line)
            if match:
                counts[match.group(1)] += 1
            kernel = re.search(r"symbol_name=([^,\]]+)", line)
            if kernel:
                kernel_counts[kernel.group(1)] += 1
            if " = gpu::gemm[" in line:
                beta = re.search(r"beta=([^,\]]+)", line)
                gemm_counts[f"beta={beta.group(1) if beta else '?'}"] += 1
        print("engine:", args.engine)
        print("instruction_count:", sum(counts.values()))
        for name, count in counts.most_common():
            print(f"{count:5d}  {name}")
        print("gemm_variants:")
        for name, count in gemm_counts.most_common():
            print(f"{count:5d}  {name}")
        print("code_object_kernels:")
        for name, count in kernel_counts.most_common():
            print(f"{count:5d}  {name}")
        return
    instructions = module.get_instructions()
    counts = Counter(operation_name(ins) for ins in instructions)

    print("engine:", args.engine)
    print("instruction_count:", sum(counts.values()))
    for name, count in counts.most_common():
        print(f"{count:5d}  {name}")


if __name__ == "__main__":
    main()
