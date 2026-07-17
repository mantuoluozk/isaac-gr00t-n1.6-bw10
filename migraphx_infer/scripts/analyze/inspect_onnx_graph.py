#!/usr/bin/env python3
import argparse
from collections import Counter

import onnx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--show-first-softmax", action="store_true")
    args = parser.parse_args()

    for path in args.models:
        print(f"MODEL: {path}")
        try:
            model = onnx.load(path, load_external_data=False)
            onnx.checker.check_model(path)
        except Exception as exc:
            print(f"CHECK: FAILED: {type(exc).__name__}: {exc}")
            continue
        counts = Counter(
            f"{node.domain or 'ai.onnx'}::{node.op_type}" for node in model.graph.node
        )
        print("CHECK: OK")
        print(f"NODES: {sum(counts.values())}")
        for name, count in counts.most_common():
            if "Attention" in name or name.endswith("::Softmax"):
                print(f"  {name}: {count}")
        if args.show_first_softmax:
            show_first_softmax(model)


def show_first_softmax(model: onnx.ModelProto) -> None:
    graph = model.graph
    producer = {output: node for node in graph.node for output in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    shapes: dict[str, str] = {}
    for value in [*graph.input, *graph.output, *graph.value_info]:
        tensor = value.type.tensor_type
        dims = []
        for dim in tensor.shape.dim:
            dims.append(str(dim.dim_value or dim.dim_param or "?"))
        shapes[value.name] = "[" + ",".join(dims) + "]"

    softmax = next((node for node in graph.node if node.op_type == "Softmax"), None)
    if softmax is None:
        return
    print(f"FIRST_SOFTMAX: {softmax.name}")

    seen: set[str] = set()

    def walk_up(tensor_name: str, depth: int) -> None:
        if depth < 0 or tensor_name in seen:
            return
        seen.add(tensor_name)
        node = producer.get(tensor_name)
        indent = "  " * (4 - depth)
        if node is None:
            print(f"{indent}INPUT {tensor_name} {shapes.get(tensor_name, '')}")
            return
        print(f"{indent}<- {node.op_type} {node.name} => {tensor_name} {shapes.get(tensor_name, '')}")
        for name in node.input:
            walk_up(name, depth - 1)

    def walk_down(tensor_name: str, depth: int) -> None:
        if depth < 0:
            return
        for node in consumers.get(tensor_name, []):
            indent = "  " * (4 - depth)
            outputs = ", ".join(f"{name}{shapes.get(name, '')}" for name in node.output)
            print(f"{indent}-> {node.op_type} {node.name} => {outputs}")
            for name in node.output:
                walk_down(name, depth - 1)

    walk_up(softmax.input[0], 4)
    print("  SOFTMAX_OUTPUT")
    walk_down(softmax.output[0], 4)


if __name__ == "__main__":
    main()
