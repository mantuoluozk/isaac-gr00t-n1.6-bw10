#!/usr/bin/env python3
"""Fuse the exported SigLIP2 attention core into com.microsoft MHA nodes.

This is intentionally strict: it only writes an output when every Softmax
matches the expected self-attention topology and the requested layer count.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--expected-layers", type=int, default=27)
    parser.add_argument(
        "--mha-input-mode",
        choices=("prebiased", "packed-bias"),
        default="prebiased",
    )
    return parser.parse_args()


def only_consumer(consumers, tensor_name: str, op_type: str):
    matches = [node for node in consumers.get(tensor_name, []) if node.op_type == op_type]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {op_type} consumer of {tensor_name}, got {len(matches)}"
        )
    return matches[0]


def parent_of_type(producer, node, op_type: str):
    matches = [producer[name] for name in node.input if name in producer and producer[name].op_type == op_type]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {op_type} parent of {node.name}, got {len(matches)}")
    return matches[0]


def main() -> None:
    args = parse_args()
    model = onnx.load(args.input, load_external_data=True)
    graph = model.graph
    original_nodes = list(graph.node)
    producer = {output: node for node in original_nodes for output in node.output}
    initializers = {item.name: item for item in graph.initializer}
    consumers = {}
    for node in original_nodes:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    softmax_nodes = [node for node in original_nodes if node.op_type == "Softmax"]
    if len(softmax_nodes) != args.expected_layers:
        raise RuntimeError(
            f"Expected {args.expected_layers} Softmax nodes, got {len(softmax_nodes)}"
        )

    remove_names: set[str] = set()
    replacements = {}
    for index, softmax in enumerate(softmax_nodes):
        qk = producer[softmax.input[0]]
        if qk.op_type != "MatMul":
            raise RuntimeError(f"Softmax parent is {qk.op_type}, expected MatMul: {softmax.name}")

        q_mul = producer[qk.input[0]]
        k_mul = producer[qk.input[1]]
        if q_mul.op_type != "Mul" or k_mul.op_type != "Mul":
            raise RuntimeError(f"Q/K scaling pattern mismatch at {softmax.name}")

        q_transpose = parent_of_type(producer, q_mul, "Transpose")
        k_transpose = parent_of_type(producer, k_mul, "Transpose")
        q_reshape = producer[q_transpose.input[0]]
        k_reshape = producer[k_transpose.input[0]]
        q_add = producer[q_reshape.input[0]]
        k_add = producer[k_reshape.input[0]]
        if (q_reshape.op_type, k_reshape.op_type, q_add.op_type, k_add.op_type) != (
            "Reshape", "Reshape", "Add", "Add"
        ):
            raise RuntimeError(f"Q/K projection pattern mismatch at {softmax.name}")

        context = only_consumer(consumers, softmax.output[0], "MatMul")
        value_input = context.input[1] if context.input[0] == softmax.output[0] else context.input[0]
        v_transpose = producer[value_input]
        v_reshape = producer[v_transpose.input[0]]
        v_add = producer[v_reshape.input[0]]
        if (v_transpose.op_type, v_reshape.op_type, v_add.op_type) != (
            "Transpose", "Reshape", "Add"
        ):
            raise RuntimeError(f"V projection pattern mismatch at {softmax.name}")

        out_transpose = only_consumer(consumers, context.output[0], "Transpose")
        out_reshape = only_consumer(consumers, out_transpose.output[0], "Reshape")

        if args.mha_input_mode == "packed-bias":
            q_matmul = parent_of_type(producer, q_add, "MatMul")
            k_matmul = parent_of_type(producer, k_add, "MatMul")
            v_matmul = parent_of_type(producer, v_add, "MatMul")

            def get_bias(add_node):
                names = [name for name in add_node.input if name in initializers]
                if len(names) != 1:
                    raise RuntimeError(f"Expected one bias initializer for {add_node.name}")
                return numpy_helper.to_array(initializers[names[0]])

            packed_bias = np.concatenate(
                [get_bias(q_add), get_bias(k_add), get_bias(v_add)], axis=0
            )
            bias_name = f"SigLip2MultiHeadAttention_{index}_qkv_bias"
            graph.initializer.append(numpy_helper.from_array(packed_bias, bias_name))
            mha_inputs = [
                q_matmul.output[0],
                k_matmul.output[0],
                v_matmul.output[0],
                bias_name,
            ]
        else:
            mha_inputs = [q_add.output[0], k_add.output[0], v_add.output[0]]

        mha = helper.make_node(
            "MultiHeadAttention",
            inputs=mha_inputs,
            outputs=[out_reshape.output[0]],
            name=f"SigLip2MultiHeadAttention_{index}",
            domain="com.microsoft",
            num_heads=args.num_heads,
        )
        replacements[out_reshape.name] = mha
        for node in (
            q_mul,
            k_mul,
            q_transpose,
            k_transpose,
            q_reshape,
            k_reshape,
            qk,
            softmax,
            v_transpose,
            v_reshape,
            context,
            out_transpose,
            out_reshape,
        ):
            remove_names.add(node.name)

    rewritten = []
    for node in original_nodes:
        replacement = replacements.get(node.name)
        if replacement is not None:
            rewritten.append(replacement)
        elif node.name not in remove_names:
            rewritten.append(node)

    # Remove constants and shape helpers that became unreachable after fusion.
    rewritten_producer = {output: i for i, node in enumerate(rewritten) for output in node.output}
    needed: set[int] = set()

    def mark_tensor(tensor_name: str) -> None:
        node_index = rewritten_producer.get(tensor_name)
        if node_index is None or node_index in needed:
            return
        needed.add(node_index)
        for input_name in rewritten[node_index].input:
            mark_tensor(input_name)

    for output in graph.output:
        mark_tensor(output.name)

    del graph.node[:]
    graph.node.extend(node for i, node in enumerate(rewritten) if i in needed)

    if not any(opset.domain == "com.microsoft" for opset in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))

    onnx.checker.check_model(model)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    data_name = os.path.basename(args.output) + ".data"
    onnx.save_model(
        model,
        args.output,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )
    print(f"Fused {len(softmax_nodes)} attention layers")
    print(f"Nodes: {len(original_nodes)} -> {len(graph.node)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
