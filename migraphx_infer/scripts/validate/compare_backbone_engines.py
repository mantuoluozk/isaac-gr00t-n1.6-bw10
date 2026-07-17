#!/usr/bin/env python3
"""Compare two zero-copy Backbone MXRs with the same saved device inputs."""

from __future__ import annotations

import argparse

import migraphx
import torch


TORCH_DTYPES = {
    "half_type": torch.float16,
    "float_type": torch.float32,
    "int64_type": torch.int64,
    "bool_type": torch.bool,
}


def run(engine_path: str, saved: dict[str, torch.Tensor]) -> torch.Tensor:
    model = migraphx.load(engine_path, format="msgpack")
    if model.get_offload_copy():
        raise RuntimeError(f"Expected offload_copy=False: {engine_path}")
    shapes = model.get_parameter_shapes()
    output_name = "main:#output_0"
    tensors = {}
    for name, shape in shapes.items():
        if name == output_name:
            continue
        dtype = TORCH_DTYPES[shape.type_string()]
        tensors[name] = saved[name].to(device="cuda", dtype=dtype).contiguous()

    output_shape = shapes[output_name]
    output = torch.empty(
        tuple(output_shape.lens()),
        device="cuda",
        dtype=TORCH_DTYPES[output_shape.type_string()],
    )
    params = {
        name: migraphx.argument_from_pointer(shapes[name], tensor.data_ptr())
        for name, tensor in tensors.items()
    }
    params[output_name] = migraphx.argument_from_pointer(
        output_shape, output.data_ptr()
    )
    stream = torch.cuda.current_stream()
    results = model.run_async(params, stream.cuda_stream, "ihipStream_t")
    stream.synchronize()
    assert results is not None
    return output.clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()

    saved = torch.load(args.reference, map_location="cpu", weights_only=False)
    baseline = run(args.baseline, saved).float()
    candidate = run(args.candidate, saved).float()
    diff = candidate - baseline
    print("baseline NaN/Inf:", int(torch.isnan(baseline).sum()), int(torch.isinf(baseline).sum()))
    print("candidate NaN/Inf:", int(torch.isnan(candidate).sum()), int(torch.isinf(candidate).sum()))
    print("MAE:", float(diff.abs().mean()))
    print("RMSE:", float(diff.square().mean().sqrt()))
    print("Max abs:", float(diff.abs().max()))
    print(
        "Cosine:",
        float(torch.nn.functional.cosine_similarity(baseline.flatten(), candidate.flatten(), dim=0)),
    )


if __name__ == "__main__":
    main()
