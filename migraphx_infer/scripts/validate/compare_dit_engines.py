#!/usr/bin/env python3
"""Compare two zero-copy DiT MXRs with identical FP16 device inputs."""

from __future__ import annotations

import argparse

import migraphx
import torch


def run(engine_path: str, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    model = migraphx.load(engine_path, format="msgpack")
    if model.get_offload_copy():
        raise RuntimeError(f"Expected offload_copy=False: {engine_path}")
    shapes = model.get_parameter_shapes()
    output = torch.empty(
        tuple(shapes["main:#output_0"].lens()), device="cuda", dtype=torch.float16
    )
    params = {
        name: migraphx.argument_from_pointer(shapes[name], tensor.data_ptr())
        for name, tensor in tensors.items()
    }
    params["main:#output_0"] = migraphx.argument_from_pointer(
        shapes["main:#output_0"], output.data_ptr()
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
    args = parser.parse_args()

    generator = torch.Generator(device="cuda").manual_seed(20260716)
    tensors = {
        "sa_embs": torch.randn(
            (1, 51, 1536), generator=generator, device="cuda", dtype=torch.float16
        ),
        "vl_embs": torch.randn(
            (1, 116, 2048), generator=generator, device="cuda", dtype=torch.float16
        ),
        "timestep": torch.ones((1,), device="cuda", dtype=torch.int64),
        "image_mask": torch.ones((1, 116), device="cuda", dtype=torch.bool),
        "backbone_attention_mask": torch.ones(
            (1, 116), device="cuda", dtype=torch.bool
        ),
    }
    baseline = run(args.baseline, tensors).float()
    candidate = run(args.candidate, tensors).float()
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
