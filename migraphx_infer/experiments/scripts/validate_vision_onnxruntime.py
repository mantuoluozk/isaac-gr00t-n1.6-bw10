#!/usr/bin/env python3
import argparse
import time

import numpy as np
import onnxruntime as ort
import torch


def metrics(reference: torch.Tensor, candidate: torch.Tensor):
    ref = reference.float().reshape(-1)
    got = candidate.float().reshape(-1)
    diff = got - ref
    return {
        "mae": diff.abs().mean().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "max_abs": diff.abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(ref, got, dim=0).item(),
        "nan": torch.isnan(got).sum().item(),
        "inf": torch.isinf(got).sum().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()
    saved = torch.load(args.reference, map_location="cpu", weights_only=False)
    pixels = saved["pixel_values"].to(dtype=torch.float16).numpy()
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    started = time.perf_counter()
    session = ort.InferenceSession(args.onnx, sess_options=options, providers=["CPUExecutionProvider"])
    print(f"Session load: {time.perf_counter() - started:.2f} s", flush=True)
    started = time.perf_counter()
    output = session.run(None, {"pixel_values": pixels})[0]
    print(f"Inference: {time.perf_counter() - started:.2f} s", flush=True)
    output_tensor = torch.from_numpy(np.asarray(output))
    print("vs FP16 PyTorch:", metrics(saved["reference_fp16"], output_tensor), flush=True)


if __name__ == "__main__":
    main()
