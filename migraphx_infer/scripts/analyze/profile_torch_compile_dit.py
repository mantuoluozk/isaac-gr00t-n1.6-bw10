#!/usr/bin/env python3
"""Standalone torch.compile DiT benchmark with the deployment-static shapes."""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


def main() -> None:
    warmup = int(os.environ.get("DIT_WARMUP", "5"))
    iterations = int(os.environ.get("DIT_ITERATIONS", "10"))
    policy = Gr00tPolicy(
        model_path=os.environ.get("MODEL_PATH", "nvidia/GR00T-N1.6-3B"),
        embodiment_tag=EmbodimentTag(os.environ.get("EMBODIMENT_TAG", "gr1")),
        device="cuda",
        strict=True,
    )
    model = policy.model.action_head.model.eval()
    dtype = next(model.parameters()).dtype
    compiled_forward = torch.compile(model.forward, mode="max-autotune")

    generator = torch.Generator(device="cuda").manual_seed(20260716)
    inputs = {
        "hidden_states": torch.randn(
            (1, 51, 1536), device="cuda", dtype=dtype, generator=generator
        ),
        "encoder_hidden_states": torch.randn(
            (1, 116, 2048), device="cuda", dtype=dtype, generator=generator
        ),
        "timestep": torch.ones((1,), device="cuda", dtype=torch.int64),
        "image_mask": torch.ones((1, 116), device="cuda", dtype=torch.bool),
        "backbone_attention_mask": torch.ones(
            (1, 116), device="cuda", dtype=torch.bool
        ),
    }

    last_output = None
    with torch.inference_mode():
        for _ in range(warmup):
            last_output = compiled_forward(**inputs)
        torch.cuda.synchronize()

        elapsed = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            last_output = compiled_forward(**inputs)
            end.record()
            end.synchronize()
            elapsed.append(start.elapsed_time(end))

    assert last_output is not None
    output = last_output.float()
    values = np.asarray(elapsed)
    print("dtype:", dtype)
    print("output shape:", tuple(output.shape))
    print("NaN count:", int(torch.isnan(output).sum().item()))
    print("Inf count:", int(torch.isinf(output).sum().item()))
    print("warmup:", warmup)
    print("iterations:", iterations)
    print("median_ms:", float(np.median(values)))
    print("mean_ms:", float(np.mean(values)))
    print("min_ms:", float(np.min(values)))
    print("max_ms:", float(np.max(values)))


if __name__ == "__main__":
    main()
