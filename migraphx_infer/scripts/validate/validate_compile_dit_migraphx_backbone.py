#!/usr/bin/env python3
"""Validate torch.compile DiT + device-pointer MIGraphX Backbone on real data."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts/deployment")
from benchmark_inference import prepare_model_inputs, set_seed
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from migraphx_backbone_wrapper import replace_backbone_with_migraphx


MODEL = "nvidia/GR00T-N1.6-3B"
DATASET = "demo_data/gr1.PickNPlace"
PROMPT = "pick the pear from the counter and place it in the plate"
VISION_ENGINE = os.environ.get(
    "MIGRAPHX_VISION_ENGINE",
    "migraphx_infer/artifacts/backbone/vision_projector_fp16_async.mxr",
)
LANGUAGE_ENGINE = os.environ.get(
    "MIGRAPHX_LANGUAGE_ENGINE",
    "migraphx_infer/artifacts/backbone/language_transformer_fp16_async.mxr",
)


def flatten_tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, np.ndarray):
        return [torch.from_numpy(value)]
    if isinstance(value, dict) or hasattr(value, "items"):
        result = []
        for _, item in sorted(value.items()):
            result.extend(flatten_tensors(item))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(flatten_tensors(item))
        return result
    return []


def report(name, reference, candidate):
    reference = reference.detach().float().cpu()
    candidate = candidate.detach().float().cpu()
    diff = candidate - reference
    cosine = torch.nn.functional.cosine_similarity(
        reference.reshape(-1), candidate.reshape(-1), dim=0
    )
    print(name, "shape=", tuple(reference.shape), flush=True)
    print("  reference_nan_inf:", int(torch.isnan(reference).sum()), int(torch.isinf(reference).sum()))
    print("  candidate_nan_inf:", int(torch.isnan(candidate).sum()), int(torch.isinf(candidate).sum()))
    print("  mae:", float(diff.abs().mean()))
    print("  rmse:", float(diff.square().mean().sqrt()))
    print("  max_abs:", float(diff.abs().max()))
    print("  cosine:", float(cosine))


def main():
    set_seed(42)
    policy = Gr00tPolicy(
        model_path=MODEL,
        embodiment_tag=EmbodimentTag("gr1"),
        device="cuda",
        strict=True,
    )
    config = policy.get_modality_config()
    dataset = LeRobotEpisodeLoader(
        dataset_path=DATASET,
        modality_configs=config,
        video_backend="decord",
    )
    step = extract_step_data(
        dataset[0],
        step_index=0,
        modality_configs=config,
        embodiment_tag=EmbodimentTag("gr1"),
        allow_padding=False,
    )
    observation = {
        "video": {key: np.stack(step.images[key])[None] for key in step.images},
        "state": {key: step.states[key][None] for key in step.states},
        "language": {config["language"].modality_keys[0]: [[PROMPT]]},
    }
    collated = prepare_model_inputs(policy, observation)
    backbone_inputs, action_inputs = policy.model.prepare_input(collated)

    with torch.inference_mode():
        reference_backbone = policy.model.backbone(backbone_inputs)
        set_seed(20260717)
        reference_action = policy.model.action_head.get_action(
            reference_backbone, action_inputs
        )
    torch.cuda.synchronize()

    policy.model.action_head.model.forward = torch.compile(
        policy.model.action_head.model.forward,
        mode="max-autotune",
    )
    replace_backbone_with_migraphx(policy, VISION_ENGINE, LANGUAGE_ENGINE)
    with torch.inference_mode():
        hybrid_backbone = policy.model.backbone(backbone_inputs)
        set_seed(20260717)
        hybrid_action = policy.model.action_head.get_action(
            hybrid_backbone, action_inputs
        )
    torch.cuda.synchronize()

    report(
        "backbone_features",
        reference_backbone["backbone_features"],
        hybrid_backbone["backbone_features"],
    )
    reference_tensors = flatten_tensors(reference_action)
    hybrid_tensors = flatten_tensors(hybrid_action)
    if len(reference_tensors) != len(hybrid_tensors):
        raise RuntimeError(
            f"Action tensor count mismatch: {len(reference_tensors)} vs {len(hybrid_tensors)}"
        )
    for index, (reference, candidate) in enumerate(zip(reference_tensors, hybrid_tensors)):
        report(f"action[{index}]", reference, candidate)


if __name__ == "__main__":
    main()
