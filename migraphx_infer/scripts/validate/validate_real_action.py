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
from migraphx_dit_wrapper import replace_dit_with_migraphx


MODEL = "nvidia/GR00T-N1.6-3B"
DATASET = "demo_data/gr1.PickNPlace"
ENGINE = os.environ.get(
    "MIGRAPHX_ENGINE", "migraphx_infer/artifacts/dit/dit_fp16_vl116_async.mxr"
)


def flatten_tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, np.ndarray):
        return [torch.from_numpy(value)]
    if isinstance(value, dict) or hasattr(value, "items"):
        result = []
        for key, item in sorted(value.items()):
            result.extend(flatten_tensors(item))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(flatten_tensors(item))
        return result
    return []


def clone_structure(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: clone_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_structure(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_structure(item) for item in value)
    return value


policy = Gr00tPolicy(
    model_path=MODEL,
    embodiment_tag=EmbodimentTag("gr1"),
    device="cuda",
    strict=True,
)
modality_config = policy.get_modality_config()
dataset = LeRobotEpisodeLoader(
    dataset_path=DATASET,
    modality_configs=modality_config,
    video_backend="decord",
)
selected = None
for traj_idx in range(len(dataset)):
    episode_data = dataset[traj_idx]
    step_data = extract_step_data(
        episode_data,
        step_index=0,
        modality_configs=modality_config,
        embodiment_tag=EmbodimentTag("gr1"),
        allow_padding=False,
    )
    observation = {
        "video": {key: np.stack(step_data.images[key])[None] for key in step_data.images},
        "state": {key: step_data.states[key][None] for key in step_data.states},
        "language": {modality_config["language"].modality_keys[0]: [[step_data.text]]},
    }
    collated_inputs = prepare_model_inputs(policy, observation)
    with torch.inference_mode():
        backbone_inputs, action_inputs = policy.model.prepare_input(collated_inputs)
        backbone_outputs = policy.model.backbone(backbone_inputs)
    token_count = int(backbone_outputs["backbone_features"].shape[1])
    print(f"trajectory={traj_idx} text={step_data.text!r} tokens={token_count}")
    if token_count == 116:
        selected = (backbone_outputs, action_inputs)
        break

if selected is None:
    raise RuntimeError("No 116-token GR1 sample found for the static MXR")
backbone_outputs, action_inputs = selected

eager_backbone_outputs = clone_structure(backbone_outputs)
eager_action_inputs = clone_structure(action_inputs)
mx_backbone_outputs = clone_structure(backbone_outputs)
mx_action_inputs = clone_structure(action_inputs)

original_forward = policy.model.action_head.model.forward
replace_dit_with_migraphx(policy, ENGINE)
set_seed(20260714)
with torch.inference_mode():
    migraphx_action = policy.model.action_head.get_action(mx_backbone_outputs, mx_action_inputs)
torch.cuda.synchronize()

policy.model.action_head.model.forward = original_forward
set_seed(20260714)
with torch.inference_mode():
    eager_action = policy.model.action_head.get_action(eager_backbone_outputs, eager_action_inputs)
torch.cuda.synchronize()

eager_tensors = flatten_tensors(eager_action)
mx_tensors = flatten_tensors(migraphx_action)
print("eager_output_type:", type(eager_action).__name__)
print("migraphx_output_type:", type(migraphx_action).__name__)
print("tensor_count:", len(eager_tensors), len(mx_tensors))
if len(eager_tensors) != len(mx_tensors):
    raise RuntimeError(f"Output tensor count mismatch: {len(eager_tensors)} vs {len(mx_tensors)}")

for index, (eager, mx) in enumerate(zip(eager_tensors, mx_tensors)):
    eager = eager.detach().float().cpu()
    mx = mx.detach().float().cpu()
    diff = eager - mx
    cosine = torch.nn.functional.cosine_similarity(eager.flatten(), mx.flatten(), dim=0)
    print(f"tensor[{index}] shape={tuple(eager.shape)}")
    print("  eager_nan_inf:", int(torch.isnan(eager).sum()), int(torch.isinf(eager).sum()))
    print("  mx_nan_inf:", int(torch.isnan(mx).sum()), int(torch.isinf(mx).sum()))
    print("  max_abs_error:", float(diff.abs().max()))
    print("  mean_abs_error:", float(diff.abs().mean()))
    print("  rmse:", float(torch.sqrt(torch.mean(diff.square()))))
    print("  cosine_similarity:", float(cosine))
