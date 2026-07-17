import sys

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

sys.path.insert(0, "scripts/deployment")

from benchmark_inference import prepare_model_inputs, set_seed
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


MODEL = "nvidia/GR00T-N1.6-3B"
DATASET = "demo_data/gr1.PickNPlace"


def clone_structure(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, BatchFeature):
        return BatchFeature(data={key: clone_structure(item) for key, item in value.items()})
    if isinstance(value, dict):
        return {key: clone_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_structure(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_structure(item) for item in value)
    return value


def report(name, reference, candidate):
    reference = reference.detach().float().cpu()
    candidate = candidate.detach().float().cpu()
    diff = reference - candidate
    cosine = torch.nn.functional.cosine_similarity(
        reference.flatten(), candidate.flatten(), dim=0
    )
    print(name, "shape=", tuple(reference.shape))
    print("  nan_inf:", int(torch.isnan(candidate).sum()), int(torch.isinf(candidate).sum()))
    print("  max_abs_error:", float(diff.abs().max()))
    print("  mean_abs_error:", float(diff.abs().mean()))
    print("  rmse:", float(torch.sqrt(torch.mean(diff.square()))))
    print("  cosine_similarity:", float(cosine))


def run_action(policy, backbone_outputs, action_inputs):
    set_seed(20260714)
    with torch.inference_mode():
        result = policy.model.action_head.get_action(
            clone_structure(backbone_outputs), clone_structure(action_inputs)
        )
    torch.cuda.synchronize()
    return result["action_pred"]


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
        candidate = policy.model.backbone(backbone_inputs)
    if int(candidate["backbone_features"].shape[1]) == 116:
        selected = (traj_idx, backbone_inputs, action_inputs, candidate)
        break

if selected is None:
    raise RuntimeError("No 116-token sample found")

traj_idx, backbone_inputs, action_inputs, flash_outputs = selected
flash_outputs = clone_structure(flash_outputs)
print("trajectory:", traj_idx)

original_forward = policy.model.backbone.forward
policy.model.backbone.forward = torch.compile(
    original_forward,
    mode="max-autotune-no-cudagraphs",
)
with torch.inference_mode():
    for _ in range(8):
        compiled_flash_outputs = policy.model.backbone(backbone_inputs)
    compiled_flash_outputs = clone_structure(policy.model.backbone(backbone_inputs))
torch.cuda.synchronize()
policy.model.backbone.forward = original_forward
torch._dynamo.reset()

eagle = policy.model.backbone.model
eagle.vision_model.config._attn_implementation = "sdpa"
eagle.language_model.config._attn_implementation = "sdpa"
with torch.inference_mode():
    sdpa_outputs = clone_structure(policy.model.backbone(backbone_inputs))
torch.cuda.synchronize()

policy.model.backbone.forward = torch.compile(
    original_forward,
    mode="max-autotune-no-cudagraphs",
)
with torch.inference_mode():
    for _ in range(8):
        compiled_outputs = policy.model.backbone(backbone_inputs)
    compiled_outputs = clone_structure(policy.model.backbone(backbone_inputs))
    compiled_outputs_repeat = clone_structure(policy.model.backbone(backbone_inputs))
torch.cuda.synchronize()

report(
    "backbone flash-vs-sdpa",
    flash_outputs["backbone_features"],
    sdpa_outputs["backbone_features"],
)
report(
    "backbone sdpa-vs-compiled",
    sdpa_outputs["backbone_features"],
    compiled_outputs["backbone_features"],
)
report(
    "backbone compiled-repeat",
    compiled_outputs["backbone_features"],
    compiled_outputs_repeat["backbone_features"],
)
report(
    "backbone compiled-flash-vs-compiled-sdpa",
    compiled_flash_outputs["backbone_features"],
    compiled_outputs["backbone_features"],
)
print(
    "attention_mask_equal:",
    torch.equal(flash_outputs["backbone_attention_mask"], sdpa_outputs["backbone_attention_mask"]),
)
print("image_mask_equal:", torch.equal(flash_outputs["image_mask"], sdpa_outputs["image_mask"]))

flash_action = run_action(policy, flash_outputs, action_inputs)
sdpa_action = run_action(policy, sdpa_outputs, action_inputs)
compiled_action = run_action(policy, compiled_outputs, action_inputs)
compiled_flash_action = run_action(policy, compiled_flash_outputs, action_inputs)
report("action flash-vs-sdpa", flash_action, sdpa_action)
report("action sdpa-vs-compiled", sdpa_action, compiled_action)
report("action compiled-flash-vs-compiled-sdpa", compiled_flash_action, compiled_action)
