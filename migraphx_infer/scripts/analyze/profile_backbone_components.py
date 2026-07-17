from collections import defaultdict
from functools import wraps
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts/deployment")
from benchmark_inference import prepare_model_inputs
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


policy = Gr00tPolicy(
    model_path="nvidia/GR00T-N1.6-3B",
    embodiment_tag=EmbodimentTag("gr1"),
    device="cuda",
    strict=True,
)
config = policy.get_modality_config()
dataset = LeRobotEpisodeLoader(
    dataset_path="demo_data/gr1.PickNPlace",
    modality_configs=config,
    video_backend="decord",
)
episode = dataset[0]
step = extract_step_data(
    episode,
    step_index=0,
    modality_configs=config,
    embodiment_tag=EmbodimentTag("gr1"),
    allow_padding=False,
)
observation = {
    "video": {key: np.stack(step.images[key])[None] for key in step.images},
    "state": {key: step.states[key][None] for key in step.states},
    "language": {config["language"].modality_keys[0]: [[step.text]]},
}
collated = prepare_model_inputs(policy, observation)
backbone_inputs, _ = policy.model.prepare_input(collated)
print("text:", repr(step.text))
for name, value in backbone_inputs.items():
    if isinstance(value, torch.Tensor):
        print("input", name, tuple(value.shape), value.dtype)
    elif isinstance(value, (list, tuple)):
        details = [
            (tuple(item.shape), item.dtype) if isinstance(item, torch.Tensor) else type(item).__name__
            for item in value
        ]
        print("input", name, type(value).__name__, details)
    else:
        print("input", name, type(value).__name__, repr(value))

backbone = policy.model.backbone
for _ in range(5):
    with torch.inference_mode():
        backbone(backbone_inputs)
torch.cuda.synchronize()

events = defaultdict(list)


def instrument(name, module):
    original = module.forward

    @wraps(original)
    def wrapped(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = original(*args, **kwargs)
        end.record()
        events[name].append((start, end))
        return result

    module.forward = wrapped


instrument("vision_model", backbone.model.vision_model)
instrument("vision_projector_mlp1", backbone.model.mlp1)
instrument("language_model", backbone.model.language_model)

overall = []
for _ in range(20):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode():
        output = backbone(backbone_inputs)
    end.record()
    overall.append((start, end))
torch.cuda.synchronize()


def summarize(name, pairs):
    values = np.array([start.elapsed_time(end) for start, end in pairs])
    print(
        name,
        f"median={np.median(values):.3f} ms",
        f"mean={np.mean(values):.3f} ms",
        f"min={np.min(values):.3f} ms",
        f"max={np.max(values):.3f} ms",
    )


summarize("backbone_total", overall)
for name, pairs in events.items():
    summarize(name, pairs)
for name, value in output.items():
    if isinstance(value, torch.Tensor):
        print("output", name, tuple(value.shape), value.dtype)
