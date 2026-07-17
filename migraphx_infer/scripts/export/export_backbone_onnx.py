#!/usr/bin/env python3
"""Export static GR00T N1.6 Backbone components for MIGraphX.

The first supported component is the Eagle SigLIP2 vision path.  Its ONNX
boundary includes the ViT, the static 2x pixel-unshuffle and Eagle's mlp1
projector, so the output can be inserted directly into the language tokens.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "scripts/deployment")
from benchmark_inference import prepare_model_inputs
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


LOG = logging.getLogger("export_backbone_onnx")


class StaticVisionProjector(torch.nn.Module):
    """SigLIP2 + static pixel-unshuffle + Eagle projector."""

    def __init__(self, vision_model: torch.nn.Module, projector: torch.nn.Module):
        super().__init__()
        self.vision_model = vision_model
        self.projector = projector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # SigLIP2 expects a list because it supports differently sized images.
        # This deployment engine deliberately fixes one 252x252 image.
        output = self.vision_model(
            pixel_values=[pixel_values],
            output_hidden_states=False,
        )
        hidden = output.last_hidden_state

        # 252 / patch14 = 18. Eagle downsamples the 18x18 token grid by 2,
        # giving 9x9=81 visual tokens with 4*1152 channels.
        batch, tokens, channels = hidden.shape
        if tokens != 324:
            raise RuntimeError(f"Expected 324 ViT tokens, got {tokens}")
        hidden = hidden.reshape(batch, 18, 18, channels).permute(0, 3, 1, 2)
        hidden = F.pixel_unshuffle(hidden, downscale_factor=2)
        hidden = hidden.flatten(start_dim=2).transpose(1, 2)
        hidden = self.projector(hidden)
        return hidden.reshape(81, 2048)


class StaticLanguageTransformer(torch.nn.Module):
    """Qwen3 transformer body without the unused vocabulary LM head."""

    def __init__(self, language_body: torch.nn.Module):
        super().__init__()
        self.language_body = language_body

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        output = self.language_body(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return output.last_hidden_state


def prepare_real_inputs(args: argparse.Namespace, policy: Gr00tPolicy):
    config = policy.get_modality_config()
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=config,
        video_backend=args.video_backend,
    )
    episode = dataset[0]
    step = extract_step_data(
        episode,
        step_index=0,
        modality_configs=config,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        allow_padding=False,
    )
    language = args.language_prompt if args.language_prompt else step.text
    observation = {
        "video": {key: np.stack(step.images[key])[None] for key in step.images},
        "state": {key: step.states[key][None] for key in step.states},
        "language": {config["language"].modality_keys[0]: [[language]]},
    }
    collated = prepare_model_inputs(policy, observation)
    backbone_inputs, _ = policy.model.prepare_input(collated)
    LOG.info("Dataset language: %r", step.text)
    LOG.info("Selected language: %r", language)
    LOG.info("input_ids: %s", tuple(backbone_inputs["input_ids"].shape))
    if backbone_inputs["input_ids"].shape[1] != args.expected_seq_len:
        raise RuntimeError(
            f"Expected language length {args.expected_seq_len}, got "
            f"{backbone_inputs['input_ids'].shape[1]}"
        )
    pixels = backbone_inputs["pixel_values"]
    if not isinstance(pixels, (list, tuple)) or len(pixels) != 1:
        raise RuntimeError(f"Expected one pixel tensor, got {type(pixels)} len={len(pixels)}")
    LOG.info("pixel_values: %s %s", tuple(pixels[0].shape), pixels[0].dtype)
    return backbone_inputs


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float().reshape(-1)
    got = candidate.float().reshape(-1)
    diff = got - ref
    return {
        "mae": diff.abs().mean().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "max_abs": diff.abs().max().item(),
        "cosine": F.cosine_similarity(ref, got, dim=0).item(),
    }


def export_vision(args: argparse.Namespace, policy: Gr00tPolicy, inputs) -> None:
    eagle = policy.model.backbone.model
    eagle.vision_model.config._attn_implementation = "sdpa"
    eagle.vision_model.eval()
    eagle.mlp1.eval()
    pixels_bf16 = inputs["pixel_values"][0].contiguous()

    with torch.inference_mode():
        reference_bf16 = eagle.extract_feature([pixels_bf16]).reshape(81, 2048)

    wrapper = StaticVisionProjector(eagle.vision_model, eagle.mlp1).eval().half()

    # The deployment input is fixed at 252x252, hence the positional grid is
    # always 18x18.  Precompute the antialiased interpolation once.  Besides
    # removing an ONNX-unsupported aten::_upsample_bilinear2d_aa node, this
    # avoids recomputing an invariant tensor on every frame.
    embeddings = wrapper.vision_model.vision_model.embeddings
    with torch.inference_mode():
        positional_grid = embeddings.position_embedding.weight.reshape(
            embeddings.position_embedding_size,
            embeddings.position_embedding_size,
            -1,
        )
        spatial_shapes = torch.tensor([[18, 18]], device=positional_grid.device)
        static_position_embeddings = embeddings.resize_positional_embeddings(
            positional_grid, spatial_shapes
        )

    def fixed_position_embeddings(_positional_embeddings, _spatial_shapes):
        return static_position_embeddings

    embeddings.resize_positional_embeddings = fixed_position_embeddings
    pixels_fp16 = pixels_bf16.half().contiguous()
    with torch.inference_mode():
        reference_fp16 = wrapper(pixels_fp16)

    accuracy = metrics(reference_bf16, reference_fp16)
    LOG.info("Vision BF16->FP16 metrics: %s", accuracy)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "vision_projector_fp16.onnx"
    reference_path = output_dir / "vision_reference.pt"
    torch.save(
        {
            "pixel_values": pixels_fp16.detach().cpu(),
            "reference_bf16": reference_bf16.detach().cpu(),
            "reference_fp16": reference_fp16.detach().cpu(),
            "metrics": accuracy,
        },
        reference_path,
    )

    LOG.info("Exporting %s", onnx_path)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (pixels_fp16,),
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["vision_features"],
            opset_version=19,
            do_constant_folding=True,
            export_params=True,
            # SigLIP2 converts static spatial-shape tensors to Python lists
            # while arranging windows and resizing position embeddings.  The
            # legacy tracer correctly freezes that control flow for the fixed
            # 252x252 deployment profile; torch.export treats it as a dynamic
            # data-dependent shape and rejects it before ONNX translation.
            dynamo=False,
        )

    import onnx

    onnx.checker.check_model(str(onnx_path))
    LOG.info("ONNX validation succeeded")
    LOG.info("Reference saved to %s", reference_path)


def export_language(args: argparse.Namespace, policy: Gr00tPolicy, inputs) -> None:
    eagle = policy.model.backbone.model
    eagle.vision_model.config._attn_implementation = "sdpa"
    eagle.language_model.config._attn_implementation = "sdpa"
    eagle.eval()

    input_ids = inputs["input_ids"].contiguous()
    attention_mask = inputs["attention_mask"].to(torch.int64).contiguous()
    position_ids = torch.arange(
        input_ids.shape[1], device=input_ids.device, dtype=torch.int64
    ).unsqueeze(0)

    with torch.inference_mode():
        vision_bf16 = eagle.extract_feature(inputs["pixel_values"]).reshape(81, 2048)
        inputs_embeds_bf16 = eagle.language_model.get_input_embeddings()(input_ids)
        flat_embeds = inputs_embeds_bf16.reshape(-1, inputs_embeds_bf16.shape[-1]).clone()
        image_tokens = input_ids.reshape(-1) == eagle.image_token_index
        if image_tokens.sum().item() != 81:
            raise RuntimeError(
                f"Expected 81 image tokens, got {image_tokens.sum().item()}"
            )
        flat_embeds[image_tokens] = vision_bf16
        inputs_embeds_bf16 = flat_embeds.reshape(1, args.expected_seq_len, 2048)

        # This is the exact public Backbone path and therefore includes the
        # Qwen vocabulary projection, although only hidden_states is consumed.
        full_output = eagle.language_model(
            inputs_embeds=inputs_embeds_bf16,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        reference_bf16 = full_output.hidden_states[-1]

    wrapper = StaticLanguageTransformer(eagle.language_model.model).eval().half()
    inputs_embeds_fp16 = inputs_embeds_bf16.half().contiguous()
    with torch.inference_mode():
        reference_fp16 = wrapper(inputs_embeds_fp16, attention_mask, position_ids)

    accuracy = metrics(reference_bf16, reference_fp16)
    LOG.info("Language BF16->FP16 metrics: %s", accuracy)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "language_transformer_fp16.onnx"
    reference_path = output_dir / "language_reference.pt"
    torch.save(
        {
            "inputs_embeds": inputs_embeds_fp16.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "position_ids": position_ids.detach().cpu(),
            "reference_bf16": reference_bf16.detach().cpu(),
            "reference_fp16": reference_fp16.detach().cpu(),
            "metrics": accuracy,
        },
        reference_path,
    )

    LOG.info("Exporting %s", onnx_path)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (inputs_embeds_fp16, attention_mask, position_ids),
            str(onnx_path),
            input_names=["inputs_embeds", "attention_mask", "position_ids"],
            output_names=["backbone_features"],
            opset_version=19,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )

    import onnx

    onnx.checker.check_model(str(onnx_path))
    LOG.info("ONNX validation succeeded")
    LOG.info("Reference saved to %s", reference_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="nvidia/GR00T-N1.6-3B")
    parser.add_argument("--dataset_path", default="demo_data/gr1.PickNPlace")
    parser.add_argument("--embodiment_tag", default="gr1")
    parser.add_argument("--video_backend", default="decord")
    parser.add_argument("--expected_seq_len", type=int, default=116)
    parser.add_argument(
        "--language_prompt",
        default="pick the pear from the counter and place it in the plate",
        help="Fix the deployment prompt so the static language length is reproducible.",
    )
    parser.add_argument("--component", choices=["vision", "language"], default="vision")
    parser.add_argument("--output_dir", default="migraphx_infer/artifacts/backbone")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(42)
    policy = Gr00tPolicy(
        model_path=args.model_path,
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        device="cuda",
        strict=True,
    )
    inputs = prepare_real_inputs(args, policy)
    if args.component == "vision":
        export_vision(args, policy, inputs)
    else:
        export_language(args, policy, inputs)


if __name__ == "__main__":
    main()
