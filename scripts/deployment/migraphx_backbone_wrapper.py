"""Device-pointer MIGraphX adapter for Eagle Vision and Language components."""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature


class _DeviceProgram:
    def __init__(self, engine_path: str):
        import migraphx

        self.migraphx = migraphx
        self.model = migraphx.load(engine_path, format="msgpack")
        if self.model.get_offload_copy():
            raise RuntimeError(f"Backbone engine must use offload_copy=False: {engine_path}")
        self.shapes = self.model.get_parameter_shapes()
        self.output_name = "main:#output_0"
        if self.output_name not in self.shapes:
            raise RuntimeError(f"Engine lacks preallocated output parameter: {engine_path}")
        self.output = None
        self.last_results = None

    def run(self, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(iter(tensors.values())).device
        for name, tensor in tensors.items():
            expected = tuple(self.shapes[name].lens())
            if tuple(tensor.shape) != expected:
                raise RuntimeError(
                    f"MIGraphX {name} shape mismatch: got {tuple(tensor.shape)}, expected {expected}"
                )
            if not tensor.is_contiguous():
                raise RuntimeError(f"MIGraphX input {name} must be contiguous")

        output_shape = tuple(self.shapes[self.output_name].lens())
        if self.output is None or self.output.device != device:
            self.output = torch.empty(output_shape, device=device, dtype=torch.float16)
        params = {
            name: self.migraphx.argument_from_pointer(self.shapes[name], tensor.data_ptr())
            for name, tensor in tensors.items()
        }
        params[self.output_name] = self.migraphx.argument_from_pointer(
            self.shapes[self.output_name], self.output.data_ptr()
        )
        stream = torch.cuda.current_stream(device=device)
        self.last_results = self.model.run_async(
            params, stream.cuda_stream, "ihipStream_t"
        )
        return self.output


class MIGraphXBackboneWrapper:
    """Run Vision and the Qwen transformer body entirely on DCU device buffers."""

    def __init__(self, backbone, vision_engine_path: str, language_engine_path: str):
        self.backbone = backbone
        self.eagle = backbone.model
        self.vision = _DeviceProgram(vision_engine_path)
        self.language = _DeviceProgram(language_engine_path)
        self.output_dtype = next(self.eagle.vision_model.parameters()).dtype
        self.position_ids = None
        self.interface_checked = False

    def __call__(self, vl_input: BatchFeature) -> BatchFeature:
        input_ids = vl_input["input_ids"].to(dtype=torch.int64).contiguous()
        attention_mask = vl_input["attention_mask"].to(dtype=torch.int64).contiguous()
        pixels = vl_input["pixel_values"]
        if not isinstance(pixels, (list, tuple)) or len(pixels) != 1:
            raise RuntimeError("Static Vision MXR requires exactly one image tensor")
        pixel_values = pixels[0].to(dtype=torch.float16).contiguous()

        vision_features = self.vision.run({"pixel_values": pixel_values})

        # The token embedding and 81-token insertion stay on the current DCU
        # stream.  They are intentionally outside the Language MXR so its
        # 620MB vocabulary embedding table is not duplicated in the engine.
        inputs_embeds = self.eagle.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = inputs_embeds.to(dtype=torch.float16).contiguous()
        image_mask = input_ids == self.eagle.image_token_index
        if not self.interface_checked:
            image_token_count = image_mask.sum().item()
            if image_token_count != vision_features.shape[0]:
                raise RuntimeError(
                    f"Image token count {image_token_count} != Vision output "
                    f"{vision_features.shape[0]}"
                )
            self.interface_checked = True
        flat = inputs_embeds.view(-1, inputs_embeds.shape[-1])
        flat[image_mask.view(-1)] = vision_features
        inputs_embeds = flat.view_as(inputs_embeds).contiguous()

        if self.position_ids is None or self.position_ids.device != input_ids.device:
            self.position_ids = torch.arange(
                input_ids.shape[1], device=input_ids.device, dtype=torch.int64
            ).unsqueeze(0)
        backbone_features_fp16 = self.language.run(
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "position_ids": self.position_ids,
            }
        )

        # Preserve the public Backbone contract for the remaining action-head
        # projector.  This conversion is device-to-device on the same stream.
        backbone_features = backbone_features_fp16.to(dtype=self.output_dtype)
        return BatchFeature(
            data={
                "backbone_features": backbone_features,
                "backbone_attention_mask": attention_mask == 1,
                "image_mask": image_mask,
            }
        )


def replace_backbone_with_migraphx(
    policy: Any,
    vision_engine_path: str,
    language_engine_path: str,
) -> MIGraphXBackboneWrapper:
    wrapper = MIGraphXBackboneWrapper(
        policy.model.backbone,
        vision_engine_path,
        language_engine_path,
    )
    policy.model.backbone.forward = wrapper
    # Keep the wrapper, preallocated outputs and async result handles alive.
    policy._migraphx_backbone_wrapper = wrapper
    logging.info("Eagle Vision and Language replaced with device-pointer MIGraphX engines")
    return wrapper
