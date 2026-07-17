"""MIGraphX adapter for the GR00T N1.6 DiT action-head model.

This first implementation intentionally uses host NumPy arrays at the
PyTorch/MIGraphX boundary.  It is useful for end-to-end correctness and an
inclusive latency measurement, but a final performance comparison should use
a device-pointer/zero-copy adapter.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch


class MIGraphXDiTWrapper:
    """Load and execute a compiled MIGraphX DiT program."""

    REQUIRED_INPUTS = {
        "sa_embs",
        "vl_embs",
        "timestep",
        "image_mask",
        "backbone_attention_mask",
    }

    def __init__(self, engine_path: str):
        try:
            import migraphx
        except ImportError as exc:
            raise RuntimeError(
                "MIGraphX Python bindings are unavailable. Check PYTHONPATH and the DTK installation."
            ) from exc

        self.model = migraphx.load(engine_path, format="msgpack")
        self.migraphx = migraphx
        self.zero_copy = not self.model.get_offload_copy()
        self.mode = "zero-copy" if self.zero_copy else "host-copy"
        self.parameter_shapes = self.model.get_parameter_shapes()
        dtype_map = {
            "float_type": torch.float32,
            "half_type": torch.float16,
            "int64_type": torch.int64,
            "bool_type": torch.bool,
        }
        self.parameter_dtypes = {}
        for name, shape in self.parameter_shapes.items():
            type_name = shape.type_string()
            if type_name not in dtype_map:
                raise RuntimeError(f"Unsupported MIGraphX interface type: {name}={type_name}")
            self.parameter_dtypes[name] = dtype_map[type_name]
        self._output = None
        self._last_results = None
        engine_inputs = set(self.model.get_inputs())
        missing = self.REQUIRED_INPUTS - engine_inputs
        if missing:
            raise RuntimeError(f"MIGraphX engine is missing inputs: {sorted(missing)}")

        logging.info("MIGraphX engine loaded: %s", engine_path)
        logging.info("MIGraphX inputs: %s", self.model.get_inputs())
        logging.info("MIGraphX outputs: %s", self.model.get_outputs())
        logging.info("MIGraphX adapter mode: %s", self.mode)

    @staticmethod
    def _float_numpy(tensor: torch.Tensor) -> np.ndarray:
        # The compiled engine keeps an FP32 external interface even though its
        # internal graph was quantized to FP16.
        return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()

    @staticmethod
    def _numpy(tensor: torch.Tensor, dtype: np.dtype) -> np.ndarray:
        return tensor.detach().to(device="cpu").contiguous().numpy().astype(dtype, copy=False)

    def __call__(
        self,
        sa_embs: torch.Tensor,
        vl_embs: torch.Tensor,
        timestep: torch.Tensor,
        image_mask: torch.Tensor | None,
        backbone_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if image_mask is None or backbone_attention_mask is None:
            raise RuntimeError(
                "This static MIGraphX engine requires image_mask and backbone_attention_mask"
            )

        if self.zero_copy:
            return self._call_zero_copy(
                sa_embs,
                vl_embs,
                timestep,
                image_mask,
                backbone_attention_mask,
            )

        params = {
            "sa_embs": self._float_numpy(sa_embs),
            "vl_embs": self._float_numpy(vl_embs),
            "timestep": self._numpy(timestep, np.int64),
            "image_mask": self._numpy(image_mask, np.bool_),
            "backbone_attention_mask": self._numpy(backbone_attention_mask, np.bool_),
        }

        outputs = self.model.run(params)
        # Copy the array because a MIGraphX argument may own/reuse its backing
        # storage on the next invocation.
        output_numpy = np.array(outputs[0], copy=True)
        return torch.from_numpy(output_numpy).to(device=sa_embs.device, dtype=sa_embs.dtype)

    @staticmethod
    def _device_tensor(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype):
        if tensor.device != device or tensor.dtype != dtype or not tensor.is_contiguous():
            tensor = tensor.to(device=device, dtype=dtype).contiguous()
        return tensor

    def _call_zero_copy(
        self,
        sa_embs: torch.Tensor,
        vl_embs: torch.Tensor,
        timestep: torch.Tensor,
        image_mask: torch.Tensor,
        backbone_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run MIGraphX directly on PyTorch-owned DCU buffers."""
        device = sa_embs.device
        if device.type != "cuda":
            raise RuntimeError(f"MIGraphX zero-copy requires a GPU tensor, got {device}")

        tensors = {
            "sa_embs": self._device_tensor(
                sa_embs, device, self.parameter_dtypes["sa_embs"]
            ),
            "vl_embs": self._device_tensor(
                vl_embs, device, self.parameter_dtypes["vl_embs"]
            ),
            "timestep": self._device_tensor(
                timestep, device, self.parameter_dtypes["timestep"]
            ),
            "image_mask": self._device_tensor(
                image_mask, device, self.parameter_dtypes["image_mask"]
            ),
            "backbone_attention_mask": self._device_tensor(
                backbone_attention_mask,
                device,
                self.parameter_dtypes["backbone_attention_mask"],
            ),
        }

        for name, tensor in tensors.items():
            expected = tuple(self.parameter_shapes[name].lens())
            if tuple(tensor.shape) != expected:
                raise RuntimeError(
                    f"MIGraphX input {name} shape mismatch: got {tuple(tensor.shape)}, "
                    f"expected {expected}"
                )

        output_shape = tuple(self.parameter_shapes["main:#output_0"].lens())
        output_dtype = self.parameter_dtypes["main:#output_0"]
        if (
            self._output is None
            or self._output.device != device
            or self._output.dtype != output_dtype
        ):
            self._output = torch.empty(output_shape, device=device, dtype=output_dtype)

        params = {
            name: self.migraphx.argument_from_pointer(
                self.parameter_shapes[name], tensor.data_ptr()
            )
            for name, tensor in tensors.items()
        }
        params["main:#output_0"] = self.migraphx.argument_from_pointer(
            self.parameter_shapes["main:#output_0"], self._output.data_ptr()
        )

        stream = torch.cuda.current_stream(device=device)
        self._last_results = self.model.run_async(
            params, stream.cuda_stream, "ihipStream_t"
        )

        # Keep the public DiT contract unchanged. If the caller uses BF16, this
        # is a device-to-device conversion on the same stream, not a host copy.
        if self._output.dtype != sa_embs.dtype:
            return self._output.to(dtype=sa_embs.dtype)
        return self._output


def replace_dit_with_migraphx(policy: Any, engine_path: str) -> MIGraphXDiTWrapper:
    """Replace the DiT forward method with a compiled MIGraphX program."""

    migraphx_dit = MIGraphXDiTWrapper(engine_path)

    def migraphx_forward(
        hidden_states,
        encoder_hidden_states,
        timestep,
        encoder_attention_mask=None,
        return_all_hidden_states=False,
        image_mask=None,
        backbone_attention_mask=None,
    ):
        if return_all_hidden_states:
            raise RuntimeError("MIGraphX DiT returns only the final output")

        return migraphx_dit(
            sa_embs=hidden_states,
            vl_embs=encoder_hidden_states,
            timestep=timestep,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
        )

    policy.model.action_head.model.forward = migraphx_forward
    logging.info("DiT replaced with MIGraphX engine (%s adapter)", migraphx_dit.mode)
    return migraphx_dit
