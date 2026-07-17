"""Standalone zero-copy smoke test and latency benchmark for the final DiT MXR."""

import os
import time

import migraphx
import numpy as np
import torch


ENGINE = os.environ.get(
    "MIGRAPHX_ENGINE", "migraphx_infer/artifacts/dit/dit_fp16_vl116_async.mxr"
)
WARMUP = int(os.environ.get("MIGRAPHX_WARMUP", "5"))
ITERATIONS = int(os.environ.get("MIGRAPHX_ITERATIONS", "100"))

model = migraphx.load(ENGINE, format="msgpack")
if model.get_offload_copy():
    raise RuntimeError("Expected a zero-copy MXR compiled with offload_copy=False")

shapes = model.get_parameter_shapes()
expected_inputs = {
    "sa_embs": ((1, 51, 1536), torch.float16),
    "vl_embs": ((1, 116, 2048), torch.float16),
    "timestep": ((1,), torch.int64),
    "image_mask": ((1, 116), torch.bool),
    "backbone_attention_mask": ((1, 116), torch.bool),
}

for name, (lens, _) in expected_inputs.items():
    actual = tuple(shapes[name].lens())
    if actual != lens:
        raise RuntimeError(f"{name}: engine shape {actual}, expected {lens}")

generator = torch.Generator(device="cuda").manual_seed(20260714)
tensors = {
    "sa_embs": torch.randn(
        expected_inputs["sa_embs"][0], generator=generator, device="cuda",
        dtype=expected_inputs["sa_embs"][1]
    ),
    "vl_embs": torch.randn(
        expected_inputs["vl_embs"][0], generator=generator, device="cuda",
        dtype=expected_inputs["vl_embs"][1]
    ),
    "timestep": torch.tensor([1], dtype=torch.int64, device="cuda"),
    "image_mask": torch.ones(
        expected_inputs["image_mask"][0], dtype=torch.bool, device="cuda"
    ),
    "backbone_attention_mask": torch.ones(
        expected_inputs["backbone_attention_mask"][0], dtype=torch.bool, device="cuda"
    ),
}

output_name = "main:#output_0"
output_shape = tuple(shapes[output_name].lens())
dtype_map = {
    "half_type": torch.float16,
    "float_type": torch.float32,
}
output_type = shapes[output_name].type_string()
if output_type not in dtype_map:
    raise RuntimeError(f"Unsupported output type: {output_type}")
output = torch.empty(output_shape, dtype=dtype_map[output_type], device="cuda")
params = {
    name: migraphx.argument_from_pointer(shapes[name], tensor.data_ptr())
    for name, tensor in tensors.items()
}
params[output_name] = migraphx.argument_from_pointer(
    shapes[output_name], output.data_ptr()
)

stream = torch.cuda.current_stream()
for _ in range(WARMUP):
    model.run_async(params, stream.cuda_stream, "ihipStream_t")
stream.synchronize()

times = []
for _ in range(ITERATIONS):
    start = time.perf_counter()
    model.run_async(params, stream.cuda_stream, "ihipStream_t")
    stream.synchronize()
    times.append((time.perf_counter() - start) * 1000)

host_output = output.float().cpu().numpy()
print("engine:", ENGINE)
print("offload_copy:", model.get_offload_copy())
print("output shape:", output_shape)
print("NaN count:", int(np.isnan(host_output).sum()))
print("Inf count:", int(np.isinf(host_output).sum()))
print("median_ms:", float(np.median(times)))
print("mean_ms:", float(np.mean(times)))
print("min_ms:", float(np.min(times)))
print("max_ms:", float(np.max(times)))
