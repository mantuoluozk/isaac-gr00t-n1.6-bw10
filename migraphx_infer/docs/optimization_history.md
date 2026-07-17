# GR00T-N1.6-3B on BW10: MIGraphX DiT Inference

本目录保存经过验证的 GR00T-N1.6-3B DiT MIGraphX FP16 零拷贝部署产物。MIGraphX 只替换 Action Head 中的 `AlternateVLDiT`；数据处理、Eagle Backbone、Action Encoder/Decoder 和 diffusion 调度仍由 PyTorch 执行。

## 当前实测结果（2026-07-15）

测试环境：海光 BW10、MIGraphX 5.2.0、GR00T-N1.6-3B、`demo_data/gr1.PickNPlace`。统一测试口径固定为 batch size 1、Action Horizon 50、4 个 denoising steps、5 次 warmup（不计入统计）和 10 次稳态测量；进程使用 `numactl -N 3 -m 3` 绑定到 HCU 所在 NUMA 3。当前 MXR 是静态 `vl_embs=[1,116,2048]` 的 FP16 零拷贝引擎。

### 完整推理标准测试

使用 `benchmark_inference.py --compile_backbone --backbone_sdpa --num_iterations 10 --warmup 5`。`--backbone_sdpa` 将 Eagle Vision/Language attention 切换到可被 Dynamo 追踪的 PyTorch SDPA。四种模式在同一进程、相同输入下测试；脚本为 compiled Backbone 额外执行至少 8 次预热，所有预热均不计入正式统计。下表使用各组件和 E2E 的中位数。

| Device | Mode | Data Processing | Backbone | Action Head | E2E | Frequency |
|---|---:|---:|---:|---:|---:|---:|
| BW10 | PyTorch Eager + Backbone SDPA | 4.57 ms | 51.99 ms | 118.92 ms | 175.5 ms | 5.7 Hz |
| BW10 | torch.compile（DiT）+ Backbone SDPA | 4.57 ms | 52.08 ms | 52.74 ms | 109.4 ms | 9.1 Hz |
| BW10 | torch.compile（DiT + Backbone SDPA） | 4.57 ms | 36.33 ms | 52.72 ms | 93.6 ms | 10.7 Hz |
| BW10 | MIGraphX FP16（zero-copy + reduce fusion）+ compiled Backbone SDPA | 4.57 ms | 37.04 ms | 62.45 ms | 104.1 ms | 9.6 Hz |

相对同轮 PyTorch Eager：

- `torch.compile（DiT）`：E2E `1.60x`，Action Head `2.26x`。
- `torch.compile（DiT + Backbone SDPA）`：E2E `1.88x`，Action Head `2.26x`，是当前整模型最优结果。
- `MIGraphX FP16（zero-copy + reduce fusion）+ compiled Backbone SDPA`：E2E `1.69x`，Action Head `1.90x`。
- SDPA + compiled Backbone 将 Backbone 从 `51.99 ms` 降至 `36.33 ms`；相对原 compiled flash-attn Backbone 的 `40.84 ms` 再降低约 11%。
- MIGraphX 混合模式 E2E `104.1 ms`，比仅编译 DiT 的 `torch.compile`（`109.4 ms`）快约 `1.05x`，但比同时编译 DiT 和 Backbone SDPA（`93.6 ms`）慢约 `10.5 ms`。

### 独立 MXR 延迟

`migraphx_zero_copy_test.py` 使用 10 次 warmup、200 次正式测试：

```text
output shape: (1, 51, 1024)
NaN count: 0
Inf count: 0
median: 14.369 ms
mean:   14.387 ms
min:    14.277 ms
max:    14.706 ms
```

这里的约 `14.37 ms` 是一次 DiT 调用；一轮动作推理包含 4 个 denoising steps，因此完整 Action Head 时间高于该数值的 4 倍，并包含投影、调度等 PyTorch 开销。

### 真实数据精度

`validate_real_action.py` 从数据集中选择 trajectory 2 的 116-token 指令，并使用相同随机种子比较 PyTorch Eager 与 MIGraphX。最终动作张量 `[1,50,128]`：

| Metric | Result |
|---|---:|
| NaN / Inf | 0 / 0 |
| Mean absolute error | 0.04044 |
| RMSE | 0.05964 |
| Max absolute error | 0.41699 |
| Cosine similarity | 0.99813 |

### 结果结论

- 当前最高稳态吞吐：`torch.compile（DiT + Backbone SDPA）`，`93.6 ms / 10.7 Hz`。
- 当前最快 MIGraphX 完整混合方案：`直接 FP16 + zero-copy + reduce fusion + compiled Backbone SDPA`，`104.1 ms / 9.6 Hz`。
- 之前 3 次短测得到的 `100.0 ms / 10.0 Hz` 仅作为冒烟数据，不再用于正式横向比较。后续结果必须遵循本文统一测试口径，并建议同时记录 `hy-smi` 监控日志。

### 直接 FP16 ONNX 与 reduce fusion

“PyTorch 直接导出 FP16 ONNX → MIGraphX”在启用 reduce fusion 后成为当前最快 MIGraphX 路线。旧引擎仍保留用于回归：

```text
migraphx_infer/export_onnx_n1d6_fp16_direct.py
migraphx_infer/migraphx_compile_fp16_direct_zero_copy.py
migraphx_infer/fp16_direct/dit_model.onnx
migraphx_infer/fp16_direct/dit_model.onnx.data
migraphx_infer/dit_model_fp16_direct_vl116_zero_copy.mxr
migraphx_infer/dit_model_fp16_direct_vl116_reduce_fusion.mxr
```

该路线在导出前执行 `dit_model.half()`，FP16 ONNX 的浮点输入、权重和输出均为 FP16；编译时不再调用 `migraphx.quantize_fp16()`。

| Item | FP32 ONNX + quantize，禁用 reduce fusion | 直接 FP16，禁用 reduce fusion | 直接 FP16，启用 reduce fusion（推荐） |
|---|---:|---:|---:|
| ONNX external data | 约 4165 MB | 2082.50 MB | 2082.50 MB |
| MXR 外部浮点接口 | FP32 | FP16 | FP16 |
| 编译后指令数 | 2844 | 2839 | 1965 |
| 独立 reduce kernels | 194 | 194 | 97 |
| 同进程 A/B DiT median | 15.285 ms | 16.084 ms | 14.339 ms |
| 最终动作 cosine similarity | 0.99823 | 0.99813 | 0.99813 |
| Action Head（历史 3 次冒烟测试） | 67.49 ms | 71.90 ms | 62.83 ms |
| E2E + compiled Backbone（完整短测） | 122.1 ms / 8.2 Hz | 123.0 ms / 8.1 Hz | 111.2 ms / 9.0 Hz |

直接 FP16 在禁用 reduce fusion 时较慢；明确 unset `MIGRAPHX_DISABLE_REDUCE_FUSION` 后，编译器将大量 pointwise 与 reduce 合并，指令数减少约 `30.8%`、独立 reduce kernel 减半。同进程 A/B 中，新引擎比直接 FP16 无融合版本快 `1.122x`，比旧 FP32→FP16 引擎快 `1.066x`。整模型数据来自相邻测试轮次；同进程 DiT A/B 更适合判断引擎稳态差异。

复现实验：

```bash
python -u migraphx_infer/export_onnx_n1d6_fp16_direct.py \
  --model_path nvidia/GR00T-N1.6-3B \
  --dataset_path demo_data/gr1.PickNPlace \
  --embodiment_tag gr1 \
  --video_backend decord \
  --output_dir ./migraphx_infer/fp16_direct

env -u MIGRAPHX_DISABLE_REDUCE_FUSION \
MIGRAPHX_OUTPUT=migraphx_infer/dit_model_fp16_direct_vl116_reduce_fusion.mxr \
python -u migraphx_infer/migraphx_compile_fp16_direct_zero_copy.py

MIGRAPHX_ENGINE=migraphx_infer/dit_model_fp16_direct_vl116_reduce_fusion.mxr \
python -u migraphx_infer/migraphx_zero_copy_test.py
```

## 1. 最终目录

```text
migraphx_infer/
├── README.md
├── export_onnx_n1d6.py                 # 从 checkpoint 导出静态 DiT ONNX
├── patch_onnx_vl116.py                 # 仅用于修复历史 105-token ONNX
├── migraphx_check_vl116.py             # 检查 MIGraphX 能否解析 ONNX
├── migraphx_compile_vl116_zero_copy.py # ONNX -> FP16 zero-copy MXR
├── export_onnx_n1d6_fp16_direct.py      # 实验：直接导出 FP16 ONNX
├── migraphx_compile_fp16_direct_zero_copy.py # 实验：直接 FP16 ONNX -> MXR
├── migraphx_zero_copy_test.py          # MXR 随机输入/延迟测试
├── validate_real_action.py              # 与 PyTorch Eager 比较真实动作输出
├── validate_backbone_sdpa.py            # compiled flash-attn 与 compiled SDPA 精度对比
├── dit_model_fp16_direct_vl116_reduce_fusion.mxr # 当前最快 MIGraphX 引擎
├── dit_model_fp16_vl116_zero_copy.mxr  # 旧 FP32->FP16 引擎（回归用）
├── dit_model_fp16_direct_vl116_zero_copy.mxr # 旧 direct FP16 无融合引擎（回归用）
├── fp16_direct/                         # 实验 FP16 ONNX（约 2.1 GB）
└── groot_n1d6_migraphx/
    ├── dit_model.onnx                  # 历史 105-token 主图，供修补恢复
    ├── dit_model.onnx.data             # 外部权重（约 4.1 GB，必须保留）
    └── dit_model_vl116.onnx            # 最终编译使用的 116-token 主图
```

`dit_model_vl116.onnx` 引用同目录的 `dit_model.onnx.data`。不能只移动或复制主 `.onnx` 文件。当前默认部署产物是 `dit_model_fp16_direct_vl116_reduce_fusion.mxr`。

## 2. 环境

```bash
docker exec -it isaac bash
cd /workspace/isaac-gr00t-n1.6

export PYTHONPATH=/opt/dtk/lib:${PYTHONPATH}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
```

检查：

```bash
python -c "import torch, migraphx; print(torch.__version__); print(migraphx.__version__)"
```

当前 MXR 在 MIGraphX 5.2.0、BW10/gfx936 上编译。MXR 与 MIGraphX MXR 版本和 GPU 架构有关，不应直接复制到不同架构设备使用。

## 3. 直接运行已有 MXR

### 3.1 独立零拷贝测试

```bash
python -u migraphx_infer/migraphx_zero_copy_test.py
```

也可以调整测试次数：

```bash
MIGRAPHX_WARMUP=10 MIGRAPHX_ITERATIONS=200 \
python -u migraphx_infer/migraphx_zero_copy_test.py
```

预期：输出 `[1,51,1024]`，无 NaN/Inf，单次 DiT 中位延迟约 14.4 ms。

### 3.2 真实 GR1 动作精度

```bash
python -u migraphx_infer/validate_real_action.py
```

脚本会从 `demo_data/gr1.PickNPlace` 中选择一个 116-token 样本，用相同随机种子比较 PyTorch Eager 与 MIGraphX。已验证最终动作 `[1,50,128]` 的 cosine similarity 约为 0.9981。

### 3.3 完整 benchmark（仅 DiT 使用 MIGraphX）

```bash
python -u scripts/deployment/benchmark_inference.py \
  --model_path nvidia/GR00T-N1.6-3B \
  --dataset_path demo_data/gr1.PickNPlace \
  --embodiment_tag gr1 \
  --video_backend decord \
  --migraphx_engine_path ./migraphx_infer/dit_model_fp16_direct_vl116_reduce_fusion.mxr \
  --num_iterations 10 \
  --warmup 5 \
  --skip_compile
```

这里 Backbone 为 PyTorch Eager，只有 DiT 使用 MIGraphX。当前完整混合方案（compiled Backbone SDPA）在统一标准测试下为 Action Head 62.45 ms、E2E 104.1 ms、9.6 Hz。

如需测试混合方案（compiled Backbone + MIGraphX DiT），去掉 `--skip_compile` 并增加：

```bash
--compile_backbone \
--backbone_sdpa
```

## 4. 从历史 FP32 ONNX 重新编译 MXR（回归路线）

先检查 ONNX：

```bash
python -u migraphx_infer/migraphx_check_vl116.py
```

再编译：

```bash
MIGRAPHX_DISABLE_REDUCE_FUSION=1 \
python -u migraphx_infer/migraphx_compile_vl116_zero_copy.py
```

编译过程：

```python
program = migraphx.parse_onnx(".../dit_model_vl116.onnx")
migraphx.quantize_fp16(program)
program.compile(
    migraphx.get_target("gpu"),
    offload_copy=False,
    fast_math=True,
    device_id=0,
)
migraphx.save(program, ".../dit_model_fp16_vl116_zero_copy.mxr")
```

- `MIGRAPHX_DISABLE_REDUCE_FUSION=1`：规避 gfx936 上已遇到的 FP16 reduce-fusion 编译错误。
- `offload_copy=False`：允许运行时直接绑定 PyTorch DCU buffer，不经过 CPU。
- ONNX 外部接口为 FP32，MIGraphX 内部计算经 `quantize_fp16` 转为 FP16。
- 编译约 117 秒，输出 MXR 约 2.1 GB。

## 5. 从 checkpoint 重新导出 ONNX

不要覆盖当前已验证目录。先导出到新目录：

```bash
python -u migraphx_infer/export_onnx_n1d6.py \
  --model_path nvidia/GR00T-N1.6-3B \
  --dataset_path demo_data/gr1.PickNPlace \
  --embodiment_tag gr1 \
  --video_backend decord \
  --output_dir ./migraphx_infer/onnx_reexport_test
```

导出配置为静态 batch=1、`sa=51`、`vl=116`、ONNX Opset 19、FP32权重。大模型权重会写入 external-data 文件。

新导出后必须先检查输入和内部 reshape 常量，并用 `migraphx.parse_onnx` 解析成功，再替换 `groot_n1d6_migraphx`。历史导出曾得到 105-token 内部常量；`patch_onnx_vl116.py` 只用于这种历史文件，正常情况下不要运行。

## 6. 零拷贝运行原理

运行适配器位于：

```text
scripts/deployment/migraphx_dit_wrapper.py
```

它执行以下步骤：

1. `migraphx.load()` 加载编译好的 MXR，不再现场编译；
2. 确认引擎为 `offload_copy=False`；
3. 将 PyTorch GPU tensor 的 `data_ptr()` 包装为 `migraphx.argument_from_pointer()`；
4. 预分配 PyTorch GPU 输出 tensor，并将其指针作为 `main:#output_0`；
5. 使用 `run_async(..., stream.cuda_stream, "ihipStream_t")` 在当前 PyTorch HIP stream 执行；
6. 将 `policy.model.action_head.model.forward` 替换为 MIGraphX 调用。一次动作推理有 4 个 denoising steps，因此 MXR 执行 4 次。

## 7. 启动服务

仅 DiT 使用 MIGraphX：

```bash
python gr00t/eval/run_gr00t_server.py \
  --embodiment-tag GR1 \
  --model-path nvidia/GR00T-N1.6-3B \
  --migraphx-engine-path ./migraphx_infer/dit_model_fp16_direct_vl116_reduce_fusion.mxr
```

MIGraphX DiT + compiled Backbone：

```bash
python gr00t/eval/run_gr00t_server.py \
  --embodiment-tag GR1 \
  --model-path nvidia/GR00T-N1.6-3B \
  --compile-backbone \
  --migraphx-engine-path ./migraphx_infer/dit_model_fp16_direct_vl116_reduce_fusion.mxr
```

## 8. 静态形状限制

当前 MXR 只接受 `vl_embs=[1,116,2048]`。数据集中已观察到：

```text
"valid"                       -> 105 tokens
"pick the pear ..."          -> 116 tokens
"pick the meat/salt ..."     -> 116 tokens
"pick the sweet potato ..."  -> 117 tokens
```

输入不是 116 token 时适配器会明确报 shape mismatch。生产部署需要固定 padding 到同一长度，或为 105/116/117 分别编译 MXR 并按 shape 路由。

## 9. 常见问题

### `ModuleNotFoundError: migraphx`

```bash
export PYTHONPATH=/opt/dtk/lib:${PYTHONPATH}
```

### 编译阶段 reduce fusion 失败

历史 FP32 ONNX + `quantize_fp16` 路线在 gfx936 上曾触发 reduce-fusion 编译错误，可用：

```bash
MIGRAPHX_DISABLE_REDUCE_FUSION=1 python -u migraphx_infer/migraphx_compile_vl116_zero_copy.py
```

当前推荐的直接 FP16 ONNX 已验证可启用 reduce fusion。编译新引擎时必须 unset 该变量：

```bash
env -u MIGRAPHX_DISABLE_REDUCE_FUSION \
MIGRAPHX_OUTPUT=migraphx_infer/dit_model_fp16_direct_vl116_reduce_fusion.mxr \
python -u migraphx_infer/migraphx_compile_fp16_direct_zero_copy.py
```

### `migraphx-driver perf` 出现 GPU register failed

最终 MXR 是 `offload_copy=False`，输出参数需要由调用方分配并绑定。优先使用 `migraphx_zero_copy_test.py` 或完整 benchmark；不要用默认 host-input driver 路径评价该引擎。

### 模型加载长时间停在 Hugging Face HEAD 请求

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 10. Vision + Language + DiT三个MXR

客户TensorRT方案中的三个engine可以对应为：

| TensorRT | GR00T N1.6组件 | MIGraphX产物 |
|---|---|---|
| `vision.plan` | SigLIP2 Vision 27层、pixel-unshuffle和`mlp1` | `vision_projector_fp16_device.mxr` |
| `language.plan` | Qwen3前16层Transformer | `language_transformer_fp16_device.mxr` |
| `dit_model_bf16.trt` | Action Head DiT | `dit_model_fp16_direct_vl116_reduce_fusion.mxr` |

其中Language ONNX只导出Transformer主体，不导出最终LM Head。Backbone只使用最后一层hidden state，不使用15万词表logits，裁掉LM Head可以减少无效计算和模型体积。词嵌入和81个图像token写入保留在PyTorch侧，但输入、输出和中间Tensor均位于DCU设备上。

完整调用链：

```text
pixel_values BF16 [1,3,252,252]
    ↓ device cast
Vision MIGraphX FP16
    ↓ [81,2048]，DCU显存
Qwen embedding + 图像token写入
    ↓ [1,116,2048] FP16，DCU显存
Language MIGraphX FP16
    ↓ [1,116,2048]，DCU显存
DiT MIGraphX FP16 × 4 denoising steps
    ↓
Action Decoder
```

### 10.1 Device数据推理

实现遵循MIGraphX文档第6章：

1. 编译时设置`offload_copy=False`；
2. 输入使用`migraphx.argument_from_pointer(shape, tensor.data_ptr())`绑定PyTorch DCU Tensor；
3. 输出使用`torch.empty(..., device="cuda")`预分配，并绑定为`main:#output_0`；
4. 使用`run_async(params, stream.cuda_stream, "ihipStream_t")`在当前PyTorch HIP stream执行；
5. Vision、Language和DiT之间不调用`.cpu()`、`.numpy()`或`migraphx.from_gpu()`。

对应文件：

```text
migraphx_infer/export_backbone_onnx.py
migraphx_infer/compile_backbone_migraphx.py
migraphx_infer/benchmark_vision_migraphx.py
migraphx_infer/benchmark_language_migraphx.py
migraphx_infer/validate_three_mxr_action.py
scripts/deployment/migraphx_backbone_wrapper.py
```

### 10.2 导出ONNX

固定条件为batch=1、图像`[1,3,252,252]`、81个图像token、语言长度116。数据加载器可能返回`valid`等短文本，导致语言长度变为105，因此导出脚本和benchmark默认固定任务指令：

```text
pick the pear from the counter and place it in the plate
```

执行：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u migraphx_infer/export_backbone_onnx.py \
  --component vision \
  --expected_seq_len 116

python -u migraphx_infer/export_backbone_onnx.py \
  --component language \
  --expected_seq_len 116
```

生成：

```text
migraphx_infer/backbone/vision_projector_fp16.onnx       814 MB
migraphx_infer/backbone/language_transformer_fp16.onnx  1.6 GB
```

Vision中的位置编码插值依赖图像shape。252×252固定后patch网格始终为18×18，因此导出前预计算一次位置编码，并作为常量写入ONNX，避免运行时重复执行`_upsample_bilinear2d_aa`。

### 10.3 编译Device MXR

```bash
export PYTHONPATH=/opt/dtk/lib:${PYTHONPATH}

python -u migraphx_infer/compile_backbone_migraphx.py \
  --onnx migraphx_infer/backbone/vision_projector_fp16.onnx \
  --output migraphx_infer/backbone/vision_projector_fp16_device.mxr

MIGRAPHX_ENABLE_MHA=1 \
python -u migraphx_infer/compile_backbone_migraphx.py \
  --onnx migraphx_infer/backbone/language_transformer_fp16.onnx \
  --output migraphx_infer/backbone/language_transformer_fp16_device.mxr
```

| 组件 | 编译时间 | 输入 | 输出 | `offload_copy` |
|---|---:|---|---|---|
| Vision | 209.02 s | FP16 `[1,3,252,252]` | FP16 `[81,2048]` | `False` |
| Language | 101.44 s | FP16 `[1,116,2048]`、两个INT64 mask | FP16 `[1,116,2048]` | `False` |

当前ONNX分别包含27个和16个`Softmax`，Attention仍为展开的`MatMul + Softmax + MatMul`，没有`Attention`或`MultiHeadAttention`节点。`MIGRAPHX_ENABLE_MHA=1`已经生效，但仅设置环境变量不能把展开图自动转换为MHA，后续需要先进行ONNX Attention图融合。

### 10.4 单组件Device性能

测试条件为NUMA 3、warmup=5、10次稳态测量：

| 组件 | PyTorch Eager | MIGraphX Device | 降低 |
|---|---:|---:|---:|
| Vision + projector | 31.27 ms | **17.77 ms** | 13.50 ms |
| Language | 23.01 ms | **11.38 ms** | 11.63 ms |

MIGraphX单组件精度：

| 组件 | 对比目标 | MAE | RMSE | Max abs | Cosine | NaN/Inf |
|---|---|---:|---:|---:|---:|---:|
| Vision | FP16 PyTorch | 0.00554 | 0.01073 | 0.19531 | 0.999946 | 0/0 |
| Language | FP16 PyTorch | 0.00467 | 0.00765 | 0.50000 | 1.000016 | 0/0 |

### 10.5 三MXR完整性能

```bash
export PYTHONPATH=/opt/dtk/lib:${PYTHONPATH}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1

numactl -N 3 -m 3 \
python -u scripts/deployment/benchmark_inference.py \
  --model_path nvidia/GR00T-N1.6-3B \
  --dataset_path demo_data/gr1.PickNPlace \
  --embodiment_tag gr1 \
  --video_backend decord \
  --migraphx_engine_path migraphx_infer/dit_model_fp16_direct_vl116_reduce_fusion.mxr \
  --migraphx_vision_engine_path migraphx_infer/backbone/vision_projector_fp16_device.mxr \
  --migraphx_language_engine_path migraphx_infer/backbone/language_transformer_fp16_device.mxr \
  --backbone_sdpa \
  --compile_backbone \
  --num_iterations 10 \
  --warmup 5
```

同一进程、batch=1、Action Horizon=50、4个denoising steps、warmup=5、10次稳态测量结果：

| 模式 | Data Processing | Backbone | Action Head | E2E | 频率 |
|---|---:|---:|---:|---:|---:|
| PyTorch Eager | 4.37 ms | 51.99 ms | 120.96 ms | 177.3 ms | 5.6 Hz |
| torch.compile DiT | 4.37 ms | 52.08 ms | 53.14 ms | 109.6 ms | 9.1 Hz |
| torch.compile DiT + Backbone SDPA | 4.37 ms | 36.20 ms | 53.11 ms | **93.7 ms** | **10.7 Hz** |
| MIGraphX Vision + Language + DiT Device | 4.37 ms | **30.92 ms** | 62.89 ms | **98.1 ms** | **10.2 Hz** |

三个MXR将Backbone从compiled SDPA的36.20ms降至30.92ms，降低5.28ms；但MIGraphX DiT Action Head比compiled DiT慢9.78ms，因此整体仍比全`torch.compile`慢4.4ms。当前整模型最快仍是全`torch.compile`，但三MXR方案已经超过仅编译DiT的109.6ms，并实现了与客户三个TensorRT engine相同的部署拆分方式。

### 10.6 最终动作精度

相同116-token真实输入和相同diffusion随机种子，对比PyTorch BF16与三个MXR最终动作`[1,50,128]`：

| 指标 | 结果 |
|---|---:|
| NaN / Inf | 0 / 0 |
| MAE | 0.004472 |
| RMSE | 0.007044 |
| Max absolute error | 0.036133 |
| Cosine similarity | 0.999975 |

完整日志：

```text
migraphx_infer/backbone_three_mxr_benchmark_10.log
migraphx_infer/backbone_three_mxr_accuracy.log
```

### 10.7 116-token规格和各参数含义

这里的105、116或117不是单纯的自然语言token数量，而是Vision和Language拼接后的VL总序列长度：

```text
VL总长度 = 图像token + 对话模板/特殊token + 任务文本token
```

当前单张252×252图像先产生18×18个patch，经过`pixel-unshuffle`后得到9×9，即81个图像token。固定任务指令：

```text
pick the pear from the counter and place it in the plate
```

加上对话模板、特殊token和81个图像token后，总长度为116。数据集中的短文本`valid`对应105，另一条更长任务指令可能为117。因此105改成116不是修改模型的最大语言能力，而是为当前任务输入选择了一套静态编译shape。

PyTorch版GR00T配置的`max_seq_len`为1024，底层Qwen3的`max_position_embeddings`为40960，可以处理不同长度；但是当前Language和DiT MXR只接受116。任务文本、对话模板、图像数量、图像分辨率或相机数量发生变化，都可能改变VL长度。长度不是116时不能直接送入当前MXR，否则会shape mismatch。强制截断还可能丢失任务语义。

相关输入参数：

| 参数 | 当前值 | 含义 |
|---|---|---|
| Batch | 1 | 每次处理1条观测 |
| `pixel_values` | `[1,3,252,252]` | 单张RGB图像 |
| Vision token | 81 | 18×18 patch经过pixel-unshuffle后得到9×9 |
| VL length | 116 | 图像token、模板token和任务文本token的总长度 |
| VL hidden | 2048 | Qwen3和DiT之间的Backbone特征宽度 |
| `attention_mask` | `[1,116]` | 标记真实token和padding token |
| `position_ids` | `[1,116]` | 每个token的位置编号 |
| `image_mask` | `[1,116]` | 标记VL序列中的图像token位置 |
| Action Horizon | 50 | 一次预测未来50步动作 |
| `sa_embs` | `[1,51,1536]` | 1个state token加50个带噪action token |
| DiT输出 | `[1,51,1024]` | DiT隐状态，后续解码出50×128动作 |
| Denoising Steps | 4 | 每次动作预测调用DiT 4次去噪 |
| Warmup / Iterations | 5 / 10 | 预热5次不计时，统计10次稳态结果 |

如果后续需要覆盖不同指令，推荐两种方式：

1. 选择固定上限，例如128，将短序列右侧padding到128，并同时重新导出Language和DiT，使用mask屏蔽padding；必须重新做最终动作精度验证。
2. 为105、116、117或128分别编译静态MXR，运行时按照实际长度路由。该方式占用更多磁盘，但不会截断任务文本。

图像数量或分辨率变化会同时改变Vision token，不能只对语言文本padding，需要为Vision和Language建立配套规格。

### 10.8 ONNX Attention融合实验

ONNX Runtime 1.22.2优化器可以直接读取ONNX并按预定义模板进行融合。分别使用`model_type=vit`和`model_type=clip`处理SigLIP2 Vision后，结果均为：

```text
Attention: 0
MultiHeadAttention: 0
Softmax: 27
```

说明ORT完成了图读取和常量清理，但当前`torch.onnx`导出的SigLIP2 QKV拓扑与内置ViT/CLIP模板不匹配，没有真正生成MHA节点。

随后按照实际图结构，将27段`QK^T -> Softmax -> V`改写为`com.microsoft::MultiHeadAttention`。实验图从1481个节点减少到995个节点，并通过ONNX checker。ORT CPU对真实输入验证正确：

| 实现 | 对FP16 PyTorch余弦相似度 | 结果 |
|---|---:|---|
| 原始Vision MXR | 0.999946 | 可用 |
| MHA ONNX / ORT CPU | 0.999979 | 图改写语义正确 |
| MHA三输入 / MIGraphX | 0.642287 | 数值错误，不可用 |
| MHA packed-bias / MIGraphX | 未完成 | 运行触发Flash MHA kernel VMFault |

三输入MHA在MIGraphX上的中位数为13.2240ms，相比原始17.7577ms降低4.5337ms，说明MHA kernel存在明显性能潜力；但是精度不满足要求，不能接入端到端。packed-bias形式在ORT中同样正确，但MIGraphX 5.2.0运行时发生非法显存访问，也不能使用。

SigLIP2的`hidden_size=1152`、`num_attention_heads=16`，每个head维度为72。VMFault日志进入`flash_fwd_kernel`，当前问题可能与MIGraphX 5.2.0 Flash MHA对非64/128 head维度或该QKV输入布局的支持有关。该判断是根据kernel日志作出的推测，需要海光侧结合最小复现图确认。

原始Vision ONNX继续测试`opt_compile`，自动调优耗时400.21s并选择NHWC，正式结果为17.7569ms，与普通编译的17.7577ms基本相同。因此当前不替换任何正式引擎，默认仍使用：

```text
migraphx_infer/backbone/vision_projector_fp16_device.mxr
```

后续优化顺序：

1. 使用MHA ONNX、`vision_reference.pt`和VMFault日志向海光确认head_dim=72及packed-bias支持，在更新的DTK/MIGraphX版本回归。
2. MHA兼容后先过ORT、MIGraphX单组件精度和最终动作精度，再测完整E2E。
3. Qwen3 Language包含16个Query Head、8个KV Head、RoPE和Q/K Norm，属于GQA结构；当前ORT没有Qwen3模板，在Vision MHA问题解决前不强行改写Language。
4. 当前三个MXR相对全`torch.compile`的主要差距仍在DiT：62.89ms对53.11ms，后续继续分析DiT kernel和混合精度比盲目改动Language更稳妥。

### 10.9 DiT kernel画像和exhaustive tune

使用相同静态shape单独测试DiT forward：

| 实现 | 精度 | 单次DiT中位数 |
|---|---|---:|
| `torch.compile` | BF16 | 12.5041 ms |
| MIGraphX基线 | FP16 | 14.3031 ms |
| MIGraphX `exhaustive_tune=True` | FP16 | **14.1532 ms** |

MIGraphX基线和exhaustive-tune均为`offload_copy=False`，测试条件为NUMA 3、warmup=5、10次稳态测量。exhaustive tune编译耗时154.42s，单次DiT降低0.1498ms，约1.05%。新旧MXR使用完全相同的FP16 Device输入直接对比：

| 指标 | 结果 |
|---|---:|
| NaN / Inf | 0 / 0 |
| MAE | 0.000517 |
| RMSE | 0.000713 |
| Max abs | 0.005859 |
| Cosine | 0.9999994 |

三MXR端到端使用batch=1、Action Horizon=50、4步去噪、warmup=5、10次稳态测量，新DiT MXR结果为：

| Data Processing | Backbone | Action Head | E2E | 频率 |
|---:|---:|---:|---:|---:|
| 4.38 ms | 31.50 ms | **62.47 ms** | **98.0 ms** | **10.2 Hz** |

相对历史同口径三MXR的Action Head 62.89ms，降低约0.42ms；不同进程会有少量波动，独立DiT测试确认exhaustive tune的稳定收益约0.15ms/次。

hipprof对MIGraphX DiT执行5次得到2484个GPU kernel，约497个kernel/次；GPU kernel累计72.71ms，约14.54ms/次。按时间分类：

- GEMM约占72%，其中每层重复的一类GEMM约占31.6%；
- GELU融合kernel约占7.6%；
- LayerNorm和Softmax reduce约占6%左右；
- 其余为大量add、mul、mask和布局处理小kernel；
- `hipExtModuleLaunchKernel`约2480次，平均约4.76us，约2.36ms host launch开销/次DiT。

`torch.compile`生成的Triton kernel会把`addmm + SiLU/GELU + LayerNorm + residual`合并，而当前MIGraphX仍有较多独立kernel。因此exhaustive tune只能优化GEMM算法，无法解决主要的图融合和launch数量差距。后续源码优化优先级：

1. 检查GEMM epilogue是否可以使用hipBLASLt融合bias、SiLU/GELU和residual。
2. 扩展pointwise/reduce fusion，将AdaLayerNorm的`LayerNorm * (1 + scale) + shift`和residual add合并。
3. 检查相邻`contiguous/transpose/reshape`是否产生真实kernel，删除只为布局转换产生的copy。
4. 在每项修改后统计单次DiT kernel数量，目标不只是降低某个kernel延迟，还要从约497次launch明显下降。

测试时发现旧版`migraphx_zero_copy_test.py`按FP32创建随机输入，而当前direct-FP16 MXR接口实际为half。测试脚本已修正为根据引擎接口使用FP16输入和输出。正式`MIGraphXDiTWrapper`原本就会读取`shape.type_string()`并转换到FP16，因此历史端到端和最终动作精度不受该测试脚本问题影响。

当前候选引擎：

```text
migraphx_infer/dit_model_fp16_direct_vl116_reduce_fusion_exhaustive.mxr
```

### 10.10 DiT图融合补充实验和异步执行优化

继续在130机器、BW10、NUMA 3上对DiT进行优化。除特别说明外，测试条件均为batch=1、Action Horizon=50、4个denoising steps、warmup=5、10次稳态测量。254机器在本轮没有继续运行代码。

首先测试了几类图融合候选：

| 候选 | 图变化 | 单次DiT结果 | 精度/结论 |
|---|---|---:|---|
| 内置LayerNorm融合 | 64个`layernorm_kernel`和1个`add_layernorm_kernel` | 14.2520 -> 14.3777 ms | 慢0.88%，不采用 |
| 自定义GEMM+bias+residual | 63处命中，code object由298降到235 | 14.2278 -> 19.1267 ms | cosine=0.9613，且慢34.43%，不采用 |
| 内置MIOpen SkipLayerNorm | 64个`gpu::miopen_skiplayernorm`，code object由298降到171 | 14.2825 -> 14.7732 ms | cosine=0.9999990，但慢3.44%，不采用 |
| 内置DiT Attention融合 | 16个`gpu::multi_head_attention_fa_std` | 14.2483 -> 13.6771 ms | cosine=0.8813，不采用 |

Attention融合虽然快约4%，但误差明显。继续设置`MIGRAPHX_Q_BHSD_TO_BSHD=1`、`MIGRAPHX_K_BHSD_TO_BSHD=1`和`MIGRAPHX_V_BHSD_TO_BSHD=1`后，误差基本不变，说明当前MIGraphX 5.2.0的Flash MHA布局或实现与该DiT图不匹配，不能只通过环境变量修复。

也尝试使用PyTorch HIP Graph捕获MIGraphX `run_async`。改为当前PyTorch stream并使用`capture_error_mode="relaxed"`后，仍然在MIGraphX code object启动处出现`invalid configuration argument`和`capturing stream has unjoined work`，因此当前执行路径不能直接被PyTorch HIP Graph捕获。

#### 删除MXR内部的逐次同步

检查已编译MXR发现主模块返回前有一个`hip::sync_stream`。完整Action Head会连续调用DiT 4次，wrapper本身已经用`torch.cuda.current_stream()`调用`run_async`，benchmark又会在整个Action Head计时结束后执行`torch.cuda.synchronize()`。同一stream天然保证4次DiT及后续PyTorch算子的先后依赖，因此每次DiT内部再同步一次没有必要。

使用`remove_migraphx_sync.cpp`加载已编译MXR，将返回值从`hip::sync_stream`改为其输入并删除该指令，再保存为：

```text
migraphx_infer/dit_model_fp16_direct_vl116_reduce_fusion_exhaustive_async.mxr
```

### 10.11 Vision和Language删除内部同步

Vision和Language MXR主模块返回前也各有一个`hip::sync_stream`。Backbone wrapper使用当前PyTorch stream依次执行Vision、图像token写入、Language和dtype转换，benchmark在整个Backbone结束后统一同步，因此可以使用同一个工具生成：

```text
migraphx_infer/backbone/vision_projector_fp16_device_async.mxr
migraphx_infer/backbone/language_transformer_fp16_device_async.mxr
```

原引擎和async引擎使用相同真实参考输入比较，Vision和Language的MAE、RMSE、Max abs均为0，输出逐元素一致。固定NUMA 3、warmup=5、iterations=10的单组件结果：

| 组件 | 原MXR | async MXR | 变化 |
|---|---:|---:|---:|
| Vision | 17.7790 ms | **17.5464 ms** | 降低0.2326 ms，快1.31% |
| Language | 11.4009 ms | **11.1700 ms** | 降低0.2309 ms，快2.03% |

三个async MXR同时接入完整模型后的结果：

| Data Processing | Backbone | Action Head | E2E | 频率 |
|---:|---:|---:|---:|---:|
| 4.35 ms | **30.42 ms** | **58.72 ms** | **93.5 ms** | **10.7 Hz** |

相对同条件原三MXR的96.7ms，E2E降低3.2ms；相对仅异步DiT的94.0ms，Backbone再降低约0.45ms，与两个单组件收益之和一致。该结果达到并略好于历史同口径`torch.compile DiT + Backbone SDPA`的93.7ms。

正式运行命令：

```bash
numactl -N 3 -m 3 python scripts/deployment/benchmark_inference.py \
  --model_path nvidia/GR00T-N1.6-3B \
  --dataset_path demo_data/gr1.PickNPlace \
  --embodiment_tag gr1 \
  --video_backend decord \
  --migraphx_engine_path migraphx_infer/dit_model_fp16_direct_vl116_reduce_fusion_exhaustive_async.mxr \
  --migraphx_vision_engine_path migraphx_infer/backbone/vision_projector_fp16_device_async.mxr \
  --migraphx_language_engine_path migraphx_infer/backbone/language_transformer_fp16_device_async.mxr \
  --backbone_sdpa \
  --skip_compile \
  --num_iterations 10 \
  --warmup 5
```

生成工具必须链接DTK内置`/opt/dtk/lib`，这样输出MXR仍为序列化版本9，可以和现有Vision、Language MXR一起加载：

```bash
c++ -std=c++17 -O2 migraphx_infer/remove_migraphx_sync.cpp \
  -I/opt/dtk/include -L/opt/dtk/lib -Wl,-rpath,/opt/dtk/lib \
  -lmigraphx -o migraphx_infer/remove_migraphx_sync_system

migraphx_infer/remove_migraphx_sync_system \
  migraphx_infer/dit_model_fp16_direct_vl116_reduce_fusion_exhaustive.mxr \
  migraphx_infer/dit_model_fp16_direct_vl116_reduce_fusion_exhaustive_async.mxr
```

新旧MXR使用完全相同的FP16 Device随机输入直接比较，MAE、RMSE、Max abs全部为0，Cosine为1.0000001，输出逐元素一致。单次DiT结果：

| 实现 | 单次DiT中位数 | 变化 |
|---|---:|---:|
| 原正式MXR，内部同步 | 14.3806 ms | 参考 |
| async MXR，删除内部同步 | **13.7946 ms** | **降低0.5860 ms，快4.08%** |

`benchmark_inference.py`原来会在MIGraphX Backbone分支将命令行的`--warmup 5`静默提高为8。本轮已修改为直接使用`args.warmup`，保证命令条件和实际条件一致。完整三MXR严格5/10复测结果：

| 三MXR实现 | Data Processing | Backbone | Action Head | E2E | 频率 |
|---|---:|---:|---:|---:|---:|
| 原DiT同步MXR | 4.33 ms | 30.84 ms | 61.48 ms | 96.7 ms | 10.3 Hz |
| async DiT MXR复现 | 4.41 ms | 30.87 ms | **58.65 ms** | **94.0 ms** | **10.6 Hz** |

删除内部同步后，Action Head降低2.83ms，E2E降低2.7ms。当前三MXR已经基本追平历史同口径`torch.compile DiT + Backbone SDPA`的93.7ms、10.7Hz，同时保留MIGRAPHX Backbone约30.9ms的优势。

顺序执行前后两轮时，第二个进程曾出现Data Processing由4.33ms升到8.41ms，但GPU部分仍为Backbone 30.72ms、Action Head 58.56ms。将async方案单独重跑后Data Processing恢复为4.41ms，E2E复现为94.0ms，因此该8.41ms属于跨进程CPU预处理波动，不是删除DiT同步导致的回退。

当前正式推荐DiT引擎更新为：

```text
migraphx_infer/dit_model_fp16_direct_vl116_reduce_fusion_exhaustive_async.mxr
```

### 10.12 MIGraphX源码编译和DiT GELU epilogue实验

原`/workspace/migraphx-source`构建产物是MIGraphX 5.1.2，不能替换当前5.2.0运行库。因此建立独立5.2.0工作副本：

```text
/workspace/migraphx-gr00t-5.2
```

使用`migraphx:5.2.0-ubuntu22.04-dtk26.04-py3.11`镜像构建，并保留`gfx906/gfx926/gfx928/gfx936/gfx938`全部架构。首次构建需要把复制来的`depend/cget/cget.cmake`前缀改为`/workspace/migraphx/depend`；镜像还缺少`librt.so`开发链接名：

```bash
ln -sfn /lib/x86_64-linux-gnu/librt.so.1 \
  /usr/lib/x86_64-linux-gnu/librt.so
```

最终`migraphx_gpu`和`all`均构建成功。自定义库放在以下目录，通过临时`LD_LIBRARY_PATH`加载，没有覆盖`/opt/dtk`：

```text
migraphx_infer/custom_migraphx_5.2_gr00t/lib
```

只设置`MIGRAPHX_ENABLE_HIPBLASLT_GEMM_FUSION=1`时，图仍为1971条指令、197个GEMM和298个code object，没有任何融合命中。DiT实际的32个MLP激活kernel名称为：

```text
add_pow_mul_add_mul_tanh_mul_add_mul_kernel
```

扩展matcher后，32处全部变为`gpu::hipblaslt_gemm_fusion`：

| 指标 | 无融合 | 命中后 |
|---|---:|---:|
| 总指令数 | 1971 | 1875 |
| `gpu::gemm` | 197 | 165 |
| `gpu::code_object` | 298 | 266 |
| `gpu::hipblaslt_gemm_fusion` | 0 | 32 |

原融合实现没有设置`HIPBLASLT_MATMUL_DESC_BIAS_POINTER`。修正bias pointer、bias数据类型、`beta=0`和零workspace算法后，结果仍然不满足精度要求：

| 指标 | 结果 |
|---|---:|
| NaN / Inf | 0 / 0 |
| MAE | 0.075649 |
| RMSE | 0.103897 |
| Max abs | 0.648438 |
| Cosine | 0.983108 |

模型使用tanh-approx GELU，而DTK 26.04 hipBLASLt没有对应的tanh-GELU epilogue。固定NUMA 3、warmup=5、10次稳态测量：

| 实现 | 单次DiT中位数 | 结论 |
|---|---:|---|
| 同源码无融合baseline | **14.2389 ms** | 参考 |
| 32处GELU_BIAS融合 | 15.5113 ms | 慢8.93%，且数值不合格 |

因此该实验不进入端到端流程，也不替换正式MXR。自编译源码的MXR序列化版本为10，DTK内置5.2.0为9，说明两者不是完全相同的源码快照；A/B必须分别用同一套自编译库生成MXR。

下一步转向数值完全等价的AdaLayerNorm、residual add和pointwise/reduce融合。正式候选仍为：

```text
migraphx_infer/dit_model_fp16_direct_vl116_reduce_fusion_exhaustive.mxr
```

### 10.13 当前正式结论

上面的源码融合实验均未替换正式引擎。最终有效优化是删除系统DTK版Vision、Language和DiT三个MXR内部的`hip::sync_stream`，三个组件的输出都逐元素一致。严格NUMA 3、batch=1、Action Horizon=50、4个denoising steps、warmup=5、10次稳态测量下，完整三MXR为Backbone 30.42ms、Action Head 58.72ms、E2E 93.5ms、10.7Hz。

当前正式推荐引擎：

```text
migraphx_infer/dit_model_fp16_direct_vl116_reduce_fusion_exhaustive_async.mxr
migraphx_infer/backbone/vision_projector_fp16_device_async.mxr
migraphx_infer/backbone/language_transformer_fp16_device_async.mxr
```
