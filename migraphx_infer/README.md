# GR00T-N1.6 MIGraphX部署说明

本文档记录海光BW10上GR00T-N1.6-3B的正式MIGraphX推理流程。所有命令均在容器`isaac`的项目根目录`/workspace/isaac-gr00t-n1.6`执行。

详细的优化过程、失败实验和原因分析保存在[optimization_history.md](docs/optimization_history.md)，本README只保留当前可复现流程。

## 1. 当前结果

固定测试条件：

```text
Device: BW10
Batch: 1
Action Horizon: 50
Denoising Steps: 4
VL Length: 116
Warmup: 5
Iterations: 10
CPU/Memory NUMA: 3
```

| 实现 | Data Processing | Backbone | Action Head | E2E | 频率 |
|---|---:|---:|---:|---:|---:|
| PyTorch Eager | 4.35 ms | 52.00 ms | 120.40 ms | 176.7 ms | 5.7 Hz |
| 历史torch.compile DiT+Backbone SDPA | 4.37 ms | 36.20 ms | 53.11 ms | 93.7 ms | 10.7 Hz |
| MIGraphX三个同步MXR | 4.33 ms | 30.84 ms | 61.48 ms | 96.7 ms | 10.3 Hz |
| MIGraphX三个async MXR | **4.35 ms** | **30.42 ms** | **58.72 ms** | **93.5 ms** | **10.7 Hz** |

三个async MXR均使用Device Pointer零拷贝，删除了MXR返回前不必要的`hip::sync_stream`。新旧引擎逐元素比较，MAE、RMSE和Max abs均为0。

## 2. 目录结构

```text
migraphx_infer/
├── README.md
├── artifacts/
│   ├── dit/                 # DiT ONNX、同步MXR、正式async MXR
│   └── backbone/            # Vision/Language ONNX、参考输入和MXR
├── scripts/
│   ├── export/              # PyTorch导出ONNX
│   ├── compile/             # ONNX编译为同步MXR
│   ├── async/               # 删除MXR内部同步
│   ├── validate/            # 精度和性能验证
│   └── analyze/             # 图结构和profile分析
├── results/
│   ├── final/               # 最终async方案日志
│   ├── baseline/            # Eager、torch.compile和旧三MXR日志
│   ├── build/               # 导出、编译和async生成日志
│   └── profiles/            # hipprof原始数据
├── experiments/             # 未进入正式方案的实验脚本和小型日志
└── docs/optimization_history.md
```

`artifacts`中同时保留同步和async MXR。同步MXR用于重新验证删除同步前后的输出，async MXR用于正式推理。

## 3. 环境准备

```bash
cd /workspace/isaac-gr00t-n1.6

export PYTHONPATH=/opt/dtk/lib:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
```

模型使用Hugging Face缓存中的`nvidia/GR00T-N1.6-3B`，数据集使用`demo_data/gr1.PickNPlace`。

## 4. 正式生成流程

如果只需要运行现有引擎，可以直接跳到第8节。如果需要从模型重新生成全部文件，按照第4至第7节顺序执行。

### 4.1 导出DiT FP16 ONNX

使用脚本：

```text
migraphx_infer/scripts/export/export_dit_onnx.py
```

执行命令：

```bash
python migraphx_infer/scripts/export/export_dit_onnx.py \
  --model_path nvidia/GR00T-N1.6-3B \
  --dataset_path demo_data/gr1.PickNPlace \
  --embodiment_tag gr1 \
  --video_backend decord \
  --output_dir migraphx_infer/artifacts/dit
```

输出：

```text
migraphx_infer/artifacts/dit/dit_model.onnx
migraphx_infer/artifacts/dit/dit_model.onnx.data
```

该脚本直接导出FP16权重和FP16 Tensor接口，使用ONNX opset 19，固定输入为：

```text
sa_embs:                   [1, 51, 1536] FP16
vl_embs:                   [1, 116, 2048] FP16
timestep:                  [1] INT64
image_mask:                [1, 116] BOOL
backbone_attention_mask:   [1, 116] BOOL
output:                    [1, 51, 1024] FP16
```

### 4.2 编译DiT同步MXR

使用脚本：

```text
migraphx_infer/scripts/compile/compile_dit_migraphx.py
```

执行命令：

```bash
unset MIGRAPHX_DISABLE_REDUCE_FUSION
export MIGRAPHX_EXHAUSTIVE_TUNE=1
export MIGRAPHX_ONNX=migraphx_infer/artifacts/dit/dit_model.onnx
export MIGRAPHX_OUTPUT=migraphx_infer/artifacts/dit/dit_fp16_vl116_sync.mxr

python migraphx_infer/scripts/compile/compile_dit_migraphx.py
```

编译使用`offload_copy=False`、`fast_math=True`和`exhaustive_tune=True`。不要设置`MIGRAPHX_DISABLE_REDUCE_FUSION=1`，否则会关闭正式方案使用的reduce fusion。

### 4.3 导出Vision和Language ONNX

使用脚本：

```text
migraphx_infer/scripts/export/export_backbone_onnx.py
```

Vision导出：

```bash
python migraphx_infer/scripts/export/export_backbone_onnx.py \
  --component vision \
  --model_path nvidia/GR00T-N1.6-3B \
  --dataset_path demo_data/gr1.PickNPlace \
  --embodiment_tag gr1 \
  --video_backend decord \
  --expected_seq_len 116 \
  --language_prompt "pick the pear from the counter and place it in the plate" \
  --output_dir migraphx_infer/artifacts/backbone
```

Language导出：

```bash
python migraphx_infer/scripts/export/export_backbone_onnx.py \
  --component language \
  --model_path nvidia/GR00T-N1.6-3B \
  --dataset_path demo_data/gr1.PickNPlace \
  --embodiment_tag gr1 \
  --video_backend decord \
  --expected_seq_len 116 \
  --language_prompt "pick the pear from the counter and place it in the plate" \
  --output_dir migraphx_infer/artifacts/backbone
```

输出包括两个ONNX和两个真实参考输入：

```text
vision_projector_fp16.onnx
vision_reference.pt
language_transformer_fp16.onnx
language_reference.pt
```

### 4.4 编译Vision和Language同步MXR

使用脚本：

```text
migraphx_infer/scripts/compile/compile_backbone_migraphx.py
```

执行命令：

```bash
python migraphx_infer/scripts/compile/compile_backbone_migraphx.py \
  --onnx migraphx_infer/artifacts/backbone/vision_projector_fp16.onnx \
  --output migraphx_infer/artifacts/backbone/vision_projector_fp16_sync.mxr

python migraphx_infer/scripts/compile/compile_backbone_migraphx.py \
  --onnx migraphx_infer/artifacts/backbone/language_transformer_fp16.onnx \
  --output migraphx_infer/artifacts/backbone/language_transformer_fp16_sync.mxr
```

Vision的`--opt_compile`已验证没有收益，正式流程不使用该参数。

## 5. 生成async MXR

### 5.1 编译同步删除工具

源码：

```text
migraphx_infer/scripts/async/remove_migraphx_sync.cpp
```

编译命令：

```bash
c++ -std=c++17 -O2 \
  migraphx_infer/scripts/async/remove_migraphx_sync.cpp \
  -I/opt/dtk/include \
  -L/opt/dtk/lib \
  -Wl,-rpath,/opt/dtk/lib \
  -lmigraphx \
  -o migraphx_infer/scripts/async/remove_migraphx_sync
```

必须链接`/opt/dtk/lib`。自编译源码库生成的MXR序列化版本与DTK内置库不同，不能和正式Vision/Language MXR混用。

### 5.2 删除三个MXR内部同步

```bash
migraphx_infer/scripts/async/remove_migraphx_sync \
  migraphx_infer/artifacts/dit/dit_fp16_vl116_sync.mxr \
  migraphx_infer/artifacts/dit/dit_fp16_vl116_async.mxr

migraphx_infer/scripts/async/remove_migraphx_sync \
  migraphx_infer/artifacts/backbone/vision_projector_fp16_sync.mxr \
  migraphx_infer/artifacts/backbone/vision_projector_fp16_async.mxr

migraphx_infer/scripts/async/remove_migraphx_sync \
  migraphx_infer/artifacts/backbone/language_transformer_fp16_sync.mxr \
  migraphx_infer/artifacts/backbone/language_transformer_fp16_async.mxr
```

wrapper使用当前PyTorch stream调用`run_async`，同一stream保证Vision、Language、DiT和后续PyTorch算子的先后依赖。计时结束处仍然会统一执行`torch.cuda.synchronize()`，因此删除组件内部同步不会漏计时。

## 6. 精度验证

### 6.1 DiT同步/async逐元素比较

```bash
python migraphx_infer/scripts/validate/compare_dit_engines.py \
  --baseline migraphx_infer/artifacts/dit/dit_fp16_vl116_sync.mxr \
  --candidate migraphx_infer/artifacts/dit/dit_fp16_vl116_async.mxr
```

### 6.2 Vision和Language同步/async逐元素比较

```bash
python migraphx_infer/scripts/validate/compare_backbone_engines.py \
  --baseline migraphx_infer/artifacts/backbone/vision_projector_fp16_sync.mxr \
  --candidate migraphx_infer/artifacts/backbone/vision_projector_fp16_async.mxr \
  --reference migraphx_infer/artifacts/backbone/vision_reference.pt

python migraphx_infer/scripts/validate/compare_backbone_engines.py \
  --baseline migraphx_infer/artifacts/backbone/language_transformer_fp16_sync.mxr \
  --candidate migraphx_infer/artifacts/backbone/language_transformer_fp16_async.mxr \
  --reference migraphx_infer/artifacts/backbone/language_reference.pt
```

三项预期结果均为：

```text
MAE: 0
RMSE: 0
Max abs: 0
Cosine: 1
```

### 6.3 最终动作精度

```bash
numactl -N 3 -m 3 \
  python migraphx_infer/scripts/validate/validate_three_mxr_action.py
```

该脚本比较相同输入、相同扩散随机种子下PyTorch和三个MIGraphX引擎的Backbone输出与最终Action。

## 7. 单组件性能验证

DiT：

```bash
MIGRAPHX_ENGINE=migraphx_infer/artifacts/dit/dit_fp16_vl116_async.mxr \
MIGRAPHX_WARMUP=5 MIGRAPHX_ITERATIONS=10 \
numactl -N 3 -m 3 \
python migraphx_infer/scripts/validate/benchmark_dit_migraphx.py
```

Vision：

```bash
numactl -N 3 -m 3 \
python migraphx_infer/scripts/validate/benchmark_vision_migraphx.py \
  --engine migraphx_infer/artifacts/backbone/vision_projector_fp16_async.mxr \
  --reference migraphx_infer/artifacts/backbone/vision_reference.pt \
  --warmup 5 --iterations 10
```

Language：

```bash
numactl -N 3 -m 3 \
python migraphx_infer/scripts/validate/benchmark_language_migraphx.py \
  --engine migraphx_infer/artifacts/backbone/language_transformer_fp16_async.mxr \
  --reference migraphx_infer/artifacts/backbone/language_reference.pt \
  --warmup 5 --iterations 10
```

当前中位数：DiT 13.7946ms、Vision 17.5464ms、Language 11.1700ms。

## 8. 完整模型运行和性能测试

### 8.1 只测试正式MIGraphX方案

```bash
numactl -N 3 -m 3 \
python scripts/deployment/benchmark_inference.py \
  --model_path nvidia/GR00T-N1.6-3B \
  --dataset_path demo_data/gr1.PickNPlace \
  --embodiment_tag gr1 \
  --video_backend decord \
  --migraphx_engine_path migraphx_infer/artifacts/dit/dit_fp16_vl116_async.mxr \
  --migraphx_vision_engine_path migraphx_infer/artifacts/backbone/vision_projector_fp16_async.mxr \
  --migraphx_language_engine_path migraphx_infer/artifacts/backbone/language_transformer_fp16_async.mxr \
  --backbone_sdpa \
  --skip_compile \
  --num_iterations 10 \
  --warmup 5
```

### 8.2 同时和torch.compile比较

去掉上面命令中的`--skip_compile`，并增加：

```text
--compile_backbone
```

这样会依次测试PyTorch Eager、torch.compile DiT、torch.compile DiT+Backbone和MIGraphX。首次编译需要较长时间。

## 9. 脚本用途对照

| 脚本 | 用途 | 主要输入 | 主要输出 |
|---|---|---|---|
| `scripts/export/export_dit_onnx.py` | 导出静态FP16 DiT | GR00T模型 | DiT ONNX |
| `scripts/compile/compile_dit_migraphx.py` | 编译DiT | DiT ONNX | DiT同步MXR |
| `scripts/export/export_backbone_onnx.py` | 分别导出Vision/Language | 模型、数据集、固定prompt | Backbone ONNX和参考输入 |
| `scripts/compile/compile_backbone_migraphx.py` | 编译Vision/Language | Backbone ONNX | Backbone同步MXR |
| `scripts/async/remove_migraphx_sync.cpp` | 删除MXR末尾同步 | 同步MXR | async MXR |
| `scripts/validate/compare_dit_engines.py` | 验证DiT async改写 | 新旧DiT MXR | 逐元素误差 |
| `scripts/validate/compare_backbone_engines.py` | 验证Backbone async改写 | 新旧MXR、参考输入 | 逐元素误差 |
| `scripts/validate/validate_three_mxr_action.py` | 验证完整动作精度 | 三个async MXR | Backbone/Action误差 |
| `scripts/validate/benchmark_*_migraphx.py` | 单组件测速 | 对应MXR | 延迟统计 |
| `scripts/analyze/inspect_migraphx_program.py` | 统计编译图算子 | MXR | 指令和算子统计 |
| `scripts/analyze/inspect_onnx_graph.py` | 检查ONNX算子 | ONNX | 算子统计 |

## 10. 未采用的实验

以下方案已经验证但没有进入正式流程，其大体积MXR已删除，只保留日志和历史结论：

| 实验 | 不采用原因 |
|---|---|
| 关闭reduce fusion | 会失去图优化，正式方案保持reduce fusion开启 |
| hipBLASLt GELU_BIAS | tanh-approx GELU语义不等价，且性能下降 |
| GEMM+bias+residual融合 | cosine约0.9613，性能下降34.43% |
| LayerNorm融合 | 性能下降约0.88% |
| MIOpen SkipLayerNorm | 精度正确但性能下降约3.44% |
| DiT Attention融合 | 快约4%，但cosine约0.8813 |
| Vision MHA改写 | 三输入MHA数值错误，packed-bias发生VMFault |
| PyTorch HIP Graph捕获MIGraphX | code object启动不能被当前捕获路径支持 |

## 11. 静态shape限制

当前MXR不是通用动态shape引擎。`VL Length=116`由81个Vision token、对话模板、特殊token和当前任务文本共同组成。修改任务文本、图片数量、分辨率或相机数量后，输入长度可能变化，现有MXR会直接报shape mismatch。

需要支持其他场景时，应为新的固定长度重新导出和编译，或者设计多个shape profile按实际长度路由。不能简单截断文本，否则可能丢失任务语义。

## 12. 日志位置

最终结果：

```text
migraphx_infer/results/final/all_async_e2e_5x10.log
migraphx_infer/results/final/system_sync_async_e2e_5x10.log
migraphx_infer/results/final/backbone_async_accuracy.log
migraphx_infer/results/final/backbone_async_component_benchmark.log
```

历史基线：

```text
migraphx_infer/results/baseline/
```

完整优化记录：

```text
migraphx_infer/docs/optimization_history.md
```

## 13. 文件整理记录

2026年7月17日对130机器上的`migraphx_infer`进行了整理：

| 项目 | 整理前 | 整理后 |
|---|---:|---:|
| 磁盘占用 | 46 GB | 14 GB |
| 顶层文件 | 78个 | README和5个分类目录 |

本次删除约32GB，主要包括：

- Attention、LayerNorm、SkipLayerNorm、GELU、bias+residual等失败实验生成的2.1GB级MXR；
- 旧FP32 ONNX、VL116补丁ONNX和重复zero-copy MXR；
- Vision MHA失败图及其外部权重；
- 自编译MIGraphX展开库和多个备份库；
- `benchmark_inference.py`的副本、`.bak`和Python缓存。

失败实验的结论、小型日志和分析脚本仍保存在`experiments`和`docs/optimization_history.md`中。

整理后使用新目录中的三个async MXR重新执行完整5/10回归：

| Data Processing | Backbone | Action Head | E2E | 频率 |
|---:|---:|---:|---:|---:|
| 4.43 ms | 30.30 ms | 58.72 ms | 93.5 ms | 10.7 Hz |

回归日志：

```text
migraphx_infer/results/final/post_cleanup_e2e_5x10.log
```
