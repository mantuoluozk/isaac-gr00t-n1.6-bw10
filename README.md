# Isaac-GR00T N1.6 on Hygon BW10

本仓库是 Isaac-GR00T N1.6-3B 在海光 BW10 上的部署、性能测试和 MIGraphX 优化源码副本。项目保留上游 GR00T 代码，并增加 BW10 适配所需的 ONNX 导出、MIGraphX 编译、Device Pointer 零拷贝、服务端混合后端和并发压测工具。

> 本仓库仅包含源码和文档。模型权重、数据集、ONNX、MXR、编译缓存和原始日志不纳入 Git。

## 许可证与使用范围

项目保留根目录 [LICENSE](LICENSE) 中的 NVIDIA License。该许可证包含非商业研究用途限制，使用和再分发前请完整阅读其条款。本 README 不替代许可证文本。

## 已实现功能

- GR00T-N1.6-3B PyTorch Eager 和 `torch.compile` 基线测试。
- DiT 直接 FP16 ONNX 导出与 MIGraphX MXR 编译。
- Eagle Vision Projector 和 Language Transformer 的 ONNX/MXR 转换。
- MIGraphX Device Pointer 零拷贝输入输出。
- 删除 MXR 返回前不必要的 `hip::sync_stream`。
- `torch.compile DiT + MIGraphX Vision/Language` 混合推理。
- 服务端混合后端启动参数。
- 单服务多客户端和多实例真并发压测。

## 当前性能

固定条件：BW10、batch=1、Action Horizon=50、4 个 denoising steps、warmup=5、10 次稳态测量、服务端 NUMA 3。

| 模式 | 统计口径 | 延迟 | 吞吐/频率 |
|---|---|---:|---:|
| PyTorch Eager | 模型分项相加 | 176.8 ms | 5.7 Hz |
| torch.compile DiT + MIGraphX Backbone | 模型分项相加 | **88.7 ms** | **11.3 Hz** |
| 单服务实例 | 客户端完整墙钟 | 94.04 ms | 10.62 req/s |
| 两个完整实例 | 两客户端真并发 | 131.76 ms | 15.03 req/s |
| 三个完整实例 | 三客户端真并发 | 184.66 ms | 16.18 req/s |
| 四个完整实例 | 加载阶段 | - | HIP OOM |

`11.3 Hz` 是 Data Processing、Backbone 和 Action Head 分项测量后相加的模型内部口径。`10.62 req/s` 还包含 msgpack、ZeroMQ、strict 检查、DCU 到 CPU 拷贝、动作反归一化和结果返回，两个数据不是同一统计边界。

## 目录

```text
gr00t/                              上游模型、数据与服务代码
scripts/deployment/                 基线测试和推理后端 wrapper
migraphx_infer/
├── README.md                      MIGraphX 完整复现文档
├── docs/                          优化历史和原理分析
├── scripts/export/                ONNX 导出
├── scripts/compile/               MXR 编译
├── scripts/async/                 删除内部同步
├── scripts/validate/              精度和性能验证
├── scripts/analyze/               图结构和 profile
├── scripts/benchmark/             并发客户端压测
└── experiments/scripts/           未进入正式方案的实验代码
```

MIGraphX 的导出、编译、精度验证和引擎生成步骤见 [migraphx_infer/README.md](migraphx_infer/README.md)。

## 环境

已验证环境：

```text
Device: Hygon BW10
CPU: Hygon C86-4G
Python: 3.10
PyTorch: 2.9.0
MIGraphX: 5.2.0
```

容器内公共环境变量：

```bash
cd /workspace/isaac-gr00t-n1.6

export PYTHONPATH=/opt/dtk/lib:${PYTHONPATH}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
```

## 准备模型和数据

代码仓库不包含以下内容：

```text
nvidia/GR00T-N1.6-3B 模型权重
demo_data/gr1.PickNPlace 数据集
migraphx_infer/artifacts 中的 ONNX/MXR/参考 Tensor
```

运行前需将数据放到 `demo_data/gr1.PickNPlace`，并确保 Hugging Face 缓存或本地目录中存在模型。如需从头生成引擎，按 [MIGraphX 部署说明](migraphx_infer/README.md) 执行。

## 跑单路最优基准

```bash
numactl -N 3 -m 3 \
python -u scripts/deployment/benchmark_inference.py \
  --model_path nvidia/GR00T-N1.6-3B \
  --dataset_path demo_data/gr1.PickNPlace \
  --embodiment_tag gr1 \
  --video_backend decord \
  --migraphx_vision_engine_path migraphx_infer/artifacts/backbone/vision_projector_fp16_async.mxr \
  --migraphx_language_engine_path migraphx_infer/artifacts/backbone/language_transformer_fp16_async.mxr \
  --compile_dit_migraphx_backbone \
  --backbone_sdpa \
  --skip_compile \
  --num_iterations 10 \
  --warmup 5
```

## 启动最优混合服务

```bash
numactl -N 3 -m 3 \
python -u gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.6-3B \
  --embodiment-tag GR1 \
  --host 0.0.0.0 \
  --port 5555 \
  --compile-action-head \
  --migraphx-vision-engine-path migraphx_infer/artifacts/backbone/vision_projector_fp16_async.mxr \
  --migraphx-language-engine-path migraphx_infer/artifacts/backbone/language_transformer_fp16_async.mxr
```

## 多实例并发压测

每个完整实例使用独立端口。MIGraphX 5.2.0 并发冷加载多份 MXR 时可能出现 `named symbol not found`，应逐个启动并等待上一个实例 ready。

两实例启动后执行：

```bash
numactl -N 2 -m 2 \
python -u migraphx_infer/scripts/benchmark/benchmark_concurrent_clients.py \
  --host 127.0.0.1 \
  --ports 5555 5556 \
  --dataset-path demo_data/gr1.PickNPlace \
  --video-backend decord \
  --warmup 5 \
  --iterations 10 \
  --output migraphx_infer/results/multi_instance/clients_2inst.json
```

## 源码仓库不包含的文件

`.gitignore` 默认排除：

- `checkpoints/`、`demo_data/`和模型权重；
- `migraphx_infer/artifacts/`中的 ONNX、MXR 和参考 Tensor；
- `migraphx_infer/results/`、profile、编译日志和 PID 文件；
- Python 缓存、虚拟环境和本地 IDE 配置；
- 编译生成的可执行文件。

## 已知限制

- 当前 MXR 固定 batch=1、VL Length=116。
- 一个 `PolicyServer` 是同步 ZeroMQ REP 服务，多客户端会排队。
- 两个完整实例可真并发；三实例收益很低；32 GiB BW10 无法容纳四个完整实例。
- MIGraphX wrapper 中的预分配输出缓冲区不能在多线程间直接共享。
- 四路真并发需要单进程多执行上下文或 batch=2/4 引擎，不能通过复制四份完整模型实现。
