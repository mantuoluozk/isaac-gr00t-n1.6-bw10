#!/usr/bin/env python3
"""Closed-loop concurrent PolicyClient benchmark for one GR00T PolicyServer."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import zmq

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.server_client import MsgSerializer, PolicyClient


DEFAULT_PROMPT = "pick the pear from the counter and place it in the plate"


def make_observation(host, port, dataset_path, video_backend, prompt, timeout_ms):
    client = PolicyClient(host=host, port=port, timeout_ms=timeout_ms, strict=False)
    client.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    client.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    modality_config = client.get_modality_config()
    dataset = LeRobotEpisodeLoader(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        video_backend=video_backend,
    )
    step = extract_step_data(
        dataset[0],
        step_index=0,
        modality_configs=modality_config,
        embodiment_tag=EmbodimentTag("gr1"),
        allow_padding=False,
    )
    observation = {
        "video": {key: np.stack(step.images[key])[None] for key in step.images},
        "state": {key: step.states[key][None] for key in step.states},
        "language": {modality_config["language"].modality_keys[0]: [[prompt]]},
    }
    client.socket.close(linger=0)
    client.context.term()
    return observation


def client_worker(
    client_id,
    host,
    port,
    observation,
    warmup,
    iterations,
    timeout_ms,
    ready_barrier,
    measure_barrier,
    result_queue,
):
    client = None
    try:
        client = PolicyClient(host=host, port=port, timeout_ms=timeout_ms, strict=False)
        client.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        client.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        if not client.ping():
            raise RuntimeError("PolicyServer ping failed")

        ready_barrier.wait()
        for _ in range(warmup):
            client.get_action(observation)
        measure_barrier.wait()

        latencies_ms = []
        start_ns = time.perf_counter_ns()
        for _ in range(iterations):
            request_start_ns = time.perf_counter_ns()
            client.get_action(observation)
            latencies_ms.append((time.perf_counter_ns() - request_start_ns) / 1e6)
        end_ns = time.perf_counter_ns()
        result_queue.put(
            {
                "client_id": client_id,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "latencies_ms": latencies_ms,
            }
        )
    except Exception as exc:
        result_queue.put({"client_id": client_id, "error": repr(exc)})
        try:
            ready_barrier.abort()
            measure_barrier.abort()
        except Exception:
            pass
    finally:
        if client is not None:
            client.socket.close(linger=0)
            client.context.term()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def run_scenario(args, observation, concurrency, ports=None):
    context = mp.get_context("fork")
    ready_barrier = context.Barrier(concurrency + 1)
    measure_barrier = context.Barrier(concurrency + 1)
    result_queue = context.Queue()
    processes = []
    for client_id in range(concurrency):
        worker_port = ports[client_id] if ports is not None else args.port
        process = context.Process(
            target=client_worker,
            args=(
                client_id,
                args.host,
                worker_port,
                observation,
                args.warmup,
                args.iterations,
                args.timeout_ms,
                ready_barrier,
                measure_barrier,
                result_queue,
            ),
        )
        process.start()
        processes.append(process)

    ready_barrier.wait(timeout=300)
    print(
        f"[{concurrency} clients] all clients ready; "
        f"running {args.warmup} warmup requests/client...",
        flush=True,
    )
    measure_barrier.wait(timeout=1800)
    print(
        f"[{concurrency} clients] measuring {args.iterations} requests/client...",
        flush=True,
    )

    raw_results = [result_queue.get(timeout=1800) for _ in range(concurrency)]
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join()

    errors = [result for result in raw_results if "error" in result]
    if errors:
        raise RuntimeError(f"Client errors: {errors}")

    raw_results.sort(key=lambda item: item["client_id"])
    global_start_ns = min(item["start_ns"] for item in raw_results)
    global_end_ns = max(item["end_ns"] for item in raw_results)
    wall_seconds = (global_end_ns - global_start_ns) / 1e9
    total_requests = concurrency * args.iterations
    aggregate_throughput = total_requests / wall_seconds

    all_latencies = [
        latency
        for result in raw_results
        for latency in result["latencies_ms"]
    ]
    clients = []
    client_throughputs = []
    for result in raw_results:
        duration_seconds = (result["end_ns"] - result["start_ns"]) / 1e9
        client_throughput = args.iterations / duration_seconds
        client_throughputs.append(client_throughput)
        latencies = result["latencies_ms"]
        clients.append(
            {
                "client_id": result["client_id"],
                "median_ms": percentile(latencies, 50),
                "mean_ms": float(np.mean(latencies)),
                "p95_ms": percentile(latencies, 95),
                "min_ms": float(np.min(latencies)),
                "max_ms": float(np.max(latencies)),
                "throughput_hz": client_throughput,
            }
        )

    summary = {
        "concurrency": concurrency,
        "ports": list(ports) if ports is not None else [args.port],
        "warmup_per_client": args.warmup,
        "iterations_per_client": args.iterations,
        "total_measured_requests": total_requests,
        "wall_seconds": wall_seconds,
        "aggregate_throughput_hz": aggregate_throughput,
        "overall_latency": {
            "median_ms": percentile(all_latencies, 50),
            "mean_ms": float(np.mean(all_latencies)),
            "p95_ms": percentile(all_latencies, 95),
            "min_ms": float(np.min(all_latencies)),
            "max_ms": float(np.max(all_latencies)),
        },
        "fairness_max_min_throughput": max(client_throughputs) / min(client_throughputs),
        "clients": clients,
    }
    print(
        f"[{concurrency} clients] aggregate={aggregate_throughput:.2f} req/s, "
        f"latency median={summary['overall_latency']['median_ms']:.2f} ms, "
        f"p95={summary['overall_latency']['p95_ms']:.2f} ms, "
        f"fairness={summary['fairness_max_min_throughput']:.3f}",
        flush=True,
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--ports",
        type=int,
        nargs="+",
        default=None,
        help="Connect one client to each listed server port for multi-instance testing.",
    )
    parser.add_argument("--dataset-path", default="demo_data/gr1.PickNPlace")
    parser.add_argument("--video-backend", default="decord")
    parser.add_argument("--language-prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--clients", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=300000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if any(value < 1 for value in args.clients):
        parser.error("--clients values must be positive")
    if args.ports and any(value < 1 or value > 65535 for value in args.ports):
        parser.error("--ports values must be valid TCP ports")
    print("Preparing one fixed real-data observation...", flush=True)
    primary_port = args.ports[0] if args.ports else args.port
    observation = make_observation(
        args.host,
        primary_port,
        args.dataset_path,
        args.video_backend,
        args.language_prompt,
        args.timeout_ms,
    )
    request_bytes = len(
        MsgSerializer.to_bytes(
            {"endpoint": "get_action", "data": {"observation": observation, "options": None}}
        )
    )
    print(f"Serialized request size: {request_bytes / 1024:.1f} KiB", flush=True)

    output = {
        "host": args.host,
        "port": primary_port,
        "ports": args.ports,
        "dataset_path": args.dataset_path,
        "request_size_bytes": request_bytes,
        "measurement_model": (
            "closed-loop clients using independent server instances"
            if args.ports
            else "closed-loop clients sharing one synchronous REP server"
        ),
        "scenarios": [],
    }
    if args.ports:
        output["scenarios"].append(
            run_scenario(args, observation, len(args.ports), ports=args.ports)
        )
    else:
        for concurrency in args.clients:
            output["scenarios"].append(run_scenario(args, observation, concurrency))

    rendered = json.dumps(output, indent=2, ensure_ascii=False)
    print(rendered, flush=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
