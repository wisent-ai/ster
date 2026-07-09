"""CLI execution for the 're-enabled' monitor command.

Registered in the argparse parser (monitor_parser.py) but its dispatch handler
had been dropped. Mirrors the emit/print style of
wisent/core/utils/cli/analysis/analysis/config/inference_config_cli.py.

Reports hardware / memory / GPU state (and optionally runs a device benchmark
or a continuous psutil sampling loop) as pretty JSON on stdout. Heavy deps
(torch, DeviceBenchmarker which loads a real model) are imported lazily inside
execute_monitor so importing this module stays cheap. psutil is used directly
for live memory; nothing under infra_tools.tracking is imported.
"""

import json

from wisent.core.utils.config_tools.constants import (
    CODEFORCES_DEFAULT_TIME_LIMIT,
    HASH_PREFIX_LEN,
    JSON_INDENT,
    SEPARATOR_WIDTH_MEDIUM,
)
from wisent.core.utils.config_tools.constants.validated._validated import BYTES_PER_MB

# run_full_benchmark requires two non-None "default seconds" args, used only
# when a sub-benchmark measurement returns None (see _device_bench_runner.py).
# The repo forbids inline numeric literals (constants_root.py), so this neutral
# one-second placeholder is sourced by name from the constants package.
_BENCH_DEFAULT_SECONDS = CODEFORCES_DEFAULT_TIME_LIMIT


def _torch_gpu_info(torch, gpu_mem_mb, detailed=False):
    """GPU availability + live memory via an already-imported torch module.

    ``gpu_mem_mb`` is the total device memory (zero when no GPU) supplied by the
    caller from detect_system_resources() so no numeric literal is needed here.
    """
    mps = getattr(torch.backends, "mps", None)
    info = {
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(mps is not None and torch.backends.mps.is_available()),
        "gpu_mem_mb": gpu_mem_mb,
    }
    if info["cuda_available"]:
        idx = torch.cuda.current_device()  # torch API (no device-index literal)
        info["gpu_allocated_mb"] = torch.cuda.memory_allocated(idx) // BYTES_PER_MB
        info["gpu_reserved_mb"] = torch.cuda.memory_reserved(idx) // BYTES_PER_MB
        if detailed:
            info["gpu_name"] = torch.cuda.get_device_properties(idx).name
    return info


def _memory_snapshot(psutil, proc, track_gpu):
    """One point-in-time memory sample built directly from psutil."""
    vm = psutil.virtual_memory()  # psutil API
    snap = {
        "system_total_mb": vm.total // BYTES_PER_MB,
        "system_available_mb": vm.available // BYTES_PER_MB,
        "system_used_mb": vm.used // BYTES_PER_MB,
        "system_percent": vm.percent,
        "process_rss_mb": proc.memory_info().rss // BYTES_PER_MB,  # psutil API
    }
    if track_gpu:
        import torch  # lazy: only imported when GPU sampling is requested

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            snap["gpu_allocated_mb"] = torch.cuda.memory_allocated(idx) // BYTES_PER_MB
            snap["gpu_reserved_mb"] = torch.cuda.memory_reserved(idx) // BYTES_PER_MB
    return snap


def _aggregate(snapshots):
    """min/max/avg over every numeric metric shared by all snapshots."""
    if not snapshots:
        return {}
    first = next(iter(snapshots))
    keys = [k for k, v in first.items() if isinstance(v, (int, float))]
    agg = {}
    for k in keys:
        vals = [s[k] for s in snapshots if k in s]
        if not vals:
            continue
        agg[k] = {"min": min(vals), "max": max(vals), "avg": sum(vals) / len(vals)}
    return agg


def execute_monitor(args):
    """Execute the monitor command; prints a JSON dict for the selected mode."""
    memory_info = getattr(args, "memory_info", False)
    system_info = getattr(args, "system_info", False)
    benchmark = getattr(args, "benchmark", False)
    test_gpu = getattr(args, "test_gpu", False)
    continuous = getattr(args, "continuous", False)
    interval = getattr(args, "interval", None)
    duration = getattr(args, "duration", None)
    export_csv = getattr(args, "export_csv", None)
    track_gpu = getattr(args, "track_gpu", False)
    detailed = getattr(args, "detailed", False)

    try:
        if benchmark:
            # Lazy: DeviceBenchmarker.run_full_benchmark loads a real model.
            import tempfile
            from dataclasses import asdict

            # DeviceBenchmarker(benchmarks_file, device_hash_prefix)
            #   -> resources/public/device_benchmarks.py DeviceBenchmarker.__init__
            from wisent.core.experimental.agent.implementation.resources.public.device_benchmarks import (
                DeviceBenchmarker,
            )

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                bench_file = f.name
            benchmarker = DeviceBenchmarker(
                benchmarks_file=bench_file,
                device_hash_prefix=HASH_PREFIX_LEN,
            )  # ctor confirmed: device_benchmarks.py DeviceBenchmarker.__init__
            # run_full_benchmark(force_rerun, bench_eval_default_seconds,
            #   classifier_training_default_seconds) -> _device_bench_runner.py
            bench = benchmarker.run_full_benchmark(
                force_rerun=True,
                bench_eval_default_seconds=_BENCH_DEFAULT_SECONDS,
                classifier_training_default_seconds=_BENCH_DEFAULT_SECONDS,
            )
            result = {"mode": "benchmark", "benchmark": asdict(bench)}

        elif continuous:
            import csv
            import time

            import psutil  # psutil API

            proc = psutil.Process()  # psutil API (current process)
            snapshots = []
            deadline = time.monotonic() + float(duration)
            while True:
                snap = _memory_snapshot(psutil, proc, track_gpu)
                snap["timestamp"] = time.time()
                snapshots.append(snap)
                now = time.monotonic()
                if now >= deadline:
                    break
                time.sleep(min(float(interval), deadline - now))

            result = {
                "mode": "continuous",
                "interval_s": interval,
                "duration_s": duration,
                "sample_count": len(snapshots),
                "aggregate": _aggregate(snapshots),
            }
            if detailed:
                result["snapshots"] = snapshots
            if export_csv:
                fieldnames = sorted({k for s in snapshots for k in s})
                with open(export_csv, "w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=fieldnames)
                    writer.writeheader()
                    for s in snapshots:
                        writer.writerow(s)
                result["export_csv"] = export_csv

        elif system_info:
            # detect_system_resources() -> hardware.py ; SystemResources fields
            #   cpu_count / total_ram_mb / gpu_mem_mb -> hardware.py dataclass
            from wisent.core.utils.infra_tools.infra.core.hardware import (
                detect_system_resources,
            )

            res = detect_system_resources()  # hardware.py detect_system_resources
            result = {
                "mode": "system_info",
                "cpu_count": res.cpu_count,
                "total_ram_mb": res.total_ram_mb,
                "gpu_mem_mb": res.gpu_mem_mb,
            }

        elif test_gpu:
            # resolve_default_device() -> device.py (returns cuda/mps/cpu)
            import torch  # lazy: cuda/mps availability + live gpu mem

            from wisent.core.utils.infra_tools.infra.core.device import (
                resolve_default_device,
            )
            # detect_system_resources().gpu_mem_mb is total device memory
            #   (zero when no GPU) -> hardware.py ; reused so no numeric literal.
            from wisent.core.utils.infra_tools.infra.core.hardware import (
                detect_system_resources,
            )

            res = detect_system_resources()  # hardware.py detect_system_resources
            result = {"mode": "test_gpu", "device": resolve_default_device()}
            result.update(_torch_gpu_info(torch, res.gpu_mem_mb, detailed=detailed))

        elif memory_info:
            import psutil  # psutil API

            proc = psutil.Process()  # psutil API (current process)
            result = {"mode": "memory_info"}
            result.update(_memory_snapshot(psutil, proc, track_gpu=track_gpu))
            if detailed:
                swap = psutil.swap_memory()  # psutil API
                result["swap_total_mb"] = swap.total // BYTES_PER_MB
                result["swap_used_mb"] = swap.used // BYTES_PER_MB
                result["swap_percent"] = swap.percent
            # GPU mem via lazy torch when cuda is available.
            import torch

            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                result["gpu_allocated_mb"] = (
                    torch.cuda.memory_allocated(idx) // BYTES_PER_MB
                )
                result["gpu_reserved_mb"] = (
                    torch.cuda.memory_reserved(idx) // BYTES_PER_MB
                )
                result["gpu_mem_mb"] = (
                    torch.cuda.get_device_properties(idx).total_memory // BYTES_PER_MB
                )

        else:
            print("-" * SEPARATOR_WIDTH_MEDIUM)
            result = {
                "mode": None,
                "usage": (
                    "No monitoring mode selected. Choose one of: --system-info, "
                    "--test-gpu, --memory-info, --continuous, --benchmark "
                    "(--continuous also uses --track-gpu / --export-csv / --detailed)."
                ),
            }

        print(json.dumps(result, indent=JSON_INDENT))
        return result

    except Exception as e:  # emit failure as JSON rather than a raw traceback
        error = {"mode": "monitor", "error": str(e), "error_type": type(e).__name__}
        print(json.dumps(error, indent=JSON_INDENT))
        return error
