# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory for creating bench executors."""

from rbg_profiler.bench.executor import BenchExecutor
from rbg_profiler.bench.http_executor import HTTPBenchExecutor
from rbg_profiler.bench.inference_perf_executor import InferencePerfExecutor


def create_bench_executor(method: str, backend: str, **kwargs) -> BenchExecutor:
    """Create a bench executor for the given method and backend.

    Args:
        method: Bench execution method — "http" or "inference-perf".
        backend: Engine/API backend — "sglang-oai", "vllm-chat", etc.
            For "http": selects the HTTP bench backend (API format).
            For "inference-perf": maps to server.type + api.type.
    """
    if method == "http":
        from rbg_profiler.bench import create_bench_backend
        bench_backend = create_bench_backend(backend)
        return HTTPBenchExecutor(bench_backend)
    elif method == "inference-perf":
        return InferencePerfExecutor(backend=backend, **kwargs)
    else:
        raise ValueError(
            f"Unknown bench method: {method}. Supported: ['http', 'inference-perf']"
        )
