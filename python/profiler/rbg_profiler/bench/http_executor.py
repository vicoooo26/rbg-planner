# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP-based bench executor.

Wraps the existing BaseBenchBackend.run_benchmark() to conform to the
BenchExecutor interface. The bench_backend determines the API format
(sglang completions, sglang chat, vllm, vllm chat).
"""

from rbg_profiler.bench.base import BaseBenchBackend, BenchMetrics
from rbg_profiler.bench.executor import BenchExecutor, BenchParams, BenchResult


class HTTPBenchExecutor(BenchExecutor):
    """HTTP-based bench using the existing BaseBenchBackend."""

    def __init__(self, bench_backend: BaseBenchBackend):
        self.bench = bench_backend

    async def execute(self, params: BenchParams) -> BenchResult:
        metrics: BenchMetrics = await self.bench.run_benchmark(
            endpoint=params.endpoint,
            model=params.model,
            num_prompts=params.num_prompts,
            isl=params.isl,
            osl=params.osl,
            request_rate=params.request_rate,
        )
        return BenchResult(
            mean_ttft_ms=metrics.mean_ttft_ms,
            mean_itl_ms=metrics.mean_itl_ms,
            mean_tpot_ms=metrics.mean_tpot_ms,
            request_throughput=metrics.request_throughput,
            output_throughput=metrics.output_throughput,
            mean_e2e_latency_ms=metrics.mean_e2e_latency_ms,
        )
