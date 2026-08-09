# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for profiler integration with bench executors."""

from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rbg_profiler.bench.executor import BenchResult
from rbg_profiler.rbg_discoverer import RoleEndpoint


@pytest.fixture
def mock_endpoint():
    return RoleEndpoint(
        base_url="http://test:8000",
        num_gpus=1,
        tokenizer="test-tokenizer",
    )


@pytest.fixture
def mock_args():
    return Namespace(
        model_name="test-model",
        max_kv_tokens=32768,
        max_context_length=32768,
        output_dir="/tmp/test_profiling",
        prefill_isl_points="128,256",
        decode_isl_points="256",
        decode_granularity=3,
        output_configmap="test-cm",
        namespace="default",
        update_configmap=False,
        bench_method="http",
    )


@pytest.fixture
def mock_bench_result():
    return BenchResult(
        mean_ttft_ms=50.0,
        mean_itl_ms=15.0,
        mean_tpot_ms=15.0,
        request_throughput=10.0,
        output_throughput=500.0,
        mean_e2e_latency_ms=100.0,
    )


@pytest.mark.asyncio
async def test_run_profiling_uses_pipeline(mock_endpoint, mock_args, mock_bench_result):
    """run_profiling should use ProfilingPipeline with the executor."""
    from rbg_profiler.profiler import run_profiling

    mock_executor = MagicMock()
    mock_executor.execute = AsyncMock(return_value=mock_bench_result)

    with patch("rbg_profiler.profiler.create_bench_executor", return_value=mock_executor):
        await run_profiling(
            mock_args, mock_endpoint, mock_endpoint, backend="sglang-oai"
        )

    # Executor should have been called for each sweep point
    # Prefill: 2 ISL points (128, 256)
    # Decode: 1 ISL point (256) × 3 concurrency points [1, 22, 43]
    assert mock_executor.execute.call_count == 5


@pytest.mark.asyncio
async def test_run_profiling_backward_compat_http(mock_endpoint, mock_args):
    """With bench_method=http, profiler should use HTTPBenchExecutor internally."""
    from rbg_profiler.profiler import run_profiling

    # Mock the executor to return valid results so the profiler completes
    mock_executor = MagicMock()
    mock_executor.execute = AsyncMock(return_value=BenchResult(
        mean_ttft_ms=50.0, mean_itl_ms=15.0, mean_tpot_ms=15.0,
        request_throughput=10.0, output_throughput=500.0,
        mean_e2e_latency_ms=100.0,
    ))

    with patch("rbg_profiler.profiler.create_bench_executor", return_value=mock_executor):
        # Should not raise
        await run_profiling(
            mock_args, mock_endpoint, mock_endpoint, backend="sglang-oai"
        )

    mock_executor.execute.assert_called()
