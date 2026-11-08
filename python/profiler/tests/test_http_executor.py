# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for HTTPBenchExecutor."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from rbg_profiler.bench.base import BenchMetrics, ContextLengthExceededError
from rbg_profiler.bench.executor import BenchExecutor, BenchParams, BenchResult
from rbg_profiler.bench.http_executor import HTTPBenchExecutor
from rbg_profiler.rbg_discoverer import RoleEndpoint


@pytest.fixture
def mock_bench_backend():
    backend = MagicMock()
    backend.run_benchmark = AsyncMock()
    return backend


@pytest.fixture
def stub_params():
    return BenchParams(
        endpoint=RoleEndpoint(base_url="http://test:8000", num_gpus=2, tokenizer="t"),
        model="test-model",
        isl=1024,
        osl=5,
        num_prompts=5,
    )


@pytest.fixture
def mock_metrics():
    return BenchMetrics(
        mean_ttft_ms=50.0,
        mean_itl_ms=15.0,
        mean_tpot_ms=15.0,
        request_throughput=10.0,
        output_throughput=500.0,
        mean_e2e_latency_ms=100.0,
    )


def test_http_executor_is_bench_executor():
    assert issubclass(HTTPBenchExecutor, BenchExecutor)


@pytest.mark.asyncio
async def test_execute_calls_run_benchmark(mock_bench_backend, stub_params, mock_metrics):
    mock_bench_backend.run_benchmark.return_value = mock_metrics
    executor = HTTPBenchExecutor(mock_bench_backend)

    result = await executor.execute(stub_params)

    mock_bench_backend.run_benchmark.assert_called_once_with(
        endpoint=stub_params.endpoint,
        model=stub_params.model,
        num_prompts=stub_params.num_prompts,
        isl=stub_params.isl,
        osl=stub_params.osl,
        request_rate=stub_params.request_rate,
    )
    assert isinstance(result, BenchResult)
    assert result.mean_ttft_ms == 50.0
    assert result.mean_itl_ms == 15.0
    assert result.output_throughput == 500.0


@pytest.mark.asyncio
async def test_execute_propagates_context_length_error(mock_bench_backend, stub_params):
    mock_bench_backend.run_benchmark.side_effect = ContextLengthExceededError()
    executor = HTTPBenchExecutor(mock_bench_backend)

    with pytest.raises(ContextLengthExceededError):
        await executor.execute(stub_params)
