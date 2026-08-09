# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for core bench executor abstractions."""

import pytest

from rbg_profiler.bench.executor import (
    BenchExecutionError,
    BenchExecutor,
    BenchParams,
    BenchResult,
)
from rbg_profiler.rbg_discoverer import RoleEndpoint


def test_bench_params_dataclass():
    endpoint = RoleEndpoint(base_url="http://test:8000", num_gpus=2, tokenizer="t")
    params = BenchParams(
        endpoint=endpoint, model="test-model",
        isl=1024, osl=5, num_prompts=1,
    )
    assert params.isl == 1024
    assert params.osl == 5
    assert params.num_prompts == 1
    assert params.request_rate == float("inf")


def test_bench_result_dataclass():
    result = BenchResult(
        mean_ttft_ms=100.0,
        mean_itl_ms=20.0,
        mean_tpot_ms=20.0,
        request_throughput=10.0,
        output_throughput=500.0,
        mean_e2e_latency_ms=1000.0,
    )
    assert result.mean_ttft_ms == 100.0
    assert result.output_throughput == 500.0


def test_bench_executor_is_abstract():
    with pytest.raises(TypeError):
        BenchExecutor()


def test_bench_execution_error_is_exception():
    err = BenchExecutionError("something failed")
    assert isinstance(err, Exception)
    assert "something failed" in str(err)
