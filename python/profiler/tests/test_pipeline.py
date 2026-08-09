# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ProfilingPipeline sweep logic and adaptation."""

import pytest

from rbg_profiler.bench.base import ContextLengthExceededError
from rbg_profiler.bench.executor import BenchExecutor, BenchParams, BenchResult
from rbg_profiler.bench.pipeline import ProfilingPipeline
from rbg_profiler.rbg_discoverer import RoleEndpoint


class _StubExecutor(BenchExecutor):
    """Stub executor that returns configurable results or raises errors."""

    def __init__(
        self, result: BenchResult | None = None, error: Exception | None = None
    ):
        self.result = result
        self.error = error
        self.calls: list[BenchParams] = []

    async def execute(self, params: BenchParams) -> BenchResult:
        self.calls.append(params)
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def stub_endpoint():
    return RoleEndpoint(base_url="http://test:8000", num_gpus=2, tokenizer="t")


@pytest.fixture
def good_result():
    return BenchResult(
        mean_ttft_ms=50.0,
        mean_itl_ms=15.0,
        mean_tpot_ms=15.0,
        request_throughput=10.0,
        output_throughput=500.0,
        mean_e2e_latency_ms=100.0,
    )


# ── bench_prefill ──


@pytest.mark.asyncio
async def test_prefill_sweep_basic(stub_endpoint, good_result):
    executor = _StubExecutor(result=good_result)
    pipeline = ProfilingPipeline(executor)

    results = await pipeline.bench_prefill(
        endpoint=stub_endpoint,
        model="m",
        isl_points=[128, 256, 512],
    )

    assert len(results) == 3
    assert [r.isl for r in results] == [128, 256, 512]
    # thpt = isl / ttft_ms * 1000 / num_gpus
    # 128 / 50.0 * 1000 / 2 = 1280.0
    assert results[0].thpt_per_gpu == 1280.0
    assert results[0].ttft_ms == 50.0
    assert len(executor.calls) == 3


@pytest.mark.asyncio
async def test_prefill_skips_isl_exceeding_context(stub_endpoint, good_result):
    executor = _StubExecutor(result=good_result)
    pipeline = ProfilingPipeline(executor)

    results = await pipeline.bench_prefill(
        endpoint=stub_endpoint,
        model="m",
        isl_points=[128, 32256],  # 32256 >= 32768 - 512 (= 32256), skipped
        max_context_length=32768,
    )

    assert len(results) == 1
    assert results[0].isl == 128
    assert len(executor.calls) == 1  # 32256 never attempted


@pytest.mark.asyncio
async def test_prefill_stops_on_context_length_error(stub_endpoint, good_result):
    error = ContextLengthExceededError()
    executor = _StubExecutor(error=error)
    pipeline = ProfilingPipeline(executor)

    results = await pipeline.bench_prefill(
        endpoint=stub_endpoint,
        model="m",
        isl_points=[128, 256, 512],
    )

    assert len(results) == 0
    assert len(executor.calls) == 1  # stopped after first error


@pytest.mark.asyncio
async def test_prefill_continues_on_generic_error(stub_endpoint, good_result):
    executor = _StubExecutor(error=RuntimeError("transient"))
    pipeline = ProfilingPipeline(executor)

    results = await pipeline.bench_prefill(
        endpoint=stub_endpoint,
        model="m",
        isl_points=[128, 256, 512],
    )

    assert len(results) == 0
    assert len(executor.calls) == 3  # all attempted, all failed


@pytest.mark.asyncio
async def test_prefill_skips_invalid_metrics(stub_endpoint):
    bad_result = BenchResult(
        mean_ttft_ms=0.0,  # invalid
        mean_itl_ms=15.0,
        mean_tpot_ms=15.0,
        request_throughput=10.0,
        output_throughput=500.0,
        mean_e2e_latency_ms=100.0,
    )
    executor = _StubExecutor(result=bad_result)
    pipeline = ProfilingPipeline(executor)

    results = await pipeline.bench_prefill(
        endpoint=stub_endpoint,
        model="m",
        isl_points=[128],
    )

    assert len(results) == 0


# ── bench_decode ──


@pytest.mark.asyncio
async def test_decode_sweep_basic(stub_endpoint, good_result):
    executor = _StubExecutor(result=good_result)
    pipeline = ProfilingPipeline(executor)

    results = await pipeline.bench_decode(
        endpoint=stub_endpoint,
        model="m",
        isl_points=[256],
        max_kv_tokens=32768,
        osl=500,
        concurrency_granularity=3,
    )

    # max_concurrency = min(32768 // (256+500), 512) = min(43, 512) = 43
    # 3 points from 1 to 43
    assert len(results) >= 1
    assert all(r.isl == 256 for r in results)
    # thpt_per_gpu = output_throughput / num_gpus = 500 / 2 = 250
    assert all(r.thpt_per_gpu == 250.0 for r in results)
    # kv_usage = (isl + osl/2) * conc / max_kv_tokens
    expected_kv = (256 + 250) * results[0].concurrency / 32768
    assert results[0].kv_usage == pytest.approx(expected_kv, rel=1e-3)


@pytest.mark.asyncio
async def test_decode_stops_on_context_length_error(stub_endpoint, good_result):
    error = ContextLengthExceededError()
    executor = _StubExecutor(error=error)
    pipeline = ProfilingPipeline(executor)

    results = await pipeline.bench_decode(
        endpoint=stub_endpoint,
        model="m",
        isl_points=[256],
        max_kv_tokens=32768,
        osl=500,
    )

    assert len(results) == 0
    assert len(executor.calls) == 1  # stopped


@pytest.mark.asyncio
async def test_decode_skips_isl_exceeding_context(stub_endpoint, good_result):
    executor = _StubExecutor(result=good_result)
    pipeline = ProfilingPipeline(executor)

    results = await pipeline.bench_decode(
        endpoint=stub_endpoint,
        model="m",
        isl_points=[128, 32268],  # 32268 + 500 >= 32768
        max_kv_tokens=32768,
        osl=500,
    )

    assert all(r.isl == 128 for r in results)
    assert len(executor.calls) >= 1
    # 32268 never attempted — all calls have isl=128
    assert all(c.isl == 128 for c in executor.calls)


# ── bench_unified ──


@pytest.mark.asyncio
async def test_unified_sweep_basic(stub_endpoint, good_result):
    executor = _StubExecutor(result=good_result)
    pipeline = ProfilingPipeline(executor)

    results = await pipeline.bench_unified(
        endpoint=stub_endpoint,
        model="m",
        isl_points=[128, 256],
        concurrency_grid=[1, 2],
        osl=500,
    )

    assert len(results) == 4  # 2 ISL × 2 concurrency
    assert results[0].isl == 128
    assert results[0].concurrency == 1
    assert results[0].ttft_ms == 50.0
    assert results[0].itl_ms == 15.0
    # thpt = output_throughput / num_gpus = 500 / 2 = 250
    assert results[0].thpt_per_gpu == 250.0


@pytest.mark.asyncio
async def test_unified_stops_on_context_length_error(stub_endpoint, good_result):
    error = ContextLengthExceededError()
    executor = _StubExecutor(error=error)
    pipeline = ProfilingPipeline(executor)

    results = await pipeline.bench_unified(
        endpoint=stub_endpoint,
        model="m",
        isl_points=[128, 256],
        concurrency_grid=[1, 2],
    )

    assert len(results) == 0
    assert len(executor.calls) == 1  # stopped on first
