# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for bench executor factory."""

import pytest

from rbg_profiler.bench.executor import BenchExecutor
from rbg_profiler.bench.factory import create_bench_executor
from rbg_profiler.bench.http_executor import HTTPBenchExecutor
from rbg_profiler.bench.inference_perf_executor import InferencePerfExecutor


def test_create_http_executor():
    executor = create_bench_executor(method="http", backend="sglang-oai")
    assert isinstance(executor, HTTPBenchExecutor)
    assert isinstance(executor, BenchExecutor)


def test_create_http_executor_vllm_chat():
    executor = create_bench_executor(method="http", backend="vllm-chat")
    assert isinstance(executor, HTTPBenchExecutor)


def test_create_inference_perf_executor():
    executor = create_bench_executor(method="inference-perf", backend="vllm")
    assert isinstance(executor, InferencePerfExecutor)
    assert isinstance(executor, BenchExecutor)


def test_create_inference_perf_executor_sglang_chat():
    executor = create_bench_executor(method="inference-perf", backend="sglang-oai-chat")
    assert isinstance(executor, InferencePerfExecutor)
    assert executor.api_type == "chat"


def test_create_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown bench method"):
        create_bench_executor(method="unknown", backend="vllm")


def test_create_unknown_backend_for_inference_perf_raises():
    with pytest.raises(ValueError, match="Unsupported backend"):
        create_bench_executor(method="inference-perf", backend="unknown-engine")
