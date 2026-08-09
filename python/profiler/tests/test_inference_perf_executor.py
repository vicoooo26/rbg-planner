# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for InferencePerfExecutor."""

import json
import os

import pytest

from rbg_profiler.bench.cli_executor import CLIBenchExecutor
from rbg_profiler.bench.executor import BenchExecutor, BenchParams, BenchResult
from rbg_profiler.bench.inference_perf_executor import InferencePerfExecutor
from rbg_profiler.rbg_discoverer import RoleEndpoint

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def stub_params():
    return BenchParams(
        endpoint=RoleEndpoint(base_url="http://test:8000", num_gpus=2, tokenizer="t"),
        model="test-model",
        isl=1024,
        osl=5,
        num_prompts=1,
    )


def test_inheritance():
    assert issubclass(InferencePerfExecutor, CLIBenchExecutor)
    assert issubclass(InferencePerfExecutor, BenchExecutor)


def test_tool_name():
    executor = InferencePerfExecutor(backend="vllm")
    assert executor.tool_name == "inference-perf"


@pytest.mark.parametrize("backend,server_type,api_type", [
    ("sglang-oai", "sglang", "completion"),
    ("sglang-oai-chat", "sglang", "chat"),
    ("vllm", "vllm", "completion"),
    ("vllm-chat", "vllm", "chat"),
])
def test_backend_mapping(backend, server_type, api_type):
    executor = InferencePerfExecutor(backend=backend)
    assert executor.server_type == server_type
    assert executor.api_type == api_type


def test_build_args(stub_params):
    executor = InferencePerfExecutor(backend="vllm")
    args = executor.build_args(stub_params, "/tmp/reports/test")

    assert args[0] == "inference-perf"
    assert "--server.type" in args
    assert "vllm" in args
    assert "--server.base_url" in args
    assert "http://test:8000" in args
    assert "--server.model_name" in args
    assert "test-model" in args
    assert "--server.ignore_eos" in args
    assert "true" in args
    assert "--api.type" in args
    assert "completion" in args
    assert "--api.streaming" in args
    assert "--data.type" in args
    assert "synthetic" in args
    assert "--data.input_distribution.mean" in args
    assert "1024" in args
    assert "--data.input_distribution.std_dev" in args
    assert "0" in args
    assert "--data.output_distribution.mean" in args
    assert "5" in args
    assert "--load.type" in args
    assert "concurrent" in args
    assert "--storage.local_storage.path" in args
    assert "/tmp/reports/test" in args

    # Verify load.stages JSON contains correct concurrency and num_requests
    stages_idx = args.index("--load.stages")
    stages_json = args[stages_idx + 1]
    stages = json.loads(stages_json)
    assert stages[0]["concurrency_level"] == 1
    assert stages[0]["num_requests"] == 1


def test_build_args_chat_backend(stub_params):
    executor = InferencePerfExecutor(backend="sglang-oai-chat")
    args = executor.build_args(stub_params, "/tmp/reports/test")
    api_type_idx = args.index("--api.type")
    assert args[api_type_idx + 1] == "chat"
    server_type_idx = args.index("--server.type")
    assert args[server_type_idx + 1] == "sglang"


def test_parse_report_bare_floats():
    """Parse report where throughput fields are bare floats."""
    executor = InferencePerfExecutor(backend="vllm")
    report_path = os.path.join(FIXTURES_DIR, "inference_perf_report.json")
    result = executor.parse_report(report_path)

    assert isinstance(result, BenchResult)
    # Latency values in seconds → ms (multiply by 1000)
    assert result.mean_ttft_ms == pytest.approx(800.0)   # 0.80 * 1000
    assert result.mean_itl_ms == pytest.approx(20.0)     # 0.02 * 1000
    assert result.mean_tpot_ms == pytest.approx(25.0)    # 0.025 * 1000
    assert result.mean_e2e_latency_ms == pytest.approx(3310.0)  # 3.31 * 1000
    # Throughput values stay as-is (already per second)
    assert result.request_throughput == pytest.approx(1.02)
    assert result.output_throughput == pytest.approx(676.12)


def test_parse_report_stat_dicts():
    """Parse report where throughput fields are stat dicts with .mean."""
    executor = InferencePerfExecutor(backend="vllm")
    report_path = os.path.join(FIXTURES_DIR, "inference_perf_report_stat_dict.json")
    result = executor.parse_report(report_path)

    assert result.mean_ttft_ms == pytest.approx(500.0)   # 0.5 * 1000
    assert result.mean_itl_ms == pytest.approx(15.0)     # 0.015 * 1000
    assert result.mean_tpot_ms == pytest.approx(20.0)    # 0.02 * 1000
    assert result.mean_e2e_latency_ms == pytest.approx(2500.0)  # 2.5 * 1000
    assert result.request_throughput == pytest.approx(2.0)
    assert result.output_throughput == pytest.approx(800.0)


def test_find_report_path(tmp_path):
    """find_report_path locates summary_lifecycle_metrics.json in nested dir."""
    executor = InferencePerfExecutor(backend="vllm")

    # inference-perf writes to reports-YYYYMMDD-HHMMSS/ subdirectory
    nested_dir = tmp_path / "reports-20260809-120000"
    nested_dir.mkdir()
    report_file = nested_dir / "summary_lifecycle_metrics.json"
    report_file.write_text('{"successes": {}}')

    found = executor.find_report_path(str(tmp_path))
    assert found == str(report_file)


def test_find_report_path_not_found(tmp_path):
    """find_report_path raises FileNotFoundError when report is missing."""
    executor = InferencePerfExecutor(backend="vllm")
    with pytest.raises(FileNotFoundError):
        executor.find_report_path(str(tmp_path))
