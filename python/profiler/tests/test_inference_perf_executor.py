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

    def _val(flag):
        idx = args.index(flag)
        return args[idx + 1]

    assert _val("--server.type") == "vllm"
    assert _val("--server.base_url") == "http://test:8000"
    assert _val("--server.model_name") == "test-model"
    assert _val("--server.ignore_eos") == "true"
    assert _val("--api.type") == "completion"
    assert _val("--api.streaming") == "true"
    assert _val("--data.type") == "synthetic"
    assert _val("--data.input_distribution.mean") == "1024"
    assert _val("--data.input_distribution.std_dev") == "0"
    assert _val("--data.input_distribution.min") == "1024"
    assert _val("--data.input_distribution.max") == "1024"
    assert _val("--data.input_distribution.total_count") == "1"
    assert _val("--data.output_distribution.mean") == "5"
    assert _val("--data.output_distribution.std_dev") == "0"
    assert _val("--data.output_distribution.min") == "5"
    assert _val("--data.output_distribution.max") == "5"
    assert _val("--data.output_distribution.total_count") == "1"
    assert _val("--load.type") == "concurrent"
    assert _val("--report.request_lifecycle.summary") == "true"
    assert _val("--report.request_lifecycle.percentiles") == "[50, 90, 99]"
    assert _val("--storage.local_storage.path") == "/tmp/reports/test"

    # Verify load.stages JSON contains correct concurrency and num_requests
    stages_json = _val("--load.stages")
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


def test_invalid_backend_raises():
    """InferencePerfExecutor raises ValueError for unsupported backend."""
    with pytest.raises(ValueError, match="Unsupported backend"):
        InferencePerfExecutor(backend="unknown-engine")


def test_parse_report_with_failures(tmp_path, caplog):
    """parse_report logs warning when failures.count > 0 but still returns result."""
    executor = InferencePerfExecutor(backend="vllm")
    report = {
        "successes": {
            "count": 477,
            "latency": {
                "request_latency": {"mean": 3.0},
                "time_to_first_token": {"mean": 0.5},
                "inter_token_latency": {"mean": 0.01},
                "time_per_output_token": {"mean": 0.02},
            },
            "throughput": {
                "requests_per_sec": 1.0,
                "output_tokens_per_sec": 600.0,
            },
        },
        "failures": {"count": 3},
        "benchmark_time_seconds": 60.0,
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))

    import logging
    with caplog.at_level(logging.WARNING):
        result = executor.parse_report(str(report_path))

    assert isinstance(result, BenchResult)
    assert result.mean_ttft_ms == pytest.approx(500.0)
    assert result.output_throughput == pytest.approx(600.0)
    assert "3 failed requests" in caplog.text
