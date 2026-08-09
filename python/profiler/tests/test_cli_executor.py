# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CLIBenchExecutor template method."""

import json
import os

import pytest

from rbg_profiler.bench.cli_executor import CLIBenchExecutor
from rbg_profiler.bench.executor import (
    BenchExecutionError,
    BenchExecutor,
    BenchParams,
    BenchResult,
)
from rbg_profiler.rbg_discoverer import RoleEndpoint


class _StubCLIExecutor(CLIBenchExecutor):
    """Minimal concrete implementation for testing the template."""

    def __init__(self):
        self.build_args_calls: list[BenchParams] = []

    @property
    def tool_name(self) -> str:
        return "stub-tool"

    def build_args(self, params: BenchParams, report_dir: str) -> list[str]:
        self.build_args_calls.append(params)
        return ["echo", "bench-done"]

    def find_report_path(self, report_dir: str) -> str:
        return os.path.join(report_dir, "summary.json")

    def parse_report(self, report_path: str) -> BenchResult:
        with open(report_path) as f:
            data = json.load(f)
        return BenchResult(
            mean_ttft_ms=data["ttft"],
            mean_itl_ms=data["itl"],
            mean_tpot_ms=data["tpot"],
            request_throughput=data["rps"],
            output_throughput=data["tps"],
            mean_e2e_latency_ms=data["e2e"],
        )


@pytest.fixture
def stub_params():
    return BenchParams(
        endpoint=RoleEndpoint(base_url="http://test:8000", num_gpus=1, tokenizer="t"),
        model="test-model",
        isl=1024,
        osl=5,
        num_prompts=1,
    )


def test_cli_executor_is_bench_executor():
    assert issubclass(CLIBenchExecutor, BenchExecutor)


def test_cli_executor_is_abstract():
    with pytest.raises(TypeError):
        CLIBenchExecutor()


@pytest.mark.asyncio
async def test_execute_success(stub_params, tmp_path):
    """execute() runs subprocess, finds report, parses it."""
    executor = _StubCLIExecutor()

    # Patch _make_report_dir to use tmp_path
    executor._make_report_dir = lambda params: str(tmp_path / "report")

    # Create the report file that parse_report expects
    report_dir = str(tmp_path / "report")
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "summary.json"), "w") as f:
        json.dump({
            "ttft": 50.0, "itl": 15.0, "tpot": 15.0,
            "rps": 10.0, "tps": 500.0, "e2e": 100.0,
        }, f)

    result = await executor.execute(stub_params)

    assert isinstance(result, BenchResult)
    assert result.mean_ttft_ms == 50.0
    assert result.output_throughput == 500.0
    assert len(executor.build_args_calls) == 1


@pytest.mark.asyncio
async def test_execute_nonzero_exit_raises(stub_params, tmp_path):
    """Non-zero exit code raises BenchExecutionError."""

    class _FailingExecutor(_StubCLIExecutor):
        def build_args(self, params: BenchParams, report_dir: str) -> list[str]:
            return ["python3", "-c", "import sys; sys.exit(1)"]

    executor = _FailingExecutor()
    executor._make_report_dir = lambda params: str(tmp_path / "report")

    with pytest.raises(BenchExecutionError, match="stub-tool exited 1"):
        await executor.execute(stub_params)


@pytest.mark.asyncio
async def test_execute_report_not_found_raises(stub_params, tmp_path):
    """Missing report file raises FileNotFoundError."""

    class _NoReportExecutor(_StubCLIExecutor):
        def find_report_path(self, report_dir: str) -> str:
            return os.path.join(report_dir, "nonexistent.json")

    executor = _NoReportExecutor()
    executor._make_report_dir = lambda params: str(tmp_path / "report")
    os.makedirs(str(tmp_path / "report"), exist_ok=True)

    with pytest.raises(FileNotFoundError):
        await executor.execute(stub_params)
