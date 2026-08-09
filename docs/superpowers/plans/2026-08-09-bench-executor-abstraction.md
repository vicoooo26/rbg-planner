# Bench Executor Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate bench execution (HTTP bench_serving, inference-perf CLI) from shared sweep+adaptation logic so the profiler can run benchmarks through different tools while producing the same output format for the planner.

**Architecture:** A `BenchExecutor` ABC defines the pluggable single-point execution interface. `HTTPBenchExecutor` wraps the existing `BaseBenchBackend`. `InferencePerfExecutor` (via `CLIBenchExecutor` template) runs the inference-perf CLI as a subprocess and parses its JSON report. `ProfilingPipeline` owns the sweep logic and adaptation (BenchResult → PrefillProfileData/DecodeProfileData), calling `executor.execute()` per sweep point. A factory selects the executor based on `--bench-method`.

**Tech Stack:** Python 3.10+, asyncio, subprocess, pytest + pytest-asyncio

## Global Constraints

- Python >=3.10 (from `python/profiler/pyproject.toml`)
- Test command: `cd python/profiler && pytest tests/ -v`
- Existing tests must not break — default `--bench-method http` preserves current behavior
- No planner-side changes (output format `prefill_raw_data.json` / `decode_raw_data.json` unchanged)
- No changes to `bench/base.py`, `bench/sglang.py`, `bench/vllm.py` — existing hierarchy stays as internal detail of HTTPBenchExecutor
- SPDX header on all new source files: `# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.` + `# SPDX-License-Identifier: Apache-2.0`

---

### Task 1: Core Executor Abstractions (BenchParams, BenchResult, BenchExecutor, BenchExecutionError)

**Files:**
- Create: `python/profiler/rbg_profiler/bench/executor.py`
- Test: `python/profiler/tests/test_executor.py`

**Interfaces:**
- Produces: `BenchParams` (dataclass), `BenchResult` (dataclass), `BenchExecutor` (ABC with `execute()`), `BenchExecutionError` (Exception). These are consumed by all later tasks.

- [ ] **Step 1: Write the failing tests**

Create `python/profiler/tests/test_executor.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python/profiler && pytest tests/test_executor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rbg_profiler.bench.executor'`

- [ ] **Step 3: Write minimal implementation**

Create `python/profiler/rbg_profiler/bench/executor.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core bench executor abstractions.

Defines the pluggable execution interface for benchmarking inference
endpoints. Each executor runs a single benchmark point (one ISL/OSL/
concurrency combination) and returns standardized BenchResult metrics.
The ProfilingPipeline owns the sweep logic and calls execute() per point.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from rbg_profiler.rbg_discoverer import RoleEndpoint


class BenchExecutionError(Exception):
    """Raised when a bench executor fails to produce results."""


@dataclass
class BenchParams:
    """Common input for a single benchmark point."""

    endpoint: RoleEndpoint
    model: str
    isl: int
    osl: int
    num_prompts: int
    request_rate: float = float("inf")


@dataclass
class BenchResult:
    """Common output from any bench execution."""

    mean_ttft_ms: float
    mean_itl_ms: float
    mean_tpot_ms: float
    request_throughput: float
    output_throughput: float
    mean_e2e_latency_ms: float


class BenchExecutor(ABC):
    """Pluggable bench execution layer.

    Each executor runs a single benchmark point (one ISL/OSL/concurrency
    combination) and returns standardized metrics. The ProfilingPipeline
    owns the sweep logic and calls execute() per point.
    """

    @abstractmethod
    async def execute(self, params: BenchParams) -> BenchResult:
        """Run a single benchmark and return standardized metrics."""
        raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python/profiler && pytest tests/test_executor.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add python/profiler/rbg_profiler/bench/executor.py python/profiler/tests/test_executor.py
git commit -m "feat(bench): add core executor abstractions (BenchParams, BenchResult, BenchExecutor)"
```

---

### Task 2: CLIBenchExecutor ABC (Template Method)

**Files:**
- Create: `python/profiler/rbg_profiler/bench/cli_executor.py`
- Test: `python/profiler/tests/test_cli_executor.py`

**Interfaces:**
- Consumes: `BenchParams`, `BenchResult`, `BenchExecutor`, `BenchExecutionError` from Task 1
- Produces: `CLIBenchExecutor` (ABC with template method `execute()`). Subclasses implement `build_args()`, `parse_report()`, `find_report_path()`, `tool_name`. Consumed by Task 4 (InferencePerfExecutor).

- [ ] **Step 1: Write the failing tests**

Create `python/profiler/tests/test_cli_executor.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python/profiler && pytest tests/test_cli_executor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rbg_profiler.bench.cli_executor'`

- [ ] **Step 3: Write minimal implementation**

Create `python/profiler/rbg_profiler/bench/cli_executor.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI-based bench executor template.

CLIBenchExecutor is a template method base for bench tools invoked as
subprocesses (inference-perf, aiperf). Subclasses implement build_args()
and parse_report(); the execute() method handles subprocess lifecycle,
error handling, and report discovery.
"""

import asyncio
import logging
import os
import tempfile

from abc import abstractmethod

from rbg_profiler.bench.executor import (
    BenchExecutionError,
    BenchExecutor,
    BenchParams,
    BenchResult,
)

logger = logging.getLogger(__name__)


class CLIBenchExecutor(BenchExecutor):
    """Template for CLI-based bench tools.

    Subclasses implement build_args() and parse_report().
    The execute() method handles subprocess lifecycle, error handling,
    and report discovery.
    """

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Name of the CLI tool, used in error messages."""
        raise NotImplementedError

    @abstractmethod
    def build_args(self, params: BenchParams, report_dir: str) -> list[str]:
        """Build CLI args list from params. Pure function."""
        raise NotImplementedError

    @abstractmethod
    def find_report_path(self, report_dir: str) -> str:
        """Locate the report JSON file within the output directory."""
        raise NotImplementedError

    @abstractmethod
    def parse_report(self, report_path: str) -> BenchResult:
        """Parse the tool's report file into BenchResult."""
        raise NotImplementedError

    def _make_report_dir(self, params: BenchParams) -> str:
        """Create a unique report directory for this benchmark point."""
        dir_name = f"{self.tool_name}_isl{params.isl}_osl{params.osl}_n{params.num_prompts}"
        report_dir = os.path.join(tempfile.gettempdir(), "rbg-bench-reports", dir_name)
        os.makedirs(report_dir, exist_ok=True)
        return report_dir

    async def execute(self, params: BenchParams) -> BenchResult:
        """Run the CLI tool and parse its report."""
        report_dir = self._make_report_dir(params)
        args = self.build_args(params, report_dir)

        logger.info("Running %s: %s", self.tool_name, " ".join(args))

        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise BenchExecutionError(
                f"{self.tool_name} exited {process.returncode}: "
                f"{stderr.decode()[:500]}"
            )

        report_path = self.find_report_path(report_dir)
        return self.parse_report(report_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python/profiler && pytest tests/test_cli_executor.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add python/profiler/rbg_profiler/bench/cli_executor.py python/profiler/tests/test_cli_executor.py
git commit -m "feat(bench): add CLIBenchExecutor template method for CLI-based bench tools"
```

---

### Task 3: HTTPBenchExecutor

**Files:**
- Create: `python/profiler/rbg_profiler/bench/http_executor.py`
- Test: `python/profiler/tests/test_http_executor.py`

**Interfaces:**
- Consumes: `BenchParams`, `BenchResult`, `BenchExecutor` from Task 1; `BaseBenchBackend` from existing `bench/base.py`; `create_bench_backend` from existing `bench/__init__.py`
- Produces: `HTTPBenchExecutor`. Consumed by Task 6 (factory).

- [ ] **Step 1: Write the failing tests**

Create `python/profiler/tests/test_http_executor.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python/profiler && pytest tests/test_http_executor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rbg_profiler.bench.http_executor'`

- [ ] **Step 3: Write minimal implementation**

Create `python/profiler/rbg_profiler/bench/http_executor.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python/profiler && pytest tests/test_http_executor.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add python/profiler/rbg_profiler/bench/http_executor.py python/profiler/tests/test_http_executor.py
git commit -m "feat(bench): add HTTPBenchExecutor wrapping BaseBenchBackend"
```

---

### Task 4: InferencePerfExecutor

**Files:**
- Create: `python/profiler/rbg_profiler/bench/inference_perf_executor.py`
- Create: `python/profiler/tests/fixtures/inference_perf_report.json`
- Create: `python/profiler/tests/fixtures/inference_perf_report_stat_dict.json`
- Test: `python/profiler/tests/test_inference_perf_executor.py`

**Interfaces:**
- Consumes: `CLIBenchExecutor` from Task 2; `BenchParams`, `BenchResult`, `BenchExecutionError` from Task 1; `RoleEndpoint` from existing `rbg_discoverer.py`
- Produces: `InferencePerfExecutor`. Consumed by Task 6 (factory).

- [ ] **Step 1: Create test fixtures**

Create `python/profiler/tests/fixtures/inference_perf_report.json`:

```json
{
  "successes": {
    "count": 480,
    "latency": {
      "request_latency": { "mean": 3.31, "median": 2.11, "p90": 5.94 },
      "time_to_first_token": { "mean": 0.80, "median": 0.20, "p90": 2.26 },
      "inter_token_latency": { "mean": 0.02, "median": 0.015, "p90": 0.04 },
      "normalized_time_per_output_token": { "mean": 0.025, "median": 0.02, "p90": 0.05 },
      "time_per_output_token": { "mean": 0.025, "median": 0.02, "p90": 0.05 }
    },
    "throughput": {
      "requests_per_sec": 1.02,
      "output_tokens_per_sec": 676.12,
      "total_tokens_per_sec": 676.12
    }
  },
  "failures": { "count": 3 },
  "benchmark_time_seconds": 60.0
}
```

Create `python/profiler/tests/fixtures/inference_perf_report_stat_dict.json` (throughput as stat dicts):

```json
{
  "successes": {
    "count": 100,
    "latency": {
      "request_latency": { "mean": 2.5 },
      "time_to_first_token": { "mean": 0.5 },
      "inter_token_latency": { "mean": 0.015 },
      "time_per_output_token": { "mean": 0.02 }
    },
    "throughput": {
      "requests_per_sec": { "mean": 2.0 },
      "output_tokens_per_sec": { "mean": 800.0 },
      "total_tokens_per_sec": { "mean": 800.0 }
    }
  },
  "failures": { "count": 0 },
  "benchmark_time_seconds": 50.0
}
```

- [ ] **Step 2: Write the failing tests**

Create `python/profiler/tests/test_inference_perf_executor.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd python/profiler && pytest tests/test_inference_perf_executor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rbg_profiler.bench.inference_perf_executor'`

- [ ] **Step 4: Write minimal implementation**

Create `python/profiler/rbg_profiler/bench/inference_perf_executor.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""inference-perf CLI bench executor.

Runs the inference-perf CLI tool as a subprocess and parses its
summary_lifecycle_metrics.json report into BenchResult.

inference-perf is a production-scale GenAI inference performance
benchmarking tool: https://github.com/kubernetes-sigs/inference-perf
"""

import json
import logging
import os

from rbg_profiler.bench.cli_executor import CLIBenchExecutor
from rbg_profiler.bench.executor import BenchParams, BenchResult

logger = logging.getLogger(__name__)

_BACKEND_MAP: dict[str, tuple[str, str]] = {
    "sglang-oai": ("sglang", "completion"),
    "sglang-oai-chat": ("sglang", "chat"),
    "vllm": ("vllm", "completion"),
    "vllm-chat": ("vllm", "chat"),
}


class InferencePerfExecutor(CLIBenchExecutor):
    """Bench executor using the inference-perf CLI tool."""

    def __init__(self, backend: str):
        if backend not in _BACKEND_MAP:
            raise ValueError(
                f"Unsupported backend for inference-perf: {backend}. "
                f"Supported: {list(_BACKEND_MAP.keys())}"
            )
        self.server_type, self.api_type = _BACKEND_MAP[backend]

    @property
    def tool_name(self) -> str:
        return "inference-perf"

    def build_args(self, params: BenchParams, report_dir: str) -> list[str]:
        return [
            "inference-perf",
            "--server.type", self.server_type,
            "--server.base_url", params.endpoint.base_url,
            "--server.model_name", params.model,
            "--server.ignore_eos", "true",
            "--api.type", self.api_type,
            "--api.streaming", "true",
            "--data.type", "synthetic",
            "--data.input_distribution.mean", str(params.isl),
            "--data.input_distribution.std_dev", "0",
            "--data.input_distribution.min", str(params.isl),
            "--data.input_distribution.max", str(params.isl),
            "--data.input_distribution.total_count", str(params.num_prompts),
            "--data.output_distribution.mean", str(params.osl),
            "--data.output_distribution.std_dev", "0",
            "--data.output_distribution.min", str(params.osl),
            "--data.output_distribution.max", str(params.osl),
            "--data.output_distribution.total_count", str(params.num_prompts),
            "--load.type", "concurrent",
            "--load.stages", json.dumps([{
                "concurrency_level": params.num_prompts,
                "num_requests": params.num_prompts,
            }]),
            "--report.request_lifecycle.summary", "true",
            "--report.request_lifecycle.percentiles", "[50, 90, 99]",
            "--storage.local_storage.path", report_dir,
        ]

    def find_report_path(self, report_dir: str) -> str:
        """Locate summary_lifecycle_metrics.json in the report directory."""
        for root, _, files in os.walk(report_dir):
            if "summary_lifecycle_metrics.json" in files:
                return os.path.join(root, "summary_lifecycle_metrics.json")
        raise FileNotFoundError(
            f"summary_lifecycle_metrics.json not found in {report_dir}"
        )

    def parse_report(self, report_path: str) -> BenchResult:
        """Parse inference-perf summary_lifecycle_metrics.json into BenchResult."""
        with open(report_path) as f:
            report = json.load(f)

        successes = report["successes"]
        latency = successes["latency"]
        throughput = successes["throughput"]

        failures = report.get("failures", {})
        if failures.get("count", 0) > 0:
            logger.warning(
                "inference-perf report has %d failed requests",
                failures["count"],
            )

        def _extract(val):
            """Handle both bare floats and stat dicts with .mean."""
            return val["mean"] if isinstance(val, dict) else val

        return BenchResult(
            mean_ttft_ms=_extract(latency["time_to_first_token"]) * 1000,
            mean_itl_ms=_extract(latency["inter_token_latency"]) * 1000,
            mean_tpot_ms=_extract(latency["time_per_output_token"]) * 1000,
            request_throughput=_extract(throughput["requests_per_sec"]),
            output_throughput=_extract(throughput["output_tokens_per_sec"]),
            mean_e2e_latency_ms=_extract(latency["request_latency"]) * 1000,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd python/profiler && pytest tests/test_inference_perf_executor.py -v`
Expected: 9 PASSED

- [ ] **Step 6: Commit**

```bash
git add python/profiler/rbg_profiler/bench/inference_perf_executor.py \
        python/profiler/tests/test_inference_perf_executor.py \
        python/profiler/tests/fixtures/inference_perf_report.json \
        python/profiler/tests/fixtures/inference_perf_report_stat_dict.json
git commit -m "feat(bench): add InferencePerfExecutor with CLI args construction and report parsing"
```

---

### Task 5: ProfilingPipeline (Sweep + Adaptation)

**Files:**
- Create: `python/profiler/rbg_profiler/bench/pipeline.py`
- Test: `python/profiler/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `BenchExecutor`, `BenchParams`, `BenchResult` from Task 1; `PrefillProfileData`, `DecodeProfileData`, `UnifiedProfileData`, `ContextLengthExceededError` from existing `bench/base.py`; `RoleEndpoint` from existing `rbg_discoverer.py`
- Produces: `ProfilingPipeline` with `bench_prefill()`, `bench_decode()`, `bench_unified()`. Consumed by Task 7 (profiler wiring).

- [ ] **Step 1: Write the failing tests**

Create `python/profiler/tests/test_pipeline.py`:

```python
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

    def __init__(self, result: BenchResult = None, error: Exception = None):
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
        endpoint=stub_endpoint, model="m",
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
        endpoint=stub_endpoint, model="m",
        isl_points=[128, 32216],  # 32216 >= 32768 - 512
        max_context_length=32768,
    )

    assert len(results) == 1
    assert results[0].isl == 128
    assert len(executor.calls) == 1  # 32216 never attempted


@pytest.mark.asyncio
async def test_prefill_stops_on_context_length_error(stub_endpoint, good_result):
    error = ContextLengthExceededError()
    executor = _StubExecutor(error=error)
    pipeline = ProfilingPipeline(executor)

    results = await pipeline.bench_prefill(
        endpoint=stub_endpoint, model="m",
        isl_points=[128, 256, 512],
    )

    assert len(results) == 0
    assert len(executor.calls) == 1  # stopped after first error


@pytest.mark.asyncio
async def test_prefill_continues_on_generic_error(stub_endpoint, good_result):
    executor = _StubExecutor(error=RuntimeError("transient"))
    pipeline = ProfilingPipeline(executor)

    results = await pipeline.bench_prefill(
        endpoint=stub_endpoint, model="m",
        isl_points=[128, 256, 512],
    )

    assert len(results) == 0
    assert len(executor.calls) == 3  # all attempted, all failed


@pytest.mark.asyncio
async def test_prefill_skips_invalid_metrics(stub_endpoint):
    bad_result = BenchResult(
        mean_ttft_ms=0.0,  # invalid
        mean_itl_ms=15.0, mean_tpot_ms=15.0,
        request_throughput=10.0, output_throughput=500.0,
        mean_e2e_latency_ms=100.0,
    )
    executor = _StubExecutor(result=bad_result)
    pipeline = ProfilingPipeline(executor)

    results = await pipeline.bench_prefill(
        endpoint=stub_endpoint, model="m",
        isl_points=[128],
    )

    assert len(results) == 0


# ── bench_decode ──

@pytest.mark.asyncio
async def test_decode_sweep_basic(stub_endpoint, good_result):
    executor = _StubExecutor(result=good_result)
    pipeline = ProfilingPipeline(executor)

    results = await pipeline.bench_decode(
        endpoint=stub_endpoint, model="m",
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
        endpoint=stub_endpoint, model="m",
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
        endpoint=stub_endpoint, model="m",
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
        endpoint=stub_endpoint, model="m",
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
        endpoint=stub_endpoint, model="m",
        isl_points=[128, 256],
        concurrency_grid=[1, 2],
    )

    assert len(results) == 0
    assert len(executor.calls) == 1  # stopped on first
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python/profiler && pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rbg_profiler.bench.pipeline'`

- [ ] **Step 3: Write minimal implementation**

Create `python/profiler/rbg_profiler/bench/pipeline.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profiling pipeline: sweep orchestration + result adaptation.

Owns the sweep logic (iterate ISL/concurrency points) and adaptation
(BenchResult → PrefillProfileData/DecodeProfileData/UnifiedProfileData).
Calls executor.execute() per sweep point — the executor is the only
pluggable part.
"""

import logging

from rbg_profiler.bench.base import (
    ContextLengthExceededError,
    DecodeProfileData,
    PrefillProfileData,
    UnifiedProfileData,
)
from rbg_profiler.bench.executor import BenchExecutor, BenchParams, BenchResult
from rbg_profiler.rbg_discoverer import RoleEndpoint

logger = logging.getLogger(__name__)


class ProfilingPipeline:
    """Orchestrates bench sweeps using a pluggable executor."""

    def __init__(self, executor: BenchExecutor):
        self.executor = executor

    async def bench_prefill(
        self,
        endpoint: RoleEndpoint,
        model: str,
        isl_points: list[int],
        num_prompts: int = 5,
        osl: int = 5,
        max_context_length: int = 32768,
    ) -> list[PrefillProfileData]:
        """Profile prefill across ISL points."""
        results: list[PrefillProfileData] = []
        for isl in isl_points:
            if isl >= max_context_length - 512:
                logger.info("Skipping ISL=%d (exceeds max_context_length)", isl)
                continue
            try:
                result = await self.executor.execute(BenchParams(
                    endpoint=endpoint, model=model,
                    isl=isl, osl=osl,
                    num_prompts=num_prompts,
                ))
            except ContextLengthExceededError:
                logger.info("ISL=%d rejected with HTTP 400, stopping prefill sweep", isl)
                break
            except Exception:
                logger.exception("ISL=%d: benchmark failed, skipping", isl)
                continue
            if not (result.mean_ttft_ms > 0) or endpoint.num_gpus <= 0:
                logger.warning(
                    "ISL=%d: invalid metrics (ttft=%s, gpus=%d), skipping",
                    isl, result.mean_ttft_ms, endpoint.num_gpus,
                )
                continue
            thpt = isl / result.mean_ttft_ms * 1000 / endpoint.num_gpus
            results.append(PrefillProfileData(
                isl=isl, ttft_ms=result.mean_ttft_ms, thpt_per_gpu=round(thpt, 2)
            ))
            logger.info(
                "ISL=%d: TTFT=%.2fms, thpt/gpu=%.1f tok/s",
                isl, result.mean_ttft_ms, thpt,
            )
        return results

    async def bench_decode(
        self,
        endpoint: RoleEndpoint,
        model: str,
        isl_points: list[int],
        max_kv_tokens: int,
        osl: int = 500,
        max_context_length: int = 32768,
        concurrency_granularity: int = 6,
    ) -> list[DecodeProfileData]:
        """Profile decode across ISL x concurrency grid."""
        results: list[DecodeProfileData] = []
        for isl in isl_points:
            if isl + osl >= max_context_length:
                logger.info("Skipping ISL=%d (isl+osl exceeds context length)", isl)
                continue
            context_length = isl + osl // 2
            max_concurrency = min(max_kv_tokens // (isl + osl), 512)
            if max_concurrency < 1:
                logger.warning("ISL=%d: max_concurrency=0, skipping", isl)
                continue
            if max_concurrency <= concurrency_granularity:
                concurrency_points = list(range(1, max_concurrency + 1))
            else:
                step = (max_concurrency - 1) / (concurrency_granularity - 1)
                concurrency_points = sorted(
                    set(max(1, int(1 + i * step)) for i in range(concurrency_granularity))
                )
            logger.info(
                "ISL=%d: context_len=%d, max_conc=%d, sweep=%s",
                isl, context_length, max_concurrency, concurrency_points,
            )
            stop_sweep = False
            for conc in concurrency_points:
                try:
                    result = await self.executor.execute(BenchParams(
                        endpoint=endpoint, model=model,
                        isl=isl, osl=osl,
                        num_prompts=conc,
                    ))
                except ContextLengthExceededError:
                    logger.info("ISL=%d rejected with HTTP 400, stopping decode sweep", isl)
                    stop_sweep = True
                    break
                except Exception:
                    logger.exception("ISL=%d conc=%d: benchmark failed, skipping", isl, conc)
                    continue
                if endpoint.num_gpus <= 0:
                    logger.warning("ISL=%d conc=%d: invalid num_gpus, skipping", isl, conc)
                    continue
                kv_usage = (isl + osl / 2) * conc / max_kv_tokens
                thpt_per_gpu = result.output_throughput / endpoint.num_gpus
                results.append(DecodeProfileData(
                    isl=isl,
                    concurrency=conc,
                    kv_usage=round(kv_usage, 6),
                    context_length=context_length,
                    itl_ms=round(result.mean_itl_ms, 3),
                    thpt_per_gpu=round(thpt_per_gpu, 2),
                ))
                logger.info(
                    "  conc=%d: kv_usage=%.4f, ITL=%.2fms, thpt/gpu=%.1f tok/s",
                    conc, kv_usage, result.mean_itl_ms, thpt_per_gpu,
                )
            if stop_sweep:
                break
        return results

    async def bench_unified(
        self,
        endpoint: RoleEndpoint,
        model: str,
        isl_points: list[int],
        concurrency_grid: list[int],
        osl: int = 500,
    ) -> list[UnifiedProfileData]:
        """Profile unified prefill+decode."""
        results: list[UnifiedProfileData] = []
        for isl in isl_points:
            stop_sweep = False
            for conc in concurrency_grid:
                try:
                    result = await self.executor.execute(BenchParams(
                        endpoint=endpoint, model=model,
                        isl=isl, osl=osl,
                        num_prompts=conc,
                    ))
                except ContextLengthExceededError:
                    logger.info("ISL=%d rejected with HTTP 400, stopping unified sweep", isl)
                    stop_sweep = True
                    break
                if endpoint.num_gpus <= 0:
                    logger.warning("ISL=%d conc=%d: invalid num_gpus, skipping", isl, conc)
                    continue
                thpt_per_gpu = result.output_throughput / endpoint.num_gpus
                results.append(UnifiedProfileData(
                    isl=isl,
                    concurrency=conc,
                    ttft_ms=round(result.mean_ttft_ms, 2),
                    itl_ms=round(result.mean_itl_ms, 3),
                    thpt_per_gpu=round(thpt_per_gpu, 2),
                ))
            if stop_sweep:
                break
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python/profiler && pytest tests/test_pipeline.py -v`
Expected: 11 PASSED

- [ ] **Step 5: Commit**

```bash
git add python/profiler/rbg_profiler/bench/pipeline.py python/profiler/tests/test_pipeline.py
git commit -m "feat(bench): add ProfilingPipeline with sweep logic and result adaptation"
```

---

### Task 6: Factory + Package Exports

**Files:**
- Create: `python/profiler/rbg_profiler/bench/factory.py`
- Modify: `python/profiler/rbg_profiler/bench/__init__.py`
- Test: `python/profiler/tests/test_factory.py`

**Interfaces:**
- Consumes: `HTTPBenchExecutor` from Task 3; `InferencePerfExecutor` from Task 4; `create_bench_backend` from existing `bench/__init__.py`
- Produces: `create_bench_executor(method, backend)` factory function. Consumed by Task 7 (profiler wiring).

- [ ] **Step 1: Write the failing tests**

Create `python/profiler/tests/test_factory.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python/profiler && pytest tests/test_factory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rbg_profiler.bench.factory'`

- [ ] **Step 3: Write the factory implementation**

Create `python/profiler/rbg_profiler/bench/factory.py`:

```python
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
```

- [ ] **Step 4: Update package exports**

Modify `python/profiler/rbg_profiler/bench/__init__.py` — add new exports at the end of the file:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable bench backend module.

Provides a unified BenchBackend interface with concrete backends for
each supported inference engine (SGLang, vLLM).

Usage::

    from rbg_profiler.bench import create_bench_backend

    backend = create_bench_backend("sglang-oai")
    metrics = await backend.run_benchmark(endpoint, model, 5, 1024, 5)

For pluggable execution methods (HTTP, inference-perf), use the factory::

    from rbg_profiler.bench import create_bench_executor

    executor = create_bench_executor("http", "sglang-oai")
    result = await executor.execute(params)
"""

from rbg_profiler.bench.base import BaseBenchBackend, BenchBackend
from rbg_profiler.bench.executor import BenchExecutor, BenchParams, BenchResult
from rbg_profiler.bench.factory import create_bench_executor
from rbg_profiler.bench.http_executor import HTTPBenchExecutor
from rbg_profiler.bench.inference_perf_executor import InferencePerfExecutor
from rbg_profiler.bench.pipeline import ProfilingPipeline
from rbg_profiler.bench.sglang import SGLangBenchBackend, SGLangChatBenchBackend
from rbg_profiler.bench.vllm import VLLMBenchBackend, VLLMChatBenchBackend

BENCH_BACKENDS: dict[str, type[BenchBackend]] = {
    "sglang-oai": SGLangBenchBackend,
    "sglang-oai-chat": SGLangChatBenchBackend,
    "vllm": VLLMBenchBackend,
    "vllm-chat": VLLMChatBenchBackend,
}


def create_bench_backend(name: str) -> BenchBackend:
    """Create a bench backend for the given name.

    Args:
        name: Backend identifier (e.g. "sglang-oai", "vllm-chat").

    Raises:
        ValueError: If the name is not a registered backend.
    """
    backend_cls = BENCH_BACKENDS.get(name)
    if backend_cls is None:
        raise ValueError(
            f"Unknown bench backend: {name}. "
            f"Available: {list(BENCH_BACKENDS.keys())}"
        )
    return backend_cls()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd python/profiler && pytest tests/test_factory.py -v`
Expected: 6 PASSED

- [ ] **Step 6: Run all tests to verify no regressions**

Run: `cd python/profiler && pytest tests/ -v`
Expected: All tests PASS (existing 47 + new tests)

- [ ] **Step 7: Commit**

```bash
git add python/profiler/rbg_profiler/bench/factory.py \
        python/profiler/rbg_profiler/bench/__init__.py \
        python/profiler/tests/test_factory.py
git commit -m "feat(bench): add create_bench_executor factory and update package exports"
```

---

### Task 7: Wire into Profiler + CLI + Update Tests

**Files:**
- Modify: `python/profiler/rbg_profiler/profiler.py:57-118` (use ProfilingPipeline)
- Modify: `python/profiler/rbg_profiler/main.py:30-171` (add --bench-method flag)
- Modify: `python/profiler/tests/test_profiler.py` (update mock for new flow)

**Interfaces:**
- Consumes: `create_bench_executor` from Task 6; `ProfilingPipeline` from Task 5
- Produces: Profiler with `--bench-method` flag, backward-compatible default `http`.

- [ ] **Step 1: Write the failing test for the new profiler flow**

Modify `python/profiler/tests/test_profiler.py` — replace the entire file with:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for profiler integration with bench executors."""

from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rbg_profiler.bench.base import PrefillProfileData, DecodeProfileData
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
    # Decode: 1 ISL point (256), multiple concurrency points
    assert mock_executor.execute.call_count >= 2


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python/profiler && pytest tests/test_profiler.py -v`
Expected: FAIL — `run_profiling` still uses old `create_bench_backend` / `bench_backend.bench_prefill`, test patches `create_bench_executor` which is not yet imported.

- [ ] **Step 3: Update profiler.py to use ProfilingPipeline**

Modify `python/profiler/rbg_profiler/profiler.py`. Replace lines 24 (import) and 57-118 (run_profiling function).

Replace the import line:

```python
from rbg_profiler.bench import create_bench_backend
```

with:

```python
from rbg_profiler.bench import create_bench_executor
from rbg_profiler.bench.pipeline import ProfilingPipeline
```

Replace the `run_profiling` function (lines 57-171) with:

```python
async def run_profiling(args, prefill_endpoint: RoleEndpoint, decode_endpoint: RoleEndpoint, backend: str = "sglang-oai"):
    """Run full SLA profiling pipeline and store results."""
    logger.info("SLA Profiler starting")
    logger.info("  PREFILL_BASE_URL: %s", prefill_endpoint.base_url)
    logger.info("  DECODE_BASE_URL: %s", decode_endpoint.base_url)
    logger.info("  MODEL: %s", args.model_name)
    logger.info("  BACKEND: %s", backend)
    logger.info("  BENCH_METHOD: %s", getattr(args, 'bench_method', 'http'))
    logger.info("  PREFILL_GPUS: %d", prefill_endpoint.num_gpus)
    logger.info("  DECODE_GPUS: %d", decode_endpoint.num_gpus)
    logger.info("  MAX_KV_TOKENS: %d", args.max_kv_tokens)
    logger.info("  MAX_CONTEXT_LENGTH: %d", args.max_context_length)

    os.makedirs(args.output_dir, exist_ok=True)

    bench_method = getattr(args, 'bench_method', 'http')
    executor = create_bench_executor(method=bench_method, backend=backend)
    pipeline = ProfilingPipeline(executor)

    prefill_isl_points = [int(x) for x in args.prefill_isl_points.split(",")]
    decode_isl_points = [int(x) for x in args.decode_isl_points.split(",")]

    # Tokenizer loading priority (spec §Tokenizer Resolution):
    #   1. endpoint.tokenizer (discoverer or operator via --tokenizer)
    #   2. args.model_name as HuggingFace ID fallback
    if not prefill_endpoint.tokenizer:
        prefill_endpoint.tokenizer = args.tokenizer or args.model_name
    if not decode_endpoint.tokenizer:
        decode_endpoint.tokenizer = args.tokenizer or args.model_name
    if not prefill_endpoint.tokenizer or not decode_endpoint.tokenizer:
        logger.error("No tokenizer source: set --tokenizer or --model-name")
        sys.exit(1)

    # Phase 1: Prefill
    logger.info("=" * 60)
    logger.info("PHASE 1: Prefill Profiling")
    logger.info("=" * 60)

    prefill_results = await pipeline.bench_prefill(
        endpoint=prefill_endpoint,
        model=args.model_name,
        isl_points=prefill_isl_points,
        max_context_length=args.max_context_length,
    )

    prefill_data = {
        "prefill_isl": [r.isl for r in prefill_results],
        "prefill_ttft": [r.ttft_ms for r in prefill_results],
        "prefill_thpt_per_gpu": [r.thpt_per_gpu for r in prefill_results],
    }
    logger.info("Prefill profiling done: %d data points", len(prefill_results))

    # Phase 2: Decode
    logger.info("=" * 60)
    logger.info("PHASE 2: Decode Profiling")
    logger.info("=" * 60)

    decode_results = await pipeline.bench_decode(
        endpoint=decode_endpoint,
        model=args.model_name,
        isl_points=decode_isl_points,
        max_kv_tokens=args.max_kv_tokens,
        max_context_length=args.max_context_length,
        concurrency_granularity=args.decode_granularity,
    )

    decode_data = {
        "x_kv_usage": [r.kv_usage for r in decode_results],
        "y_context_length": [r.context_length for r in decode_results],
        "z_itl": [r.itl_ms for r in decode_results],
        "z_thpt_per_gpu": [r.thpt_per_gpu for r in decode_results],
        "max_kv_tokens": args.max_kv_tokens,
    }
    logger.info("Decode profiling done: %d data points", len(decode_results))

    # Save locally
    prefill_path = os.path.join(args.output_dir, "prefill_raw_data.json")
    decode_path = os.path.join(args.output_dir, "decode_raw_data.json")
    with open(prefill_path, "w") as f:
        json.dump(prefill_data, f, indent=2)
    with open(decode_path, "w") as f:
        json.dump(decode_data, f, indent=2)
    logger.info("Results saved to %s", args.output_dir)

    # Print summary
    logger.info("=" * 60)
    logger.info("PROFILING SUMMARY")
    logger.info("=" * 60)
    logger.info("Prefill: %d points", len(prefill_data["prefill_isl"]))
    for i, isl in enumerate(prefill_data["prefill_isl"]):
        logger.info(
            "  ISL=%d: TTFT=%.1fms, thpt/gpu=%.0f tok/s",
            isl, prefill_data["prefill_ttft"][i], prefill_data["prefill_thpt_per_gpu"][i],
        )
    logger.info("Decode: %d points", len(decode_data["x_kv_usage"]))
    if decode_data["z_thpt_per_gpu"]:
        logger.info(
            "  thpt/gpu range: %.0f - %.0f tok/s",
            min(decode_data["z_thpt_per_gpu"]), max(decode_data["z_thpt_per_gpu"]),
        )
        logger.info(
            "  ITL range: %.1f - %.1f ms",
            min(decode_data["z_itl"]), max(decode_data["z_itl"]),
        )

    # Update ConfigMap
    if args.update_configmap:
        _update_configmap(args.output_configmap, args.namespace, prefill_data, decode_data)

    # Print JSON to stdout for recovery from pod logs
    print("\n===PREFILL_RAW_DATA_JSON_START===")
    print(json.dumps(prefill_data, indent=2))
    print("===PREFILL_RAW_DATA_JSON_END===")
    print("\n===DECODE_RAW_DATA_JSON_START===")
    print(json.dumps(decode_data, indent=2))
    print("===DECODE_RAW_DATA_JSON_END===")

    logger.info("Profiling complete!")
```

- [ ] **Step 4: Add --bench-method flag to main.py**

Modify `python/profiler/rbg_profiler/main.py`. Add the `--bench-method` argument after the `--backend` argument (after line 60, the `--engine` hidden alias line). Insert:

```python
    parser.add_argument(
        "--bench-method",
        default=os.environ.get("BENCH_METHOD", "http"),
        choices=["http", "inference-perf"],
        help="Bench execution method (default: http). "
             "'http' uses built-in HTTP benchmarking. "
             "'inference-perf' uses the inference-perf CLI tool.",
    )
```

- [ ] **Step 5: Run profiler tests to verify they pass**

Run: `cd python/profiler && pytest tests/test_profiler.py -v`
Expected: 2 PASSED

- [ ] **Step 6: Run ALL tests to verify no regressions**

Run: `cd python/profiler && pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add python/profiler/rbg_profiler/profiler.py \
        python/profiler/rbg_profiler/main.py \
        python/profiler/tests/test_profiler.py
git commit -m "feat(profiler): wire ProfilingPipeline and --bench-method flag into profiler"
```
