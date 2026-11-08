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
