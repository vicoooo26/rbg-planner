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
