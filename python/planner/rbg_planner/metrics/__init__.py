# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable metrics adapter module.

Provides a config-driven PrometheusAdapter built from built-in engine
defaults, optionally overridden by a metricConfig from the CRD.

Usage::

    from rbg_planner.metrics import create_metrics_adapter

    adapter = create_metrics_adapter("sglang", "http://prometheus:9090")
    ttft = adapter.get_avg_ttft("180s", model_name="Qwen/Qwen3-0.6B")
"""

from rbg_planner.metrics.base import MetricsAdapter
from rbg_planner.metrics.defaults import BUILTIN_ENGINES
from rbg_planner.metrics.prometheus import PrometheusAdapter

SUPPORTED_SOURCES = list(BUILTIN_ENGINES.keys())


def create_metrics_adapter(source: str, prometheus_url: str) -> MetricsAdapter:
    """Create a metrics adapter for the given source.

    Args:
        source: Metric source identifier (sglang, vllm, dynamo, patio).
        prometheus_url: Prometheus server URL.

    Raises:
        ValueError: If the source is not a built-in engine.
    """
    if source not in BUILTIN_ENGINES:
        raise ValueError(f"Unsupported metric source: {source}. Supported: {SUPPORTED_SOURCES}")
    default = BUILTIN_ENGINES[source]
    return PrometheusAdapter(
        prometheus_url,
        default["names"],
        dict(default["shapes"]),
        {},
    )
