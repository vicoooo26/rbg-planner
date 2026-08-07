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

import dataclasses
from typing import Any, Optional

from rbg_planner.metrics.base import MetricsAdapter
from rbg_planner.metrics.defaults import BUILTIN_ENGINES
from rbg_planner.metrics.prometheus import MetricNames, PrometheusAdapter

SUPPORTED_SOURCES = list(BUILTIN_ENGINES.keys())

# CRD camelCase metric-name keys -> MetricNames snake_case fields.
_METRIC_FIELD_MAP = {
    "ttft": "ttft",
    "itl": "itl",
    "requestDuration": "request_duration",
    "requestsTotal": "requests_total",
    "promptTokens": "prompt_tokens",
    "generationTokens": "generation_tokens",
}

# CRD camelCase shape keys -> internal snake_case logical-metric keys.
_SHAPE_KEY_MAP = {
    "ttft": "ttft",
    "itl": "itl",
    "requestDuration": "request_duration",
    "requestsTotal": "requests_total",
    "isl": "isl",
    "osl": "osl",
}

_REQUIRED_NAME_FIELDS = [
    "ttft",
    "itl",
    "request_duration",
    "requests_total",
    "prompt_tokens",
    "generation_tokens",
]


def _validate_complete(names: MetricNames) -> None:
    missing = [f for f in _REQUIRED_NAME_FIELDS if not getattr(names, f)]
    if missing:
        raise ValueError(f"Custom engine missing required metric names: {missing}")


def create_metrics_adapter(
    source: str,
    prometheus_url: str,
    metric_config: Optional[dict[str, Any]] = None,
) -> MetricsAdapter:
    """Create a metrics adapter for the given source.

    Args:
        source: Metric source identifier. Built-in: sglang, vllm, dynamo,
            patio. Any other value is a custom engine and requires
            ``metric_config`` with all metric names defined.
        prometheus_url: Prometheus server URL.
        metric_config: Optional override dict mirroring the CRD MetricConfig
            (modelLabel, metrics, metricShapes, rawTemplates).

    Raises:
        ValueError: If the source is unknown and no metric_config is given,
            or a custom engine's metric_config is missing required names.
    """
    is_builtin = source in BUILTIN_ENGINES
    if not is_builtin and not metric_config:
        raise ValueError(
            f"Unsupported metric source: {source}. Provide metricConfig to "
            f"define a custom engine, or use one of: {SUPPORTED_SOURCES}"
        )

    if is_builtin:
        base = BUILTIN_ENGINES[source]
        names = dataclasses.replace(base["names"])
        shapes = dict(base["shapes"])
    else:
        names = MetricNames()
        shapes = {}

    raw_templates: dict[str, str] = {}

    if metric_config:
        if "modelLabel" in metric_config:
            names = dataclasses.replace(names, model_label=metric_config["modelLabel"])
        overrides = {
            _METRIC_FIELD_MAP[k]: v for k, v in metric_config.get("metrics", {}).items() if k in _METRIC_FIELD_MAP
        }
        if overrides:
            names = dataclasses.replace(names, **overrides)
        for k, v in metric_config.get("metricShapes", {}).items():
            if k in _SHAPE_KEY_MAP:
                shapes[_SHAPE_KEY_MAP[k]] = v
        raw_templates = dict(metric_config.get("rawTemplates", {}))

    if not is_builtin:
        _validate_complete(names)

    return PrometheusAdapter(prometheus_url, names, shapes, raw_templates)
