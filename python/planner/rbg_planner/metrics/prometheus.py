# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from dynamo/planner/monitoring/traffic_metrics.py for RoleBasedGroup.

"""Prometheus-based metrics adapter with shared query utilities.

Concrete adapters (SGLang, vLLM, Dynamo, ...) inherit from this class
and only need to define metric names, label mappings, and override
methods where query patterns differ.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from prometheus_api_client import PrometheusConnect

from rbg_planner.metrics.base import MetricsAdapter

logger = logging.getLogger(__name__)


@dataclass
class MetricNames:
    """Metric name mapping for a specific inference engine."""

    ttft: str = ""
    itl: str = ""
    request_duration: str = ""
    requests_total: str = ""
    prompt_tokens: str = ""
    generation_tokens: str = ""
    model_label: str = "model_name"


# Default query shape per logical metric. The dispatch helpers fall back
# to these when a metric has no explicit shape override.
DEFAULT_SHAPES: dict[str, str] = {
    "ttft": "histogram_avg",
    "itl": "histogram_avg",
    "request_duration": "histogram_avg",
    "requests_total": "counter_increase",
    "isl": "counter_ratio",
    "osl": "counter_ratio",
}


class PrometheusAdapter(MetricsAdapter):
    """Config-driven Prometheus adapter.

    Built from a MetricNames set, a per-metric shape map, and an optional
    raw-PromQL template map. Each get_* method resolves its logical metric:
    a raw template (if present) wins; otherwise the resolved shape selects
    one of the _query_* helpers.
    """

    def __init__(self, url, names, shapes=None, raw_templates=None):
        self.prom = PrometheusConnect(url=url, disable_ssl=True)
        self.metrics = names
        self.shapes = shapes or {}
        self.raw_templates = raw_templates or {}

    # ── shared query helpers ─────────────────────────────────────────

    def _build_label_filter(
        self,
        model_name: Optional[str] = None,
        role: Optional[str] = None,
    ) -> str:
        filters = []
        if model_name:
            filters.append(f'{self.metrics.model_label}="{model_name}"')
        if role:
            filters.append(f'role="{role}"')
        return "{" + ",".join(filters) + "}" if filters else ""

    def _query_avg_metric(
        self,
        metric_name: str,
        interval: str,
        model_name: Optional[str] = None,
        role: Optional[str] = None,
    ) -> float:
        """Average of a histogram: increase(sum) / increase(count)."""
        lf = self._build_label_filter(model_name, role=role)
        query = f"increase({metric_name}_sum{lf}[{interval}])/increase({metric_name}_count{lf}[{interval}])"
        return self._exec_avg(query, metric_name)

    def _query_increase(
        self,
        metric_name: str,
        interval: str,
        model_name: Optional[str] = None,
        role: Optional[str] = None,
    ) -> float:
        """Sum of increase of a counter metric."""
        lf = self._build_label_filter(model_name, role=role)
        query = f"increase({metric_name}{lf}[{interval}])"
        try:
            result = self.prom.custom_query(query=query)
            if not result:
                return 0.0
            total = 0.0
            for r in result:
                val = float(r["value"][1])
                if not (val != val):  # NaN check
                    total += val
            return total
        except Exception as e:
            logger.error(f"Error querying {metric_name}: {e}")
            return 0.0

    def _query_counter_ratio(
        self,
        numerator: str,
        denominator: str,
        interval: str,
        model_name: Optional[str] = None,
    ) -> float:
        """Ratio of two counter increases: increase(num) / increase(denom)."""
        lf = self._build_label_filter(model_name)
        query = f"increase({numerator}{lf}[{interval}])/increase({denominator}{lf}[{interval}])"
        return self._exec_avg(query, numerator)

    def _query_raw(
        self,
        template: str,
        interval: str,
        model_name: Optional[str] = None,
    ) -> float:
        """Execute a user-provided PromQL template.

        Substitutes {interval} with the window and {labelFilter} with the
        full label selector (or empty string).
        """
        lf = self._build_label_filter(model_name)
        query = template.replace("{interval}", interval).replace("{labelFilter}", lf)
        return self._exec_avg(query, "raw")

    def _exec_avg(self, query: str, label: str) -> float:
        """Execute a PromQL query and return the average of all series."""
        try:
            result = self.prom.custom_query(query=query)
            if not result:
                logger.debug(f"No data for {label}")
                return 0.0
            values = [float(r["value"][1]) for r in result if r["value"][1] != "NaN"]
            if not values:
                return 0.0
            return sum(values) / len(values)
        except Exception as e:
            logger.error(f"Error querying {label}: {e}")
            return 0.0

    # ── dispatch ─────────────────────────────────────────────────────

    def _dispatch_simple(
        self,
        logical: str,
        name: str,
        interval: str,
        model_name: Optional[str],
    ) -> float:
        """Dispatch for metrics using histogram_avg or counter_increase."""
        if logical in self.raw_templates:
            return self._query_raw(self.raw_templates[logical], interval, model_name)
        shape = self.shapes.get(logical, DEFAULT_SHAPES[logical])
        if shape == "histogram_avg":
            return self._query_avg_metric(name, interval, model_name)
        if shape == "counter_increase":
            return self._query_increase(name, interval, model_name)
        raise ValueError(f"Unsupported shape {shape!r} for {logical!r}")

    def _dispatch_ratio(
        self,
        logical: str,
        token_name: str,
        interval: str,
        model_name: Optional[str],
    ) -> float:
        """Dispatch for isl/osl using counter_ratio or histogram_avg."""
        if logical in self.raw_templates:
            return self._query_raw(self.raw_templates[logical], interval, model_name)
        shape = self.shapes.get(logical, DEFAULT_SHAPES[logical])
        if shape == "counter_ratio":
            return self._query_counter_ratio(
                token_name,
                self.metrics.requests_total,
                interval,
                model_name,
            )
        if shape == "histogram_avg":
            return self._query_avg_metric(token_name, interval, model_name)
        raise ValueError(f"Unsupported shape {shape!r} for {logical!r}")

    # ── MetricsAdapter interface ─────────────────────────────────────

    def get_avg_ttft(self, interval: str, model_name: Optional[str] = None) -> float:
        return self._dispatch_simple("ttft", self.metrics.ttft, interval, model_name)

    def get_avg_itl(self, interval: str, model_name: Optional[str] = None) -> float:
        return self._dispatch_simple("itl", self.metrics.itl, interval, model_name)

    def get_avg_request_duration(self, interval: str, model_name: Optional[str] = None) -> float:
        return self._dispatch_simple(
            "request_duration",
            self.metrics.request_duration,
            interval,
            model_name,
        )

    def get_request_count(self, interval: str, model_name: Optional[str] = None) -> float:
        return self._dispatch_simple(
            "requests_total",
            self.metrics.requests_total,
            interval,
            model_name,
        )

    def get_avg_isl(self, interval: str, model_name: Optional[str] = None) -> float:
        return self._dispatch_ratio(
            "isl",
            self.metrics.prompt_tokens,
            interval,
            model_name,
        )

    def get_avg_osl(self, interval: str, model_name: Optional[str] = None) -> float:
        return self._dispatch_ratio(
            "osl",
            self.metrics.generation_tokens,
            interval,
            model_name,
        )
