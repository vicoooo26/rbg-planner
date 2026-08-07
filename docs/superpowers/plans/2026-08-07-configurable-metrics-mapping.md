# Configurable Metrics Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make rbg-planner's inference-engine metric names and query shapes user-configurable via the CRD, so adapting to a vLLM version rename or onboarding a custom engine is a config change, not a code change.

**Architecture:** Collapse the four hardcoded engine adapter classes into a single config-driven `PrometheusAdapter` backed by a built-in defaults registry. Each logical metric dispatches to one of the existing `_query_*` helpers based on a per-metric "shape", with a raw-PromQL template escape hatch. The CRD `MetricsEndpointConfig` gains an optional `metricConfig` sub-field; the controller serializes it to a `METRIC_MAPPING_JSON` env var that the planner merges over the built-in defaults.

**Tech Stack:** Python 3 (pytest, prometheus_api_client), Go (kubebuilder/controller-runtime, controller-gen), Kubernetes CRD.

## Global Constraints

- Apache-2.0 license header on all new source files (copy the SPDX header from existing files).
- Python files live under `python/planner/rbg_planner/`; tests under `python/planner/tests/`.
- Run Python tests with `cd python/planner && pytest tests/ -v`.
- Run Go tests with `go test ./internal/controller/ -v -run <TestName>`.
- Regenerate Go deepcopy with `make generate` and CRD manifests with `make manifests` after editing `api/v1alpha1/roleautoscaler_types.go`.
- Backward compatibility: a CRD with only `metricSource` (no `metricConfig`) must reproduce today's behavior exactly.

---

## File Structure

**Python (create/modify):**
- Create `python/planner/rbg_planner/metrics/defaults.py` — built-in engine defaults registry (names + shape overrides) extracted from the four old subclasses.
- Modify `python/planner/rbg_planner/metrics/prometheus.py` — config-driven `PrometheusAdapter`: add `DEFAULT_SHAPES`, shape/raw state, `_query_raw`, dispatch helpers; rewrite `get_*` methods.
- Modify `python/planner/rbg_planner/metrics/__init__.py` — `create_metrics_adapter(source, url, metric_config=None)` factory with merge logic + custom-engine validation.
- Delete `python/planner/rbg_planner/metrics/sglang.py`, `vllm.py`, `dynamo.py`, `patio.py` — replaced by `defaults.py`.
- Modify `python/planner/rbg_planner/config.py` — add `metric_mapping_json` field.
- Modify `python/planner/rbg_planner/planner.py` — parse `METRIC_MAPPING_JSON`, pass `metric_config` to factory.
- Modify `python/planner/rbg_planner/main.py` — relax `--metric-source` choices, add `--metric-mapping-json` arg.
- Modify `python/planner/tests/test_prometheus_client.py` — refactor isinstance tests to behavior checks; add shape/raw/merge tests.

**Go (modify):**
- Modify `api/v1alpha1/roleautoscaler_types.go` — extend `MetricsEndpointConfig`, add `MetricConfig` type, drop `metricSource` enum.
- Regenerate `api/v1alpha1/zz_generated.deepcopy.go` and `config/crd/inference-extension.rolebasedgroup.io_autoscalers.yaml`.
- Modify `internal/controller/roleautoscaler_controller.go` — parse `metricConfig`, marshal to JSON, emit `METRIC_MAPPING_JSON` env var.
- Modify `internal/controller/roleautoscaler_controller_test.go` — add tests for `metricConfig` parsing + env var.
- Modify `config/samples/roleautoscaler_full.yaml` — add `metricConfig` example.

---

### Task 1: Built-in engine defaults registry

**Files:**
- Create: `python/planner/rbg_planner/metrics/defaults.py`
- Test: `python/planner/tests/test_defaults.py`

**Interfaces:**
- Produces: `defaults.BUILTIN_ENGINES` — `dict[str, dict]` keyed by engine name, each value `{"names": MetricNames, "shapes": dict[str,str]}`. `defaults.DEFAULT_SHAPES` is defined in `prometheus.py` (Task 2 reuses it) — for this task, also add `DEFAULT_SHAPES` to `prometheus.py`.

- [ ] **Step 1: Add `DEFAULT_SHAPES` to `prometheus.py`**

Add this constant below the `MetricNames` dataclass in `python/planner/rbg_planner/metrics/prometheus.py` (after line 34):

```python
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
```

- [ ] **Step 2: Write the failing test**

Create `python/planner/tests/test_defaults.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the built-in engine defaults registry."""

from rbg_planner.metrics.defaults import BUILTIN_ENGINES
from rbg_planner.metrics.prometheus import DEFAULT_SHAPES


class TestBuiltinEngines:
    def test_sglang_names(self):
        n = BUILTIN_ENGINES["sglang"]["names"]
        assert n.ttft == "sglang_time_to_first_token_seconds"
        assert n.itl == "sglang_inter_token_latency_seconds"
        assert n.request_duration == "sglang_e2e_request_latency_seconds"
        assert n.requests_total == "sglang_num_requests_total"
        assert n.prompt_tokens == "sglang_prompt_tokens_total"
        assert n.generation_tokens == "sglang_generation_tokens_total"
        assert n.model_label == "model_name"

    def test_vllm_names(self):
        n = BUILTIN_ENGINES["vllm"]["names"]
        assert n.ttft == "vllm:time_to_first_token_seconds"
        assert n.requests_total == "vllm:num_requests_total"
        assert n.model_label == "model_name"

    def test_dynamo_names_and_shapes(self):
        e = BUILTIN_ENGINES["dynamo"]
        n = e["names"]
        assert n.ttft == "dynamo_frontend_time_to_first_token_seconds"
        assert n.prompt_tokens == "dynamo_frontend_input_sequence_tokens"
        assert n.generation_tokens == "dynamo_frontend_output_sequence_tokens"
        assert n.model_label == "model"
        # Dynamo overrides ISL/OSL to histogram_avg (replaces the old subclass override).
        assert e["shapes"]["isl"] == "histogram_avg"
        assert e["shapes"]["osl"] == "histogram_avg"

    def test_patio_names(self):
        n = BUILTIN_ENGINES["patio"]["names"]
        assert n.ttft == "patio:time_to_first_token_seconds"
        assert n.requests_total == "patio:requests_total"

    def test_builtin_shapes_default_empty(self):
        # sglang/vllm/patio use all default shapes (no overrides).
        for engine in ("sglang", "vllm", "patio"):
            assert BUILTIN_ENGINES[engine]["shapes"] == {}

    def test_default_shapes_complete(self):
        assert set(DEFAULT_SHAPES.keys()) == {
            "ttft", "itl", "request_duration", "requests_total", "isl", "osl",
        }
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd python/planner && pytest tests/test_defaults.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rbg_planner.metrics.defaults'`

- [ ] **Step 4: Create `defaults.py`**

Create `python/planner/rbg_planner/metrics/defaults.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in engine defaults: metric names + shape overrides per engine.

Replaces the former per-engine adapter subclasses (SGLangAdapter, VLLMAdapter,
DynamoAdapter, PatioAdapter). The factory merges user-provided metricConfig
over these defaults.
"""

from rbg_planner.metrics.prometheus import MetricNames

# Engine name -> {"names": MetricNames, "shapes": dict[str, str]}.
# An empty "shapes" dict means all metrics use DEFAULT_SHAPES.
BUILTIN_ENGINES: dict[str, dict] = {
    "sglang": {
        "names": MetricNames(
            ttft="sglang_time_to_first_token_seconds",
            itl="sglang_inter_token_latency_seconds",
            request_duration="sglang_e2e_request_latency_seconds",
            requests_total="sglang_num_requests_total",
            prompt_tokens="sglang_prompt_tokens_total",
            generation_tokens="sglang_generation_tokens_total",
            model_label="model_name",
        ),
        "shapes": {},
    },
    "vllm": {
        "names": MetricNames(
            ttft="vllm:time_to_first_token_seconds",
            itl="vllm:time_per_output_token_seconds",
            request_duration="vllm:e2e_request_latency_seconds",
            requests_total="vllm:num_requests_total",
            prompt_tokens="vllm:prompt_tokens_total",
            generation_tokens="vllm:generation_tokens_total",
            model_label="model_name",
        ),
        "shapes": {},
    },
    "dynamo": {
        "names": MetricNames(
            ttft="dynamo_frontend_time_to_first_token_seconds",
            itl="dynamo_frontend_inter_token_latency_seconds",
            request_duration="dynamo_frontend_request_duration_seconds",
            requests_total="dynamo_frontend_requests_total",
            prompt_tokens="dynamo_frontend_input_sequence_tokens",
            generation_tokens="dynamo_frontend_output_sequence_tokens",
            model_label="model",
        ),
        # Dynamo exposes ISL/OSL as histograms, not counter ratios.
        "shapes": {"isl": "histogram_avg", "osl": "histogram_avg"},
    },
    "patio": {
        "names": MetricNames(
            ttft="patio:time_to_first_token_seconds",
            itl="patio:inter_token_latency_seconds",
            request_duration="patio:request_duration_seconds",
            requests_total="patio:requests_total",
            prompt_tokens="patio:prompt_tokens_total",
            generation_tokens="patio:generation_tokens_total",
            model_label="model_name",
        ),
        "shapes": {},
    },
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd python/planner && pytest tests/test_defaults.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add python/planner/rbg_planner/metrics/defaults.py \
        python/planner/rbg_planner/metrics/prometheus.py \
        python/planner/tests/test_defaults.py
git commit -m "feat(metrics): add built-in engine defaults registry"
```

---

### Task 2: Config-driven PrometheusAdapter + remove subclasses

**Files:**
- Modify: `python/planner/rbg_planner/metrics/prometheus.py`
- Modify: `python/planner/rbg_planner/metrics/__init__.py`
- Delete: `python/planner/rbg_planner/metrics/sglang.py`, `vllm.py`, `dynamo.py`, `patio.py`
- Test: `python/planner/tests/test_prometheus_client.py`

**Interfaces:**
- Consumes: `defaults.BUILTIN_ENGINES`, `DEFAULT_SHAPES` (from Task 1).
- Produces: `PrometheusAdapter(url, names, shapes, raw_templates)` — constructed directly by the factory. `create_metrics_adapter(source, url)` (no merge yet — Task 3 adds `metric_config`).

- [ ] **Step 1: Write the failing tests for shape dispatch and raw**

Append these tests to `python/planner/tests/test_prometheus_client.py` (inside new classes at the end of the file). Add the import `from rbg_planner.metrics.prometheus import PrometheusAdapter, MetricNames` at the top, and `from rbg_planner.metrics.defaults import BUILTIN_ENGINES`:

```python
class TestShapeDispatch:
    def test_ttft_uses_histogram_query(self, mock_prom):
        mock_prom.custom_query.return_value = []
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        adapter.get_avg_ttft("180s", "Qwen")
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert query == (
            'increase(sglang_time_to_first_token_seconds_sum{model_name="Qwen"}[180s])'
            '/increase(sglang_time_to_first_token_seconds_count{model_name="Qwen"}[180s])'
        )

    def test_request_count_uses_increase_query(self, mock_prom):
        mock_prom.custom_query.return_value = []
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        adapter.get_request_count("180s")
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert query == "increase(sglang_num_requests_total[180s])"

    def test_dynamo_isl_uses_histogram(self, mock_prom):
        mock_prom.custom_query.return_value = []
        adapter = create_metrics_adapter("dynamo", "http://prom:9090")
        adapter.get_avg_isl("180s", "Qwen")
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert query == (
            'increase(dynamo_frontend_input_sequence_tokens_sum{model="Qwen"}[180s])'
            '/increase(dynamo_frontend_input_sequence_tokens_count{model="Qwen"}[180s])'
        )

    def test_sglang_isl_uses_counter_ratio(self, mock_prom):
        mock_prom.custom_query.return_value = []
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        adapter.get_avg_isl("180s", "Qwen")
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert query == (
            'increase(sglang_prompt_tokens_total{model_name="Qwen"}[180s])'
            '/increase(sglang_num_requests_total{model_name="Qwen"}[180s])'
        )


class TestRawTemplates:
    def test_raw_substitutes_label_filter_and_interval(self, mock_prom):
        mock_prom.custom_query.return_value = [{"value": [0, "0.5"]}]
        adapter = PrometheusAdapter(
            "http://prom:9090", MetricNames(), {},
            {"ttft": "avg(custom_ttft{labelFilter})"},
        )
        result = adapter.get_avg_ttft("180s", "Qwen")
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert query == 'avg(custom_ttft{model_name="Qwen"})'
        assert result == 0.5

    def test_raw_empty_label_filter(self, mock_prom):
        mock_prom.custom_query.return_value = [{"value": [0, "0.5"]}]
        adapter = PrometheusAdapter(
            "http://prom:9090", MetricNames(), {},
            {"ttft": "increase(custom{labelFilter}[{interval}])"},
        )
        adapter.get_avg_ttft("180s")
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert query == "increase(custom[180s])"

    def test_raw_overrides_shape(self, mock_prom):
        # Even with a shape set, a raw template takes precedence.
        mock_prom.custom_query.return_value = [{"value": [0, "1.0"]}]
        adapter = PrometheusAdapter(
            "http://prom:9090", MetricNames(), {"ttft": "histogram_avg"},
            {"ttft": "avg(custom{labelFilter})"},
        )
        adapter.get_avg_ttft("180s", "Qwen")
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert query == 'avg(custom{model_name="Qwen"})'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python/planner && pytest tests/test_prometheus_client.py::TestShapeDispatch tests/test_prometheus_client.py::TestRawTemplates -v`
Expected: FAIL — `PrometheusAdapter` does not accept `(names, shapes, raw_templates)` and `create_metrics_adapter` still returns subclass instances.

- [ ] **Step 3: Rewrite `prometheus.py` adapter to be config-driven**

Replace the `PrometheusAdapter` class (lines 37–155 of `python/planner/rbg_planner/metrics/prometheus.py`) with the version below. Keep the `MetricNames` dataclass and `DEFAULT_SHAPES` (added in Task 1) and the module header/imports unchanged. The three `_query_*` helpers, `_build_label_filter`, and `_exec_avg` are unchanged; only `__init__`, the new dispatch helpers, `_query_raw`, and the `get_*` methods change:

```python
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
        self, model_name: Optional[str] = None, role: Optional[str] = None,
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
        query = (
            f"increase({metric_name}_sum{lf}[{interval}])"
            f"/increase({metric_name}_count{lf}[{interval}])"
        )
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
        query = (
            f"increase({numerator}{lf}[{interval}])"
            f"/increase({denominator}{lf}[{interval}])"
        )
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
        self, logical: str, name: str, interval: str, model_name: Optional[str],
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
        self, logical: str, token_name: str, interval: str, model_name: Optional[str],
    ) -> float:
        """Dispatch for isl/osl using counter_ratio or histogram_avg."""
        if logical in self.raw_templates:
            return self._query_raw(self.raw_templates[logical], interval, model_name)
        shape = self.shapes.get(logical, DEFAULT_SHAPES[logical])
        if shape == "counter_ratio":
            return self._query_counter_ratio(
                token_name, self.metrics.requests_total, interval, model_name,
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
            "request_duration", self.metrics.request_duration, interval, model_name,
        )

    def get_request_count(self, interval: str, model_name: Optional[str] = None) -> float:
        return self._dispatch_simple(
            "requests_total", self.metrics.requests_total, interval, model_name,
        )

    def get_avg_isl(self, interval: str, model_name: Optional[str] = None) -> float:
        return self._dispatch_ratio(
            "isl", self.metrics.prompt_tokens, interval, model_name,
        )

    def get_avg_osl(self, interval: str, model_name: Optional[str] = None) -> float:
        return self._dispatch_ratio(
            "osl", self.metrics.generation_tokens, interval, model_name,
        )
```

- [ ] **Step 4: Rewrite the factory in `__init__.py`**

Replace the entire contents of `python/planner/rbg_planner/metrics/__init__.py` with:

```python
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
        raise ValueError(
            f"Unsupported metric source: {source}. "
            f"Supported: {SUPPORTED_SOURCES}"
        )
    default = BUILTIN_ENGINES[source]
    return PrometheusAdapter(
        prometheus_url, default["names"], dict(default["shapes"]), {},
    )
```

- [ ] **Step 5: Delete the four subclass files**

```bash
git rm python/planner/rbg_planner/metrics/sglang.py \
       python/planner/rbg_planner/metrics/vllm.py \
       python/planner/rbg_planner/metrics/dynamo.py \
       python/planner/rbg_planner/metrics/patio.py
```

- [ ] **Step 6: Refactor existing isinstance tests to behavior checks**

In `python/planner/tests/test_prometheus_client.py`, replace the imports (lines 7–11) and the `TestFactory` class (lines 22–41) with behavior-based checks. New imports:

```python
from rbg_planner.metrics import SUPPORTED_SOURCES, create_metrics_adapter
from rbg_planner.metrics.base import MetricsAdapter
from rbg_planner.metrics.defaults import BUILTIN_ENGINES
from rbg_planner.metrics.prometheus import MetricNames, PrometheusAdapter
```

New `TestFactory` class (replaces the old one):

```python
class TestFactory:
    def test_create_sglang(self, mock_prom):
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        assert isinstance(adapter, PrometheusAdapter)
        assert adapter.metrics.ttft == "sglang_time_to_first_token_seconds"

    def test_create_vllm(self, mock_prom):
        adapter = create_metrics_adapter("vllm", "http://prom:9090")
        assert isinstance(adapter, PrometheusAdapter)
        assert adapter.metrics.ttft == "vllm:time_to_first_token_seconds"

    def test_create_dynamo(self, mock_prom):
        adapter = create_metrics_adapter("dynamo", "http://prom:9090")
        assert isinstance(adapter, PrometheusAdapter)
        assert adapter.metrics.model_label == "model"
        assert adapter.shapes["isl"] == "histogram_avg"

    def test_invalid_source_raises(self, mock_prom):
        with pytest.raises(ValueError, match="Unsupported metric source"):
            create_metrics_adapter("invalid", "http://prom:9090")

    def test_all_builtins_create_metrics_adapter(self, mock_prom):
        mock_prom.custom_query.return_value = []
        for name in SUPPORTED_SOURCES:
            adapter = create_metrics_adapter(name, "http://prom:9090")
            assert isinstance(adapter, MetricsAdapter)
```

- [ ] **Step 7: Run the full Python test suite**

Run: `cd python/planner && pytest tests/ -v`
Expected: PASS — all existing tests (TestLabelFilter, TestQueryMetrics) plus new TestShapeDispatch, TestRawTemplates, and refactored TestFactory.

- [ ] **Step 8: Commit**

```bash
git add python/planner/rbg_planner/metrics/prometheus.py \
        python/planner/rbg_planner/metrics/__init__.py \
        python/planner/tests/test_prometheus_client.py
git commit -m "feat(metrics): config-driven PrometheusAdapter with shape dispatch"
```

---

### Task 3: Factory merge logic + custom engine validation

**Files:**
- Modify: `python/planner/rbg_planner/metrics/__init__.py`
- Test: `python/planner/tests/test_metric_config.py`

**Interfaces:**
- Consumes: `defaults.BUILTIN_ENGINES`, `prometheus.MetricNames`, `prometheus.PrometheusAdapter`.
- Produces: `create_metrics_adapter(source, url, metric_config=None)` — `metric_config` is a dict mirroring the CRD `MetricConfig` JSON: `{"modelLabel", "metrics": {…}, "metricShapes": {…}, "rawTemplates": {…}}`. CRD camelCase keys are mapped to Python snake_case.

- [ ] **Step 1: Write the failing tests**

Create `python/planner/tests/test_metric_config.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for metric_config merge logic and custom-engine validation."""

from unittest.mock import MagicMock, patch

import pytest

from rbg_planner.metrics import create_metrics_adapter


@pytest.fixture
def mock_prom():
    with patch("rbg_planner.metrics.prometheus.PrometheusConnect") as mock_cls:
        mock_conn = MagicMock()
        mock_cls.return_value = mock_conn
        yield mock_conn


class TestMergeOverride:
    def test_override_single_metric_name(self, mock_prom):
        cfg = {"metrics": {"ttft": "vllm:time_to_first_token_seconds_v2"}}
        adapter = create_metrics_adapter("vllm", "http://prom:9090", cfg)
        assert adapter.metrics.ttft == "vllm:time_to_first_token_seconds_v2"
        # Unspecified names fall back to the built-in default.
        assert adapter.metrics.itl == "vllm:time_per_output_token_seconds"

    def test_override_model_label(self, mock_prom):
        cfg = {"modelLabel": "model"}
        adapter = create_metrics_adapter("vllm", "http://prom:9090", cfg)
        assert adapter.metrics.model_label == "model"

    def test_override_shape(self, mock_prom):
        cfg = {"metricShapes": {"isl": "histogram_avg"}}
        adapter = create_metrics_adapter("vllm", "http://prom:9090", cfg)
        assert adapter.shapes["isl"] == "histogram_avg"
        # osl unchanged (default counter_ratio, not in shapes map).
        assert "osl" not in adapter.shapes

    def test_raw_templates_added(self, mock_prom):
        cfg = {"rawTemplates": {"ttft": "avg(custom{labelFilter})"}}
        adapter = create_metrics_adapter("vllm", "http://prom:9090", cfg)
        assert adapter.raw_templates["ttft"] == "avg(custom{labelFilter})"

    def test_dynamo_shape_override_still_mergeable(self, mock_prom):
        # User can override dynamo's histogram ISL back to counter_ratio.
        cfg = {"metricShapes": {"isl": "counter_ratio"}}
        adapter = create_metrics_adapter("dynamo", "http://prom:9090", cfg)
        assert adapter.shapes["isl"] == "counter_ratio"

    def test_merged_raw_dispatch_uses_override(self, mock_prom):
        mock_prom.custom_query.return_value = [{"value": [0, "0.5"]}]
        cfg = {"rawTemplates": {"ttft": "avg(custom{labelFilter})"}}
        adapter = create_metrics_adapter("vllm", "http://prom:9090", cfg)
        adapter.get_avg_ttft("180s", "Qwen")
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert query == 'avg(custom{model_name="Qwen"})'


class TestCustomEngine:
    def test_custom_engine_complete(self, mock_prom):
        cfg = {
            "metrics": {
                "ttft": "acme_ttft", "itl": "acme_itl",
                "requestDuration": "acme_dur", "requestsTotal": "acme_reqs",
                "promptTokens": "acme_prompt", "generationTokens": "acme_gen",
            },
        }
        adapter = create_metrics_adapter("acme-llm", "http://prom:9090", cfg)
        assert adapter.metrics.ttft == "acme_ttft"
        assert adapter.metrics.model_label == "model_name"  # default retained

    def test_custom_engine_missing_names_raises(self, mock_prom):
        cfg = {"metrics": {"ttft": "acme_ttft"}}  # missing the rest
        with pytest.raises(ValueError, match="missing required metric names"):
            create_metrics_adapter("acme-llm", "http://prom:9090", cfg)

    def test_custom_engine_no_config_raises(self, mock_prom):
        with pytest.raises(ValueError, match="Unsupported metric source"):
            create_metrics_adapter("acme-llm", "http://prom:9090")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd python/planner && pytest tests/test_metric_config.py -v`
Expected: FAIL — `create_metrics_adapter() takes 2 positional arguments but 3 were given`.

- [ ] **Step 3: Implement merge logic in `__init__.py`**

Replace the entire contents of `python/planner/rbg_planner/metrics/__init__.py` with:

```python
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
    "ttft", "itl", "request_duration",
    "requests_total", "prompt_tokens", "generation_tokens",
]


def _validate_complete(names: MetricNames) -> None:
    missing = [f for f in _REQUIRED_NAME_FIELDS if not getattr(names, f)]
    if missing:
        raise ValueError(
            f"Custom engine missing required metric names: {missing}"
        )


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
            _METRIC_FIELD_MAP[k]: v
            for k, v in metric_config.get("metrics", {}).items()
            if k in _METRIC_FIELD_MAP
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd python/planner && pytest tests/test_metric_config.py tests/test_prometheus_client.py tests/test_defaults.py -v`
Expected: PASS — all tests including the existing `test_invalid_source_raises` (still matches "Unsupported metric source").

- [ ] **Step 5: Commit**

```bash
git add python/planner/rbg_planner/metrics/__init__.py \
        python/planner/tests/test_metric_config.py
git commit -m "feat(metrics): merge metricConfig over built-in defaults, validate custom engines"
```

---

### Task 4: Python wiring (config, planner, main)

**Files:**
- Modify: `python/planner/rbg_planner/config.py`
- Modify: `python/planner/rbg_planner/planner.py:160-164`
- Modify: `python/planner/rbg_planner/main.py:27-30,63-109`
- Test: `python/planner/tests/test_config.py` (existing — verify it still passes)

**Interfaces:**
- Consumes: `create_metrics_adapter(source, url, metric_config)` from Task 3.
- Produces: `PlannerConfig.metric_mapping_json` (env `METRIC_MAPPING_JSON`); planner passes parsed `metric_config` to the factory.

- [ ] **Step 1: Add `metric_mapping_json` to `config.py`**

In `python/planner/rbg_planner/config.py`, add a field after `metric_source` (line 26) and re-read it in `__init__`. After line 27 add:

```python
    metric_mapping_json: str = os.environ.get("METRIC_MAPPING_JSON", "")
```

In `__init__` (after line 58), add:

```python
        self.metric_mapping_json = os.environ.get("METRIC_MAPPING_JSON", "")
```

- [ ] **Step 2: Pass `metric_config` in `planner.py`**

In `python/planner/rbg_planner/planner.py`, add `import json` near the top imports. Replace the metrics-client construction (lines 160–164):

```python
        # Metrics adapter for querying inference metrics
        metric_config = None
        if config.metric_mapping_json:
            try:
                metric_config = json.loads(config.metric_mapping_json)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid METRIC_MAPPING_JSON, ignoring: {e}")
        self.metrics_client = create_metrics_adapter(
            source=config.metric_source,
            prometheus_url=config.prometheus_endpoint,
            metric_config=metric_config,
        )
```

- [ ] **Step 3: Relax `--metric-source` and add `--metric-mapping-json` in `main.py`**

In `python/planner/rbg_planner/main.py`, replace the `--metric-source` argument (lines 27–30):

```python
    parser.add_argument(
        "--metric-source", type=str, default=None,
        help="Metric source (sglang, vllm, dynamo, patio, or a custom engine name)",
    )
    parser.add_argument(
        "--metric-mapping-json", type=str, default=None,
        help="JSON metricConfig overriding metric names/shapes/templates",
    )
```

In `build_config`, after the `metric_source` override (line 79), add:

```python
    if args.metric_mapping_json is not None:
        config.metric_mapping_json = args.metric_mapping_json
```

- [ ] **Step 4: Run the full Python test suite**

Run: `cd python/planner && pytest tests/ -v`
Expected: PASS — including `tests/test_config.py`.

- [ ] **Step 5: Commit**

```bash
git add python/planner/rbg_planner/config.py \
        python/planner/rbg_planner/planner.py \
        python/planner/rbg_planner/main.py
git commit -m "feat(planner): wire metricConfig from env/CLI through to the adapter factory"
```

---

### Task 5: Go CRD types + regenerate

**Files:**
- Modify: `api/v1alpha1/roleautoscaler_types.go:186-198`
- Regenerate: `api/v1alpha1/zz_generated.deepcopy.go`, `config/crd/inference-extension.rolebasedgroup.io_autoscalers.yaml`

**Interfaces:**
- Produces: `MetricsEndpointConfig.MetricConfig *MetricConfig` and the `MetricConfig` struct with `ModelLabel`, `Metrics`, `MetricShapes`, `RawTemplates` fields, for CRD schema generation. `MetricSource` loses its enum.

- [ ] **Step 1: Edit the Go types**

In `api/v1alpha1/roleautoscaler_types.go`, replace the `MetricsEndpointConfig` struct (lines 186–198) with:

```go
// MetricsEndpointConfig configures metrics collection.
type MetricsEndpointConfig struct {
	// MetricSource selects the metric source type. Built-in: sglang, vllm,
	// dynamo, patio. Any other value is a custom engine whose metricConfig
	// must define all metric names.
	// +optional
	// +kubebuilder:default="sglang"
	MetricSource string `json:"metricSource,omitempty"`

	// Port is the port for planner's own Prometheus metrics exposition.
	// +optional
	// +kubebuilder:default=9091
	Port int `json:"port,omitempty"`

	// MetricConfig optionally overrides metric names, query shapes, and raw
	// PromQL templates for the selected MetricSource. Omitted means use the
	// built-in defaults for MetricSource.
	// +optional
	MetricConfig *MetricConfig `json:"metricConfig,omitempty"`
}

// MetricConfig overrides the metric mapping for a single engine.
type MetricConfig struct {
	// ModelLabel is the Prometheus label used to filter by model name.
	// Defaults to "model_name" (dynamo built-in uses "model").
	// +optional
	ModelLabel string `json:"modelLabel,omitempty"`

	// Metrics overrides metric names. Keys: ttft, itl, requestDuration,
	// requestsTotal, promptTokens, generationTokens. Subset allowed; merges
	// over built-in defaults.
	// +optional
	Metrics map[string]string `json:"metrics,omitempty"`

	// MetricShapes overrides the query shape per logical metric. Keys: ttft,
	// itl, requestDuration, requestsTotal, isl, osl. Values: histogram_avg,
	// counter_increase, counter_ratio.
	// +optional
	MetricShapes map[string]string `json:"metricShapes,omitempty"`

	// RawTemplates provides a full PromQL template per logical metric. When a
	// metric has an entry here, it overrides the shape dispatch entirely.
	// Templates may use {interval} and {labelFilter} placeholders.
	// +optional
	RawTemplates map[string]string `json:"rawTemplates,omitempty"`
}
```

- [ ] **Step 2: Regenerate deepcopy and CRD manifests**

Run:

```bash
make generate
make manifests
```

Expected: `zz_generated.deepcopy.go` gains `MetricConfig` deepcopy methods; the CRD YAML gains `metricConfig` with `modelLabel`, `metrics`, `metricShapes`, `rawTemplates` properties, and `metricSource` loses its `enum` constraint.

- [ ] **Step 3: Verify the generated CRD schema**

Run: `grep -A3 'metricSource:' config/crd/inference-extension.rolebasedgroup.io_autoscalers.yaml | head -8`
Expected: `metricSource` has no `enum:` line (just `default: sglang` and `type: string`).

Run: `grep -A2 'metricConfig:' config/crd/inference-extension.rolebasedgroup.io_autoscalers.yaml | head -5`
Expected: `metricConfig` appears as a property with `type: object`.

- [ ] **Step 4: Verify Go still builds and tests pass**

Run: `go build ./... && go test ./internal/controller/ -v`
Expected: build succeeds, existing Go tests PASS.

- [ ] **Step 5: Commit**

```bash
git add api/v1alpha1/roleautoscaler_types.go \
        api/v1alpha1/zz_generated.deepcopy.go \
        config/crd/inference-extension.rolebasedgroup.io_autoscalers.yaml
git commit -m "feat(crd): add MetricConfig, drop metricSource enum for custom engines"
```

---

### Task 6: Go controller delivery

**Files:**
- Modify: `internal/controller/roleautoscaler_controller.go:19-26,241-242,308-311,700-706`
- Test: `internal/controller/roleautoscaler_controller_test.go` (append tests)

**Interfaces:**
- Consumes: the `metricConfig` map parsed from the CRD `metricsEndpoint`.
- Produces: `parsedSpec.metricMappingJSON` (string) and the `METRIC_MAPPING_JSON` env var on the planner pod.

- [ ] **Step 1: Write the failing tests**

Append to `internal/controller/roleautoscaler_controller_test.go`:

```go
func TestBuildPlannerEnv_MetricMappingJSON(t *testing.T) {
	spec := &parsedSpec{
		rbgName:            "test-rbg",
		prefillRoleName:    "prefill",
		decodeRoleName:     "decode",
		backend:            "sglang-oai",
		modelName:          "Qwen",
		scalingInterval:    180,
		ttft:               200,
		itl:                20,
		loadPredictor:      "arima",
		predictionWindow:   50,
		metricsPort:        9091,
		prometheusEndpoint: "http://prom:9090",
		metricMappingJSON:  `{"metrics":{"ttft":"vllm:ttft_v2"}}`,
	}

	env := buildPlannerEnv("test-rbg", spec, "ns", 8, 1, 1)

	var got string
	for _, e := range env {
		if e.Name == "METRIC_MAPPING_JSON" {
			got = e.Value
		}
	}
	if got != `{"metrics":{"ttft":"vllm:ttft_v2"}}` {
		t.Errorf("expected METRIC_MAPPING_JSON set, got %q", got)
	}
}

func TestBuildPlannerEnv_NoMetricMappingJSONWhenEmpty(t *testing.T) {
	spec := &parsedSpec{
		rbgName:            "test-rbg",
		prefillRoleName:    "prefill",
		decodeRoleName:     "decode",
		backend:            "sglang-oai",
		modelName:          "Qwen",
		scalingInterval:    180,
		ttft:               200,
		itl:                20,
		loadPredictor:      "arima",
		predictionWindow:   50,
		metricsPort:        9091,
		prometheusEndpoint: "http://prom:9090",
	}

	env := buildPlannerEnv("test-rbg", spec, "ns", 8, 1, 1)

	for _, e := range env {
		if e.Name == "METRIC_MAPPING_JSON" {
			t.Errorf("did not expect METRIC_MAPPING_JSON when empty, got %q", e.Value)
		}
	}
}

func TestParseRASSpec_MetricConfig(t *testing.T) {
	ras := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"spec": map[string]interface{}{
				"scalingInterval": 180,
				"backend":         "sglang-oai",
				"pattern": map[string]interface{}{
					"PDDisaggregated": map[string]interface{}{
						"prefill": map[string]interface{}{
							"roleName":    "prefill",
							"maxReplicas": int64(10),
							"minReplicas": int64(1),
						},
						"decode": map[string]interface{}{
							"roleName":    "decode",
							"maxReplicas": int64(10),
							"minReplicas": int64(1),
						},
					},
				},
				"implementation": map[string]interface{}{
					"DynamoPlanner": map[string]interface{}{
						"modelName": "Qwen",
						"ttft":       200.0,
						"itl":        20.0,
						"metricsEndpoint": map[string]interface{}{
							"metricSource": "vllm",
							"port":         int64(9091),
							"metricConfig": map[string]interface{}{
								"metrics": map[string]interface{}{
									"ttft": "vllm:time_to_first_token_seconds_v2",
								},
							},
						},
					},
				},
			},
		},
	}

	r := &AutoScalerReconciler{}
	spec, err := r.parseRASSpec(ras)
	if err != nil {
		t.Fatalf("parseRASSpec failed: %v", err)
	}
	if spec.metricSource != "vllm" {
		t.Errorf("expected metricSource vllm, got %q", spec.metricSource)
	}
	if spec.metricMappingJSON == "" {
		t.Fatal("expected non-empty metricMappingJSON")
	}
	if !strings.Contains(spec.metricMappingJSON, "vllm:time_to_first_token_seconds_v2") {
		t.Errorf("expected metricMappingJSON to contain the overridden ttft, got %q", spec.metricMappingJSON)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `go test ./internal/controller/ -v -run 'TestBuildPlannerEnv_MetricMappingJSON|TestBuildPlannerEnv_NoMetricMappingJSONWhenEmpty|TestParseRASSpec_MetricConfig'`
Expected: FAIL — `parsedSpec` has no `metricMappingJSON` field and `buildPlannerEnv` does not emit `METRIC_MAPPING_JSON`.

- [ ] **Step 3: Add the `metricMappingJSON` field to `parsedSpec`**

In `internal/controller/roleautoscaler_controller.go`, add a field to the `parsedSpec` struct (after `metricsPort` on line 242):

```go
	metricMappingJSON   string
```

- [ ] **Step 4: Parse `metricConfig` in `parseRASSpec`**

Add `encoding/json` to the import block (lines 19–25). Then replace the metricsEndpoint parsing (lines 308–311) with:

```go
	// metricsEndpoint
	me, _ := dp["metricsEndpoint"].(map[string]interface{})
	s.metricSource = getStrField(me, "metricSource", "")
	s.metricsPort = getIntField(me, "port", 9091)
	if mc, ok := me["metricConfig"].(map[string]interface{}); ok && mc != nil {
		if b, err := json.Marshal(mc); err == nil {
			s.metricMappingJSON = string(b)
		}
	}
```

- [ ] **Step 5: Emit the `METRIC_MAPPING_JSON` env var in `buildPlannerEnv`**

In `buildPlannerEnv`, after the `METRIC_SOURCE` block (line 705), add:

```go
	if spec.metricMappingJSON != "" {
		env = append(env, corev1.EnvVar{Name: "METRIC_MAPPING_JSON", Value: spec.metricMappingJSON})
	}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `go test ./internal/controller/ -v -run 'TestBuildPlannerEnv_MetricMappingJSON|TestBuildPlannerEnv_NoMetricMappingJSONWhenEmpty|TestParseRASSpec_MetricConfig'`
Expected: PASS.

- [ ] **Step 7: Run the full Go test suite**

Run: `go test ./... -v`
Expected: PASS — all existing tests plus the three new ones.

- [ ] **Step 8: Commit**

```bash
git add internal/controller/roleautoscaler_controller.go \
        internal/controller/roleautoscaler_controller_test.go
git commit -m "feat(controller): deliver metricConfig to planner as METRIC_MAPPING_JSON env var"
```

---

### Task 7: Sample CRD with metricConfig example

**Files:**
- Modify: `config/samples/roleautoscaler_full.yaml:54-57`

- [ ] **Step 1: Add a `metricConfig` example**

In `config/samples/roleautoscaler_full.yaml`, replace the metrics configuration block (lines 54–57) with an example showing a name override and a shape override:

```yaml
      # Metrics configuration
      metricsEndpoint:
        metricSource: sglang
        port: 9091
        # metricConfig optionally overrides metric names, query shapes, and
        # raw PromQL templates for the selected metricSource. Omitted fields
        # fall back to built-in defaults.
        metricConfig:
          metrics:
            ttft: "sglang_time_to_first_token_seconds"   # override a single name
          metricShapes:
            isl: counter_ratio                            # explicit shape (default)
```

- [ ] **Step 2: Verify the sample is valid YAML**

Run: `python3 -c "import yaml; yaml.safe_load(open('config/samples/roleautoscaler_full.yaml'))" && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add config/samples/roleautoscaler_full.yaml
git commit -m "docs(samples): add metricConfig override example to roleautoscaler sample"
```

---

## Self-Review

**Spec coverage:**
- Python adapter evolution (shapes, raw, dispatch, built-in defaults) → Tasks 1–2.
- Factory merge + custom engine validation → Task 3.
- `metric_mapping_json` config + planner/main wiring → Task 4.
- CRD `MetricConfig` type + drop enum + regenerate → Task 5.
- Controller delivery via `METRIC_MAPPING_JSON` → Task 6.
- Sample CRD → Task 7.
- All spec testing bullets (per-shape, raw substitution, merge, custom validation, backward compat, dynamo equivalence) covered by Tasks 2–3 tests.

**Placeholder scan:** No TBD/TODO; every code step contains complete code.

**Type consistency:** `create_metrics_adapter(source, url, metric_config=None)` signature is consistent across Task 2 (no arg), Task 3 (adds arg), Task 4 (planner call). `PrometheusAdapter(url, names, shapes, raw_templates)` consistent across Tasks 2–3. `parsedSpec.metricMappingJSON` consistent across Task 6 steps. CRD field names (`metricConfig`, `modelLabel`, `metrics`, `metricShapes`, `rawTemplates`) match the Python `_METRIC_FIELD_MAP`/`_SHAPE_KEY_MAP` keys and the JSON the controller marshals.
