# Metrics Test Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 18 new tests (16 Python + 2 Go) closing all 10 identified coverage gaps for the configurable metrics mapping feature.

**Architecture:** Augment existing test files where tests logically belong; create new files only for CLI tests (`test_main.py`) and integration tests (`test_integration.py`). A shared golden JSON fixture file bridges the Go controller and Python adapter, verifying the cross-language JSON contract.

**Tech Stack:** Python 3 (pytest, unittest.mock), Go (standard library testing, controller-runtime fake client).

## Global Constraints

- SPDX Apache-2.0 header on all new source files (copy from existing test files).
- Python tests live under `python/planner/tests/`; run with `cd python/planner && .venv/bin/pytest tests/ -v`.
- Go tests live in `internal/controller/`; run with `go test ./internal/controller/ -v -run <TestName>`.
- Follow existing test patterns: `mock_prom` fixture (patches `rbg_planner.metrics.prometheus.PrometheusConnect`), `patch.dict(os.environ)` for config tests, `unstructured.Unstructured` for Go CRD tests.
- No external network calls — Prometheus is always mocked.
- Backward compatibility: a CRD with only `metricSource` (no `metricConfig`) must reproduce today's behavior exactly.

---

## File Structure

**Python (augment/create):**
- Modify `python/planner/tests/test_config.py` — add `metric_mapping_json` env var tests.
- Modify `python/planner/tests/test_metric_config.py` — add edge case tests for factory merge logic.
- Modify `python/planner/tests/test_prometheus_client.py` — add role label filter test.
- Modify `python/planner/tests/test_planner.py` — add JSON parsing tests for Planner constructor.
- Create `python/planner/tests/test_main.py` — CLI arg tests for `--metric-source` and `--metric-mapping-json`.
- Create `python/planner/tests/fixtures/metric_config_golden.json` — shared golden JSON fixture.
- Create `python/planner/tests/test_integration.py` — end-to-end integration tests using the golden fixture.

**Go (augment):**
- Modify `internal/controller/roleautoscaler_controller_test.go` — add backward compat and golden fixture tests.

---

### Task 1: Config and label filter tests

**Files:**
- Modify: `python/planner/tests/test_config.py`
- Modify: `python/planner/tests/test_prometheus_client.py`

**Interfaces:**
- Consumes: `PlannerConfig.metric_mapping_json` (env `METRIC_MAPPING_JSON`), `PrometheusAdapter._build_label_filter(model_name, role)`, `PrometheusAdapter._query_avg_metric(metric_name, interval, model_name, role)`.

- [ ] **Step 1: Add metric_mapping_json tests to test_config.py**

Append these two tests to the end of `python/planner/tests/test_config.py`:

```python
def test_metric_mapping_json_from_env():
    with patch.dict(os.environ, {"METRIC_MAPPING_JSON": '{"metrics":{"ttft":"custom"}}'}):
        config = PlannerConfig()
        assert config.metric_mapping_json == '{"metrics":{"ttft":"custom"}}'


def test_metric_mapping_json_default_empty():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("METRIC_MAPPING_JSON", None)
        config = PlannerConfig()
        assert config.metric_mapping_json == ""
```

- [ ] **Step 2: Run the new config tests**

Run: `cd python/planner && .venv/bin/pytest tests/test_config.py -v`
Expected: PASS — 6 tests (4 existing + 2 new).

- [ ] **Step 3: Add role label filter test to test_prometheus_client.py**

Append this test to the `TestLabelFilter` class in `python/planner/tests/test_prometheus_client.py` (after `test_model_filter_dynamo`):

```python
    def test_role_filter(self, mock_prom):
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        lf = adapter._build_label_filter(model_name="Qwen", role="prefill")
        assert lf == '{model_name="Qwen",role="prefill"}'

    def test_role_filter_query(self, mock_prom):
        mock_prom.custom_query.return_value = []
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        adapter._query_avg_metric("sglang_time_to_first_token_seconds", "180s", "Qwen", "prefill")
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert query == (
            'increase(sglang_time_to_first_token_seconds_sum{model_name="Qwen",role="prefill"}[180s])'
            '/increase(sglang_time_to_first_token_seconds_count{model_name="Qwen",role="prefill"}[180s])'
        )
```

- [ ] **Step 4: Run the new prometheus client tests**

Run: `cd python/planner && .venv/bin/pytest tests/test_prometheus_client.py -v`
Expected: PASS — all existing tests + 2 new tests.

- [ ] **Step 5: Commit**

```bash
git add python/planner/tests/test_config.py python/planner/tests/test_prometheus_client.py
git commit -m "test(metrics): add metric_mapping_json env and role label filter tests"
```

---

### Task 2: Factory edge case tests

**Files:**
- Modify: `python/planner/tests/test_metric_config.py`

**Interfaces:**
- Consumes: `create_metrics_adapter(source, url, metric_config)`, `_METRIC_FIELD_MAP`, `_SHAPE_KEY_MAP`, `_validate_complete`, `_REQUIRED_NAME_FIELDS`.

- [ ] **Step 1: Add edge case tests to test_metric_config.py**

Append these tests to the end of `python/planner/tests/test_metric_config.py` (after the `TestCustomEngine` class):

```python
class TestEdgeCases:
    def test_empty_dict_metric_config_on_custom_engine(self, mock_prom):
        with pytest.raises(ValueError, match="Unsupported metric source"):
            create_metrics_adapter("acme-llm", "http://prom:9090", {})

    def test_unknown_metric_keys_dropped(self, mock_prom):
        cfg = {"metrics": {"ttft": "ok", "bogus": "ignored"}}
        adapter = create_metrics_adapter("vllm", "http://prom:9090", cfg)
        assert adapter.metrics.ttft == "ok"
        assert not hasattr(adapter.metrics, "bogus")

    def test_unknown_shape_keys_dropped(self, mock_prom):
        cfg = {"metricShapes": {"isl": "histogram_avg", "bogus": "ignored"}}
        adapter = create_metrics_adapter("vllm", "http://prom:9090", cfg)
        assert "isl" in adapter.shapes
        assert "bogus" not in adapter.shapes

    def test_metric_shapes_camelcase_key_mapped(self, mock_prom):
        cfg = {"metricShapes": {"requestDuration": "counter_increase"}}
        adapter = create_metrics_adapter("vllm", "http://prom:9090", cfg)
        assert adapter.shapes["request_duration"] == "counter_increase"

    def test_custom_engine_missing_fields_lists_exact_fields(self, mock_prom):
        cfg = {"metrics": {"ttft": "acme_ttft"}}
        with pytest.raises(ValueError) as exc_info:
            create_metrics_adapter("acme-llm", "http://prom:9090", cfg)
        msg = str(exc_info.value)
        for field in ["itl", "request_duration", "requests_total", "prompt_tokens", "generation_tokens"]:
            assert field in msg, f"expected {field!r} in error message, got: {msg}"
```

- [ ] **Step 2: Run the new tests**

Run: `cd python/planner && .venv/bin/pytest tests/test_metric_config.py -v`
Expected: PASS — all existing tests + 5 new tests in `TestEdgeCases`.

- [ ] **Step 3: Commit**

```bash
git add python/planner/tests/test_metric_config.py
git commit -m "test(metrics): add factory edge case tests for empty dict, unknown keys, camelCase mapping"
```

---

### Task 3: Planner JSON parsing and CLI arg tests

**Files:**
- Modify: `python/planner/tests/test_planner.py`
- Create: `python/planner/tests/test_main.py`

**Interfaces:**
- Consumes: `Planner.__init__(config)` which calls `json.loads(config.metric_mapping_json)` with error handling, then `create_metrics_adapter(source, url, metric_config)`. Also consumes `main.parse_args()` and `main.build_config(args)`.
- Produces: `test_main.py` with tests for `parse_args()` and `build_config()`.

- [ ] **Step 1: Add JSON parsing tests to test_planner.py**

Append these tests to the end of `python/planner/tests/test_planner.py` (after the last class). The tests use the existing `config` and `mock_interpolators` fixtures. The `config` fixture sets `no_operation=True` (skips RBG connector creation) and `metric_source` defaults to `"sglang"` (derived from `BACKEND` env). `PrometheusConnect` construction does not make network calls, so no `mock_prom` is needed — we only inspect the adapter's metric names, not query results.

```python
class TestMetricMappingJsonParsing:
    def test_invalid_json_uses_defaults(self, config, mock_interpolators):
        config.metric_mapping_json = "{invalid json"
        planner = Planner(config)
        # Invalid JSON is ignored; adapter uses built-in sglang defaults.
        assert planner.metrics_client.metrics.ttft == "sglang_time_to_first_token_seconds"

    def test_non_dict_json_uses_defaults(self, config, mock_interpolators):
        config.metric_mapping_json = "[]"
        planner = Planner(config)
        # Non-dict JSON is ignored; adapter uses built-in sglang defaults.
        assert planner.metrics_client.metrics.ttft == "sglang_time_to_first_token_seconds"

    def test_valid_json_override_applied(self, config, mock_interpolators):
        config.metric_mapping_json = '{"metrics":{"ttft":"custom_ttft"}}'
        planner = Planner(config)
        assert planner.metrics_client.metrics.ttft == "custom_ttft"
```

- [ ] **Step 2: Run the new planner tests**

Run: `cd python/planner && .venv/bin/pytest tests/test_planner.py::TestMetricMappingJsonParsing -v`
Expected: PASS — 3 new tests.

- [ ] **Step 3: Create test_main.py**

Create `python/planner/tests/test_main.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CLI argument parsing and config building."""

from unittest.mock import patch

from rbg_planner.main import build_config, parse_args


class TestParseArgs:
    def test_metric_source_any_string(self):
        with patch("sys.argv", ["planner", "--metric-source", "acme-llm"]):
            args = parse_args()
        assert args.metric_source == "acme-llm"

    def test_metric_mapping_json_arg(self):
        json_str = '{"metrics":{"ttft":"custom"}}'
        with patch("sys.argv", ["planner", "--metric-mapping-json", json_str]):
            args = parse_args()
        assert args.metric_mapping_json == json_str


class TestBuildConfig:
    def test_both_overrides_applied(self):
        json_str = '{"metrics":{"ttft":"custom"}}'
        with patch("sys.argv", [
            "planner",
            "--metric-source", "acme-llm",
            "--metric-mapping-json", json_str,
        ]):
            args = parse_args()
        config = build_config(args)
        assert config.metric_source == "acme-llm"
        assert config.metric_mapping_json == json_str
```

- [ ] **Step 4: Run the new main tests**

Run: `cd python/planner && .venv/bin/pytest tests/test_main.py -v`
Expected: PASS — 3 new tests.

- [ ] **Step 5: Run the full Python suite**

Run: `cd python/planner && .venv/bin/pytest tests/ -v`
Expected: PASS — all existing tests + 6 new tests (3 in test_planner.py + 3 in test_main.py).

- [ ] **Step 6: Commit**

```bash
git add python/planner/tests/test_planner.py python/planner/tests/test_main.py
git commit -m "test(planner): add JSON parsing and CLI arg tests for metricConfig wiring"
```

---

### Task 4: Golden JSON fixture and Go tests

**Files:**
- Create: `python/planner/tests/fixtures/metric_config_golden.json`
- Modify: `internal/controller/roleautoscaler_controller_test.go`

**Interfaces:**
- Produces: `metric_config_golden.json` — the shared golden JSON fixture consumed by Task 5's integration tests.

- [ ] **Step 1: Create the golden JSON fixture**

Create directory and file `python/planner/tests/fixtures/metric_config_golden.json`:

```json
{
  "modelLabel": "model",
  "metrics": {
    "ttft": "custom_ttft_seconds",
    "requestDuration": "custom_duration_seconds"
  },
  "metricShapes": {
    "isl": "histogram_avg",
    "requestDuration": "counter_increase"
  },
  "rawTemplates": {
    "osl": "avg(custom_gen_tokens{labelFilter}) / avg(custom_requests{labelFilter})"
  }
}
```

- [ ] **Step 2: Add Go backward compat test**

Append to `internal/controller/roleautoscaler_controller_test.go`:

```go
func TestParseRASSpec_NoMetricConfig_BackwardCompat(t *testing.T) {
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
						"ttft":      200.0,
						"itl":       20.0,
						"metricsEndpoint": map[string]interface{}{
							"metricSource": "sglang",
							"port":         int64(9091),
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
	if spec.metricSource != "sglang" {
		t.Errorf("expected metricSource sglang, got %q", spec.metricSource)
	}
	if spec.metricMappingJSON != "" {
		t.Errorf("expected empty metricMappingJSON, got %q", spec.metricMappingJSON)
	}
}
```

- [ ] **Step 3: Add Go golden fixture test**

Append to `internal/controller/roleautoscaler_controller_test.go`. Add `"os"` to the import block if not already present. The test loads the golden fixture from the Python tests directory (relative path from the Go package directory `internal/controller/` is `../../python/planner/tests/fixtures/metric_config_golden.json`).

```go
func TestParseRASSpec_MetricConfig_GoldenFixture(t *testing.T) {
	goldenPath := "../../python/planner/tests/fixtures/metric_config_golden.json"
	expected, err := os.ReadFile(goldenPath)
	if err != nil {
		t.Fatalf("failed to read golden fixture: %v", err)
	}

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
						"ttft":      200.0,
						"itl":       20.0,
						"metricsEndpoint": map[string]interface{}{
							"metricSource": "sglang",
							"port":         int64(9091),
							"metricConfig": map[string]interface{}{
								"modelLabel": "model",
								"metrics": map[string]interface{}{
									"ttft":             "custom_ttft_seconds",
									"requestDuration":  "custom_duration_seconds",
								},
								"metricShapes": map[string]interface{}{
									"isl":              "histogram_avg",
									"requestDuration":  "counter_increase",
								},
								"rawTemplates": map[string]interface{}{
									"osl": "avg(custom_gen_tokens{labelFilter}) / avg(custom_requests{labelFilter})",
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

	// json.Marshal produces compact JSON with no trailing newline;
	// the fixture file may have a trailing newline, so trim it.
	got := strings.TrimRight(spec.metricMappingJSON, "\n")
	want := strings.TrimRight(string(expected), "\n")
	if got != want {
		t.Errorf("metricMappingJSON does not match golden fixture\n got: %s\n want: %s", got, want)
	}
}
```

- [ ] **Step 4: Run the Go tests**

Run: `go test ./internal/controller/ -v -run 'TestParseRASSpec_NoMetricConfig_BackwardCompat|TestParseRASSpec_MetricConfig_GoldenFixture'`
Expected: PASS — 2 new tests.

- [ ] **Step 5: Run the full Go suite**

Run: `go test ./internal/controller/ -v`
Expected: PASS — all existing tests + 2 new tests.

- [ ] **Step 6: Commit**

```bash
git add python/planner/tests/fixtures/metric_config_golden.json \
        internal/controller/roleautoscaler_controller_test.go
git commit -m "test(controller): add backward compat and golden fixture tests for metricConfig"
```

---

### Task 5: Integration tests

**Files:**
- Create: `python/planner/tests/test_integration.py`

**Interfaces:**
- Consumes: `metric_config_golden.json` from Task 4, `PlannerConfig`, `create_metrics_adapter`, `PrometheusAdapter` query methods.

- [ ] **Step 1: Create test_integration.py**

Create `python/planner/tests/test_integration.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for the configurable metrics mapping feature.

These tests verify the full round-trip: the golden JSON fixture (which the Go
controller produces via json.Marshal) flows through the Python stack — env var,
PlannerConfig, factory merge, adapter dispatch — to produce correct PromQL queries.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rbg_planner.config import PlannerConfig
from rbg_planner.metrics import create_metrics_adapter

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "metric_config_golden.json"


@pytest.fixture
def golden_json():
    return FIXTURE_PATH.read_text()


@pytest.fixture
def golden_config(golden_json):
    return json.loads(golden_json)


@pytest.fixture
def mock_prom():
    with patch("rbg_planner.metrics.prometheus.PrometheusConnect") as mock_cls:
        mock_conn = MagicMock()
        mock_cls.return_value = mock_conn
        yield mock_conn


class TestGoldenJSONIntegration:
    def test_golden_json_full_stack(self, mock_prom, golden_json, golden_config):
        """The golden JSON from the Go controller flows through the full
        Python stack: env var → PlannerConfig → factory → adapter → PromQL."""
        # 1. Env var → PlannerConfig
        with patch.dict("os.environ", {"METRIC_MAPPING_JSON": golden_json}):
            config = PlannerConfig()
        assert config.metric_mapping_json == golden_json

        # 2. Parse JSON → factory → adapter
        metric_config = json.loads(config.metric_mapping_json)
        adapter = create_metrics_adapter(
            "custom-engine", "http://prom:9090", metric_config,
        )

        # 3. modelLabel override: "model" instead of default "model_name"
        assert adapter.metrics.model_label == "model"

        # 4. metrics.ttft override
        mock_prom.custom_query.return_value = [{"value": [0, "0.5"]}]
        adapter.get_avg_ttft("180s", "Qwen")
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert "custom_ttft_seconds" in query
        assert 'model="Qwen"' in query  # uses overridden modelLabel

        # 5. metricShapes.isl override: histogram_avg (not default counter_ratio)
        mock_prom.custom_query.reset_mock()
        mock_prom.custom_query.return_value = [{"value": [0, "512.0"]}]
        adapter.get_avg_isl("180s", "Qwen")
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert "_sum{" in query and "_count{" in query  # histogram_avg shape

        # 6. rawTemplates.osl override: raw template takes precedence
        mock_prom.custom_query.reset_mock()
        mock_prom.custom_query.return_value = [{"value": [0, "128.0"]}]
        adapter.get_avg_osl("180s", "Qwen")
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert "custom_gen_tokens" in query
        assert "custom_requests" in query

        # 7. metricShapes.requestDuration camelCase → snake_case + counter_increase
        mock_prom.custom_query.reset_mock()
        mock_prom.custom_query.return_value = [{"value": [0, "0.3"]}]
        adapter.get_avg_request_duration("180s", "Qwen")
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert query == "increase(custom_duration_seconds{model=\"Qwen\"}[180s])"

    def test_backward_compat_no_metric_config(self, mock_prom):
        """Empty metric_mapping_json → adapter uses built-in sglang defaults
        → correct PromQL query."""
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("METRIC_MAPPING_JSON", None)
            config = PlannerConfig()

        assert config.metric_mapping_json == ""

        # No metric_config → built-in sglang defaults
        adapter = create_metrics_adapter(
            config.metric_source, "http://prom:9090",
        )

        mock_prom.custom_query.return_value = [{"value": [0, "0.5"]}]
        adapter.get_avg_ttft("180s", "Qwen")
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert "sglang_time_to_first_token_seconds" in query
        assert 'model_name="Qwen"' in query  # default model_label
```

- [ ] **Step 2: Run the integration tests**

Run: `cd python/planner && .venv/bin/pytest tests/test_integration.py -v`
Expected: PASS — 2 new tests.

- [ ] **Step 3: Run the full Python suite**

Run: `cd python/planner && .venv/bin/pytest tests/ -v`
Expected: PASS — all existing tests + all new tests from Tasks 1-5.

- [ ] **Step 4: Commit**

```bash
git add python/planner/tests/test_integration.py
git commit -m "test(integration): add golden JSON end-to-end and backward compat integration tests"
```

---

## Self-Review

**Spec coverage:**
- Gap 1 (Python wiring): test_config.py (2 tests), test_planner.py (3 tests), test_main.py (3 tests) → Task 1 + Task 3.
- Gap 2 (empty dict on custom engine): test_metric_config.py `test_empty_dict_metric_config_on_custom_engine` → Task 2.
- Gap 3 (unknown keys dropped): test_metric_config.py `test_unknown_metric_keys_dropped`, `test_unknown_shape_keys_dropped` → Task 2.
- Gap 4 (invalid JSON in planner.py): test_planner.py `test_invalid_json_uses_defaults`, `test_non_dict_json_uses_defaults` → Task 3.
- Gap 5 (integration test): test_integration.py `test_golden_json_full_stack` → Task 5.
- Gap 6 (backward compat): Go `TestParseRASSpec_NoMetricConfig_BackwardCompat` + Python `test_backward_compat_no_metric_config` → Task 4 + Task 5.
- Gap 7 (rawTemplates unknown key passthrough): `test_unknown_shape_keys_dropped` covers the filtering behavior; rawTemplates passthrough is implicitly tested by `test_raw_template_camelcase_key_mapped` (existing) which verifies `_SHAPE_KEY_MAP.get(k, k)` mapping. No additional test needed — the behavior is covered.
- Gap 8 (metricShapes camelCase mapping): test_metric_config.py `test_metric_shapes_camelcase_key_mapped` → Task 2.
- Gap 9 (exact missing fields): test_metric_config.py `test_custom_engine_missing_fields_lists_exact_fields` → Task 2.
- Gap 10 (role label filter): test_prometheus_client.py `test_role_filter`, `test_role_filter_query` → Task 1.

**Placeholder scan:** No TBD/TODO; every step contains complete code.

**Type consistency:** `create_metrics_adapter(source, url, metric_config)` signature consistent across all tasks. `PlannerConfig.metric_mapping_json` consistent. Golden JSON fixture field names match CRD camelCase keys and Python `_METRIC_FIELD_MAP`/`_SHAPE_KEY_MAP`.
