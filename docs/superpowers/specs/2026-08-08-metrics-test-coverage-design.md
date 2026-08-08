# Metrics Mapping Test Coverage Design

## Goal

Add comprehensive unit tests and integration tests for the configurable metrics mapping feature, closing all 10 identified coverage gaps from the final code review.

## Background

The configurable metrics mapping feature (implemented in `feat/configurable-metrics-mapping`) makes inference-engine metric names and query shapes user-configurable via the CRD. The feature spans:

- **Python**: `defaults.py` (built-in registry), `prometheus.py` (config-driven adapter with shape dispatch + raw templates), `__init__.py` (factory with merge logic + validation), `config.py` (env var), `planner.py` (JSON parsing), `main.py` (CLI args)
- **Go**: `roleautoscaler_types.go` (MetricConfig CRD type), `roleautoscaler_controller.go` (parseRASSpec + buildPlannerEnv)

Current test coverage: 73 Python tests + 22 Go tests. The final review identified 6 confirmed gaps plus 4 additional minor gaps. Two of the original 8 gaps (shape validation, rawTemplates camelCase mapping) were closed by the fix commit `6205ab7`.

## Identified Gaps (10 total)

### Confirmed open gaps (6)

1. **No test for Python wiring** — `config.py` `metric_mapping_json` env var reading, `planner.py` JSON parsing/error handling/non-dict guard, `main.py` CLI args (`--metric-source`, `--metric-mapping-json`).
2. **No test for empty dict `metric_config` on custom engine** — `create_metrics_adapter("acme", url, {})` raises "Unsupported metric source" because `not {}` is `True`. Subtle behavior, untested.
3. **No test for unknown/extra keys being silently dropped** — `metrics` and `metricShapes` filter via `_METRIC_FIELD_MAP`/`_SHAPE_KEY_MAP`; `rawTemplates` uses `_SHAPE_KEY_MAP.get(k, k)` passthrough. All untested.
4. **No test for invalid JSON in `planner.py`** — `JSONDecodeError` catch and non-dict `isinstance` guard are untested.
5. **No integration test for the full CRD → controller → env var → Python planner → adapter → PromQL path** — Go tests stop at `buildPlannerEnv`; Python tests start at `create_metrics_adapter`. The JSON contract between them is unverified.
6. **No test for the backward compatibility path** — `parseRASSpec` with no `metricConfig` → `metricMappingJSON == ""` is untested on the Go side. On the Python side, empty `metric_mapping_json` → `metric_config=None` → built-in defaults is only partially covered.

### Additional gaps found during exploration (4)

7. **No test for `rawTemplates` unknown key passthrough** — `_SHAPE_KEY_MAP.get(k, k)` fallback (unlike `metrics`/`metricShapes` which drop unknowns) is an intentional asymmetry, undocumented in tests.
8. **No test for `metricShapes` camelCase key mapping** — `requestDuration`/`requestsTotal` shape keys mapping to snake_case is only tested for `rawTemplates`, not `metricShapes`.
9. **No test for `_validate_complete` listing exact missing fields** — test only checks substring match, not which fields are reported.
10. **No test for the `role` parameter in `_build_label_filter`** — only `model_name` filtering is tested; the `role` filter branch is never exercised.

## Test Organization

Approach: augment existing test files where tests logically belong; create new files only for CLI tests and integration tests.

### File plan

| File | Action | New tests |
|------|--------|-----------|
| `python/planner/tests/test_config.py` | Augment | 2 |
| `python/planner/tests/test_metric_config.py` | Augment | 5 |
| `python/planner/tests/test_prometheus_client.py` | Augment | 1 |
| `python/planner/tests/test_planner.py` | Augment | 3 |
| `python/planner/tests/test_main.py` | Create | 3 |
| `python/planner/tests/test_integration.py` | Create | 2 |
| `python/planner/tests/fixtures/metric_config_golden.json` | Create | — |
| `internal/controller/roleautoscaler_controller_test.go` | Augment | 2 |

Total: 16 new Python tests + 2 new Go tests = 18 new tests.

## Golden JSON Fixture

**File:** `python/planner/tests/fixtures/metric_config_golden.json`

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

This fixture covers all 4 sub-fields, uses camelCase keys (to verify mapping), and mixes shape/raw dispatch. It represents the exact JSON the Go controller's `json.Marshal(mc)` produces for a `metricConfig` map with these values.

### Go golden fixture test

`TestParseRASSpec_MetricConfig_GoldenFixture`:
1. Build an unstructured AutoScaler with a `metricConfig` containing the fixture's values.
2. Run `r.parseRASSpec(ras)`.
3. Assert `spec.metricMappingJSON` equals the contents of `metric_config_golden.json`.

This verifies the Go controller produces the golden JSON.

### Python integration test

`test_integration.py::TestGoldenJSONIntegration::test_golden_json_full_stack`:
1. Load `tests/fixtures/metric_config_golden.json`.
2. Set `METRIC_MAPPING_JSON` env var, create `PlannerConfig`.
3. Assert `config.metric_mapping_json` equals the golden JSON string.
4. Parse JSON, call `create_metrics_adapter("custom-engine", url, metric_config)`.
5. Mock Prometheus, call `get_avg_ttft` → assert PromQL uses `custom_ttft_seconds` (name override).
6. Call `get_avg_isl` → assert PromQL uses histogram_avg shape (shape override).
7. Call `get_avg_osl` → assert PromQL uses the raw template (raw override).
8. Call `get_avg_request_duration` → assert PromQL uses counter_increase shape (camelCase key mapping + shape override).

## Unit Test Details

### test_config.py (2 new tests)

- `test_metric_mapping_json_from_env` — Set `METRIC_MAPPING_JSON` to a JSON string via `patch.dict(os.environ)`, create `PlannerConfig`, assert `config.metric_mapping_json` equals the env value.
- `test_metric_mapping_json_default_empty` — No `METRIC_MAPPING_JSON` env, assert `config.metric_mapping_json == ""`.

### test_metric_config.py (5 new tests)

- `test_empty_dict_metric_config_on_custom_engine` — `create_metrics_adapter("acme", url, {})` raises `ValueError` matching "Unsupported metric source" (empty dict is falsy, treated as None).
- `test_unknown_metric_keys_dropped` — Config `{"metrics": {"ttft": "ok", "bogus": "ignored"}}` → `adapter.metrics.ttft == "ok"`, no error, "bogus" has no effect.
- `test_unknown_shape_keys_dropped` — Config `{"metricShapes": {"isl": "histogram_avg", "bogus": "ignored"}}` → `adapter.shapes` has `"isl"` but not `"bogus"`.
- `test_metric_shapes_camelcase_key_mapped` — Config `{"metricShapes": {"requestDuration": "counter_increase"}}` → `adapter.shapes["request_duration"] == "counter_increase"`.
- `test_custom_engine_missing_fields_lists_exact_fields` — Config with only `{"ttft": "x"}` → `ValueError` message lists the 5 missing fields: `itl`, `request_duration`, `requests_total`, `prompt_tokens`, `generation_tokens`.

### test_prometheus_client.py (1 new test)

- `test_label_filter_with_role` — Create adapter, call `get_avg_ttft("180s", model_name="Qwen", role="prefill")` (note: the `get_avg_ttft` method signature only takes `interval` and `model_name`; this test calls `_query_avg_metric` directly or uses a method that passes `role`). Assert the PromQL query contains `role="prefill"` in the label selector.

  **Implementation note:** The `get_avg_ttft` method does not accept a `role` parameter (only `interval` and `model_name`). The `role` parameter exists on `_query_avg_metric`, `_query_increase`, and `_build_label_filter` but is not exposed through the public `get_*` interface. This test should call `_query_avg_metric` directly to verify the role filter works, and document that `role` is an internal parameter not yet exposed through the public API.

### test_planner.py (3 new tests)

- `test_invalid_json_metric_mapping_ignored` — Set `config.metric_mapping_json = "{invalid json"`, construct Planner. Assert no crash, adapter uses built-in defaults (metric_config is None).
- `test_non_dict_json_metric_mapping_ignored` — Set `config.metric_mapping_json = "[]"`, construct Planner. Assert no crash, `metric_config` is None (non-dict guard).
- `test_valid_json_metric_mapping_passed_to_factory` — Set `config.metric_mapping_json = '{"metrics":{"ttft":"custom_ttft"}}'`, construct Planner. Assert `adapter.metrics.ttft == "custom_ttft"`.

  **Fixture note:** These tests need to mock `PrometheusConnect` (same `mock_prom` pattern) and may need to mock the RBG connector to avoid network calls during Planner construction. Follow the existing `test_planner.py` fixture patterns.

### test_main.py (3 new tests, new file)

- `test_metric_source_any_string` — Parse args with `--metric-source "acme-llm"`, assert no error (choices restriction removed).
- `test_metric_mapping_json_arg` — Parse args with `--metric-mapping-json '{"metrics":{"ttft":"x"}}'`, assert `config.metric_mapping_json` is set.
- `test_build_config_overrides` — Parse args with both `--metric-source` and `--metric-mapping-json`, run `build_config`, assert both config fields are overridden.

### Go tests (2 new tests)

- `TestParseRASSpec_NoMetricConfig_BackwardCompat` — Build unstructured AutoScaler with `metricsEndpoint` containing only `metricSource` and `port` (no `metricConfig`). Run `r.parseRASSpec(ras)`. Assert `spec.metricMappingJSON == ""` and `spec.metricSource` flows through correctly.
- `TestParseRASSpec_MetricConfig_AllFields` — Build unstructured AutoScaler with `metricConfig` containing all 4 sub-fields (`modelLabel`, `metrics`, `metricShapes`, `rawTemplates`). Run `r.parseRASSpec(ras)`. Assert `spec.metricMappingJSON` contains all 4 field names.

### test_integration.py (2 new tests, new file)

- `test_golden_json_full_stack` — As described in the Golden JSON Fixture section above.
- `test_backward_compat_no_metric_config` — Empty `metric_mapping_json` → `metric_config=None` → `create_metrics_adapter("sglang", url)` → adapter uses built-in sglang defaults → `get_avg_ttft` produces the correct sglang PromQL query (`sglang_time_to_first_token_seconds`).

## Test Infrastructure

### Python
- Framework: pytest with existing `mock_prom` fixture pattern (patches `rbg_planner.metrics.prometheus.PrometheusConnect`).
- Config tests: `patch.dict(os.environ, {...})` pattern.
- Integration tests: load fixture file relative to test file path (`Path(__file__).parent / "fixtures" / "metric_config_golden.json"`).
- No new dependencies required.

### Go
- Framework: standard library `testing` + `sigs.k8s.io/controller-runtime/pkg/client/fake`.
- Unstructured CRD objects built as `map[string]interface{}` trees.
- Golden fixture loaded via `os.ReadFile` relative to test file.

## Error Handling

- All Python tests handle fixture file loading with clear error messages if the file is missing.
- Go test uses `os.ReadFile` with `t.Fatalf` if the fixture is missing.
- No external network calls — Prometheus is always mocked.

## Backward Compatibility

- The backward compatibility path (no `metricConfig` in CRD, no `METRIC_MAPPING_JSON` env var) is explicitly tested on both Go and Python sides, plus the integration test verifies the full backward-compatible flow produces correct default PromQL queries.
