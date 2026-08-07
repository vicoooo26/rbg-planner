# Configurable Metrics Mapping for rbg-planner

**Date:** 2026-08-07
**Status:** Design (pending implementation plan)

## Motivation

Different inference engines — and different *versions* of the same engine — expose
Prometheus metrics under different names and with different aggregation shapes.
For example, vLLM may rename `vllm:time_to_first_token_seconds` between releases,
and Dynamo exposes ISL/OSL as histograms whereas SGLang/vLLM derive them from
counter ratios.

Today rbg-planner handles this with **one hardcoded Python adapter class per
engine** (`SGLangAdapter`, `VLLMAdapter`, `DynamoAdapter`, `PatioAdapter`). Each
class sets a `MetricNames` dataclass of bare metric-name strings and overrides
methods where the query shape differs. Supporting a new engine version or a
third-party engine therefore requires a code change and a new release.

**Goal:** make metric names *and* query shapes user-configurable so that adapting
to a vLLM version rename or onboarding a custom engine is a configuration change,
not a code change — while keeping the common cases structured and safe.

The design takes inspiration from the llm-d router's `core-metrics-extractor`
(`Spec`/`Mapping`/`defaultEngineConfigs` merge-by-name pattern), adapted to
rbg-planner's need for time-window PromQL aggregations (`increase(...)`, sum/count
ratios) that a bare metric-name selector cannot express.

## Goals

- Users override individual metric names per engine via configuration (CRD field).
- Users override the query *shape* per logical metric (e.g. ISL as histogram vs counter ratio).
- Users declare a fully custom third-party engine via configuration, no code.
- A raw-PromQL escape hatch covers engines that fit no built-in shape.
- The four existing engine adapters reproduce today's exact behavior as built-in defaults.

## Non-goals

- Per-role (prefill vs decode) metric querying — stays model-scoped as today.
- Custom logical metrics beyond the existing six (ttft, itl, request_duration,
  requests_total, isl, osl).
- A validating admission webhook — completeness validation happens at planner startup.

## Background

### Current architecture

- `metrics/base.py` — `MetricsAdapter` ABC with six abstract methods:
  `get_avg_ttft`, `get_avg_itl`, `get_request_count`, `get_avg_request_duration`,
  `get_avg_isl`, `get_avg_osl`.
- `metrics/prometheus.py` — `PrometheusAdapter` base with three query helpers that
  are effectively the query *shapes*:
  - `_query_avg_metric` → histogram average: `increase(name_sum[win])/increase(name_count[win])`
  - `_query_increase` → counter total: `increase(name[win])` (sums across series)
  - `_query_counter_ratio` → ratio: `increase(num[win])/increase(denom[win])`
  - plus default `get_*` method implementations wiring metrics to helpers.
- `MetricNames` dataclass — flat string fields: `ttft`, `itl`, `request_duration`,
  `requests_total`, `prompt_tokens`, `generation_tokens`, `model_label`.
- Four subclasses set `MetricNames`; `DynamoAdapter` overrides `get_avg_isl`/`get_avg_osl`
  to use `_query_avg_metric` (histogram) instead of the default counter ratio.
- `metrics/__init__.py` — `create_metrics_adapter(source, url)` factory keyed by string.
- `config.py` — `PlannerConfig` reads `METRIC_SOURCE` env var.
- `planner.py` — calls `create_metrics_adapter(...)` then the six `get_*` methods with
  `(interval, model_name)`.
- CRD `MetricsEndpointConfig` (`api/v1alpha1/roleautoscaler_types.go`) carries only
  `metricSource` (closed enum `sglang|vllm|patio`) and `port`. The controller translates
  `metricSource` to the `METRIC_SOURCE` env var.

### llm-d router inspiration

llm-d's extractor parses a selector string `metric_name{label=value}` into a `Spec`,
holds one `*Spec` per logical metric in a `Mapping`, and ships
`defaultEngineConfigs` that auto-merge with user overrides by engine name. Critically,
llm-d operates on **pre-fetched** metric families (gauge/counter point values), so a
bare name+labels selector suffices. rbg-planner performs **time-window aggregations**
that a selector cannot express, so the abstraction must carry a *shape* in addition to
a name.

## Design overview

The existing `MetricNames` + `_query_*` helpers already encode the "name + shape"
pattern in code. This design makes that same pattern **data-driven**:

- `MetricNames` stays as the per-engine name set — now loadable from config.
- A new **shape config** selects which `_query_*` helper each logical metric uses.
- A **raw-template** map provides a PromQL escape hatch.
- The four engine subclasses collapse into **built-in default data** (names + isl/osl
  shapes), merged with user overrides by engine name (llm-d pattern).
- `PrometheusAdapter` becomes config-driven: each `get_*` method dispatches on the
  resolved shape (or raw template).
- The CRD `MetricsEndpointConfig` gains an optional `metricConfig` sub-field mirroring
  this structure; the controller delivers it to the planner as a JSON env var.

## Python adapter evolution

### Config model

A loaded engine configuration is the tuple `(MetricNames, shapes, rawTemplates)`:

- `MetricNames` — unchanged dataclass: the six name fields + `model_label`.
- `shapes: dict[str, str]` — logical metric → shape enum
  (`histogram_avg` | `counter_increase` | `counter_ratio`). Overrides built-in defaults.
- `rawTemplates: dict[str, str]` — logical metric → PromQL template. Presence for a
  metric overrides shape dispatch entirely.

### Key naming

Two distinct key sets, mirroring the existing split between `MetricNames` (raw metric
names) and the `get_*` methods (logical metrics):

- `metrics` map keys (name fields, → `MetricNames`):
  `ttft`, `itl`, `requestDuration`, `requestsTotal`, `promptTokens`, `generationTokens`.
- `metricShapes` / `rawTemplates` map keys (logical metrics, → `get_*` methods):
  `ttft`, `itl`, `requestDuration`, `requestsTotal`, `isl`, `osl`.

The first four keys coincide; the token metrics differ — `metrics` carries the raw names
(`promptTokens`/`generationTokens`) while `metricShapes`/`rawTemplates` address the
derived logical metrics (`isl`/`osl`). This preserves today's structure where
`MetricNames` holds `prompt_tokens`/`generation_tokens` but the planner calls
`get_avg_isl`/`get_avg_osl`.

CRD keys are camelCase (Kubernetes convention); the Python `MetricNames` fields are
snake_case. The config loader maps between the two.

### Dispatch table

Each `get_*` method resolves its logical metric and dispatches. The binding of
(logical metric, shape) → helper + name field is deterministic:

| logical metric | default shape | `histogram_avg` | `counter_increase` | `counter_ratio` | raw |
|---|---|---|---|---|---|
| ttft | histogram_avg | `_query_avg_metric(names.ttft)` | — | — | `_query_raw(rawTemplates.ttft)` |
| itl | histogram_avg | `…(names.itl)` | — | — | `…(rawTemplates.itl)` |
| request_duration | histogram_avg | `…(names.request_duration)` | — | — | `…(rawTemplates.request_duration)` |
| requests_total | counter_increase | — | `_query_increase(names.requests_total)` | — | `…(rawTemplates.requests_total)` |
| isl | counter_ratio | `…(names.prompt_tokens)` | — | `…(names.prompt_tokens, names.requests_total)` | `…(rawTemplates.isl)` |
| osl | counter_ratio | `…(names.generation_tokens)` | — | `…(names.generation_tokens, names.requests_total)` | `…(rawTemplates.osl)` |

The `DynamoAdapter` override disappears entirely — the Dynamo built-in default simply
sets `shapes.isl = histogram_avg` and `shapes.osl = histogram_avg`.

### Resolution precedence

For each logical metric, a `get_*` method resolves in this order:

1. **`rawTemplates[metric]`** — if present, substitute placeholders and execute; ignore shape.
2. **`shapes[metric]`** — if present, dispatch to the matching `_query_*` helper.
3. **built-in default shape** for that metric
   (ttft/itl/request_duration → histogram_avg, requests_total → counter_increase,
   isl/osl → counter_ratio).

`raw` is **not** a value in the `shapes` enum — it is purely a precedence rule
triggered by template presence. This avoids the footgun of setting `shapes.x: raw`
without a template.

### Per-shape PromQL construction

All shapes share `_build_label_filter(model_name)` → `{model_label="model"}` or `""`
(unchanged from today).

- **`histogram_avg`** → `increase({name}_sum{lf}[{interval}])/increase({name}_count{lf}[{interval}])`,
  then average across returned series (`_exec_avg`).
- **`counter_increase`** → `increase({name}{lf}[{interval}])`, then **sum** across series
  (preserves current total-count behavior in `_query_increase`).
- **`counter_ratio`** → `increase({num}{lf}[{interval}])/increase({denom}{lf}[{interval}])`,
  then average across series (`_exec_avg`).
- **`raw`** (new `_query_raw`) → take `rawTemplates[metric]`, substitute placeholders,
  execute via `_exec_avg`.

### Raw placeholder substitution

Two placeholders:

- `{interval}` → the window string, e.g. `180s`.
- `{labelFilter}` → the full label selector `{model_name="Qwen"}` (with braces) when a
  model is set, or `""` when none.

The adapter performs literal string replacement, then executes the resulting PromQL.
Because the user writes the full expression, they control final aggregation: wrapping in
`sum()`/`avg()` yields a single series, so `_exec_avg` returns that value. Without a
wrapper, multiple series are averaged.

Example: `rawTemplates.ttft = 'avg(acme_ttft_current{labelFilter})'` with model `Qwen`
and interval `180s` → `avg(acme_ttft_current{model_name="Qwen"})` (the `{interval}`
token is simply absent and unused).

### Built-in engine defaults

The four subclass files collapse into a single default-engines registry — a dict keyed
by engine name, each entry holding `(MetricNames, shapes)`. Values reproduce today's
adapters exactly:

- **sglang**, **vllm**, **patio**: standard names; `isl/osl` shape = `counter_ratio`.
- **dynamo**: dynamo names, `model_label = "model"`, `isl/osl` shape = `histogram_avg`.

`dynamo` becomes a recognized built-in (it already exists in Python but is missing from
the CRD enum today).

### Factory and merge

`create_metrics_adapter(source, url, metric_config=None)`:

1. Look up the built-in default for `source` (names + shapes). For a custom engine with
   no built-in, start from an empty `MetricNames` and default shapes.
2. Merge `metric_config` (from CRD) field-by-field over the default: `model_label`
   overrides; each `metrics` entry overrides the matching name field; each `metricShapes`
   entry overrides the matching shape; `rawTemplates` are added.
3. For a **custom engine** (no built-in default), validate at startup that all six name
   fields are non-empty.
4. Construct `PrometheusAdapter` with the merged `(names, shapes, rawTemplates)`.

With no `metric_config`, behavior is identical to today.

## CRD schema

### Design decisions

1. **Single-engine override, not a multi-engine array.** One `RoleAutoscaler` runs one
   engine; the planner uses one adapter. `metricSource` selects the engine (built-in or
   custom); an optional `metricConfig` provides overrides. A custom engine is
   `metricSource: acme-llm` + a full `metricConfig`.
2. **`metricSource` drops its closed enum → free string.** The `sglang|vllm|patio` enum
   blocks custom engines and omits `dynamo`. A free string accepts built-in and custom
   names; completeness validation moves to planner startup (like llm-d's `defaultEngine`).
3. **Name overrides are a flat map merged over built-in defaults.** Renaming one metric
   is a one-line change; unspecified fields fall back to the built-in default.
4. **Shape selection via a separate `metricShapes` map; `rawTemplates` presence overrides
   shape dispatch.** Keeps `metrics` as simple strings (matching `MetricNames`) and limits
   shape variation to where it actually occurs (isl/osl).

### Go types

```go
// MetricsEndpointConfig configures metrics collection.
type MetricsEndpointConfig struct {
    // MetricSource selects the metric source type. Built-in: sglang, vllm,
    // dynamo, patio. Any other value is treated as a custom engine whose
    // metricConfig must define all metric names.
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
    // counter_increase, counter_ratio. Defaults: latency metrics →
    // histogram_avg, requestsTotal → counter_increase, isl/osl → counter_ratio.
    // +optional
    MetricShapes map[string]string `json:"metricShapes,omitempty"`

    // RawTemplates provides a full PromQL template per logical metric. When a
    // metric has an entry here, it overrides the shape dispatch entirely.
    // Templates may use {interval} and {labelFilter} placeholders.
    // +optional
    RawTemplates map[string]string `json:"rawTemplates,omitempty"`
}
```

The `MetricShapes` values use a validation enum
(`histogram_avg;counter_increase;counter_ratio`) — `raw` is intentionally absent.

### YAML example

```yaml
metricsEndpoint:
  metricSource: vllm
  port: 9091
  metricConfig:
    modelLabel: model_name
    metrics:
      ttft: "vllm:time_to_first_token_seconds_v2"   # vLLM v0.9 rename
    metricShapes:
      isl: histogram_avg                             # ISL from histogram, not ratio
      osl: histogram_avg
    rawTemplates:
      requestsTotal: 'sum(increase(acme_reqs_success{labelFilter}[{interval}])) + sum(increase(acme_reqs_error{labelFilter}[{interval}]))'
```

### Backward compatibility

- Existing CRDs with `metricSource: sglang` still validate (free string accepts it).
- `metricConfig` is optional; omitted = pure built-in default = today's behavior.
- The four adapter classes become built-in data; `create_metrics_adapter(source, url)`
  without `metric_config` reproduces today's behavior exactly.

## Controller delivery

The controller reads `metricsEndpoint.metricConfig` from the CRD and serializes it to a
`METRIC_MAPPING_JSON` env var on the planner pod (JSON string). The config is small
(six metrics per engine), well within env-var limits.

- `config.py` `PlannerConfig` gains `metric_mapping_json: str` (env `METRIC_MAPPING_JSON`).
- `planner.py` passes `metric_config` (parsed JSON, or `None`) to `create_metrics_adapter`.
- The controller only sets `METRIC_MAPPING_JSON` when `metricConfig` is non-empty.

`METRIC_SOURCE` continues to select the built-in engine; `METRIC_MAPPING_JSON` overlays
overrides on top.

## File-level change summary

**Python:**

- `metrics/prometheus.py` — `MetricNames` unchanged; add shape/rawTemplate state to
  `PrometheusAdapter`; make `get_*` methods dispatch on shape/raw; add `_query_raw`.
- `metrics/sglang.py`, `metrics/vllm.py`, `metrics/dynamo.py`, `metrics/patio.py` —
  collapse into a built-in defaults registry (e.g. `metrics/defaults.py`).
- `metrics/__init__.py` — `create_metrics_adapter(source, url, metric_config=None)`
  with merge logic.
- `metrics/base.py` — unchanged.
- `config.py` — add `metric_mapping_json` field.
- `planner.py` — pass parsed `metric_config` to the factory.
- `tests/test_prometheus_client.py` — preserve existing behavior; add shape/raw/merge tests.

**Go:**

- `api/v1alpha1/roleautoscaler_types.go` — extend `MetricsEndpointConfig`, add
  `MetricConfig` type, drop `metricSource` enum.
- `api/v1alpha1/zz_generated.deepcopy.go` — regenerate.
- `config/crd/inference-extension.rolebasedgroup.io_autoscalers.yaml` — regenerate via
  `controller-gen`.
- `internal/controller/roleautoscaler_controller.go` — read `metricConfig`, serialize to
  `METRIC_MAPPING_JSON` env var.
- `config/samples/roleautoscaler_full.yaml` — add an example `metricConfig` override.

## Testing

- **Per-shape query construction:** verify each shape builds the expected PromQL string.
- **Raw substitution:** `{interval}` and `{labelFilter}` (present and empty) cases.
- **Merge logic:** CRD overrides merge over built-in defaults field-by-field; unspecified
  fields fall back.
- **Custom engine validation:** a custom `metricSource` with missing name fields errors at
  startup.
- **Backward compatibility:** no `metricConfig` reproduces today's per-engine behavior
  (existing `test_prometheus_client.py` passes unchanged).
- **Dynamo equivalence:** built-in dynamo defaults (isl/osl = histogram_avg) produce the
  same queries as today's `DynamoAdapter` overrides.
