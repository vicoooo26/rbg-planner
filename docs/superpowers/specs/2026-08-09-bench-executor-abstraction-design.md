# Pluggable Bench Executor Abstraction

> Enable the profiler to run benchmarks through different execution layers (HTTP bench_serving, inference-perf, aiperf) while producing a unified output format for the rbg-planner's decision-making.

Date: 2026-08-09
Branch: `feat/bench-executor-abstraction`

## 1. Problem

The current `BenchBackend` ABC couples HTTP request sending (`run_benchmark()`) with sweep orchestration (`bench_prefill()`/`bench_decode()`/`bench_unified()` are default implementations that call `run_benchmark()`). External CLI tools like inference-perf and aiperf don't fit this model — they are subprocess-invoked tools with their own load generation, not HTTP request senders.

There is no abstraction for "run an external bench tool and adapt its report to rbg-planner's format."

## 2. Goals

- Profiler can run benchmarks through different execution layers (HTTP bench, inference-perf, aiperf).
- Sweep logic (iterate ISL/concurrency points) and adaptation logic (BenchResult → PrefillProfileData/DecodeProfileData) are shared — only the execution layer is pluggable.
- Output format stays unchanged: `prefill_raw_data.json` / `decode_raw_data.json` with parallel arrays. No planner-side changes.
- Prioritize inference-perf integration. aiperf is a future extension point.

## 3. Design

### 3.1 Layer Overview

```
BenchParams (common input)          BenchResult (common output)
    │                                      ▲
    │    ┌─────────────────────────────────┘
    │    │
    ▼    │
BenchExecutor.execute(params) → BenchResult
    │
    ├── CLIBenchExecutor (ABC) — shared subprocess template
    │     │  execute(): build_args → Popen → communicate → find_report → parse_report
    │     │
    │     ├── InferencePerfExecutor
    │     │     build_args(params) → list[str]   # --server.type, --data.type, ...
    │     │     parse_report(path) → BenchResult  # summary_lifecycle_metrics.json
    │     │
    │     └── AIPerfExecutor (future)
    │           build_args(params) → list[str]   # --model, --concurrency, ...
    │           parse_report(path) → BenchResult  # profile_export_aiperf.json
    │
    └── HTTPBenchExecutor
          execute(params) → BenchResult           # direct call to run_benchmark()

ProfilingPipeline
    │  owns sweep logic + adaptation
    │  bench_prefill(): iterate ISL → executor.execute() → adapt → PrefillProfileData
    │  bench_decode():  iterate concurrency → executor.execute() → adapt → DecodeProfileData
    │  bench_unified(): iterate ISL×concurrency → executor.execute() → adapt → UnifiedProfileData
```

### 3.2 Core Data Structures

```python
@dataclass
class BenchParams:
    """Common input for a single benchmark point."""
    endpoint: RoleEndpoint      # base_url, num_gpus, tokenizer
    model: str
    isl: int                    # input sequence length (tokens)
    osl: int                    # output sequence length (tokens)
    num_prompts: int            # number of concurrent requests
    request_rate: float = float("inf")  # requests/sec (inf = all at once)


@dataclass
class BenchResult:
    """Common output from any bench execution."""
    mean_ttft_ms: float
    mean_itl_ms: float
    mean_tpot_ms: float
    request_throughput: float       # req/s
    output_throughput: float        # tokens/s
    mean_e2e_latency_ms: float
```

`BenchResult` has the same fields as the current `BenchMetrics`. The rename signals it's the output of any executor, not just HTTP benchmarking.

### 3.3 BenchExecutor (ABC)

```python
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

### 3.4 CLIBenchExecutor (ABC) — Template Method

Shared subprocess logic for CLI-based bench tools (inference-perf, aiperf). Follows the Dynamo/aiperf pattern: pure functions build `list[str]` args, `subprocess.Popen` runs the tool, report JSON is parsed from the output directory.

```python
class CLIBenchExecutor(BenchExecutor):
    """Template for CLI-based bench tools.

    Subclasses implement build_args() and parse_report().
    The execute() method handles subprocess lifecycle, error handling,
    and report discovery.
    """

    @abstractmethod
    def build_args(self, params: BenchParams, report_dir: str) -> list[str]:
        """Build CLI args list from params. Pure function."""
        raise NotImplementedError

    @abstractmethod
    def parse_report(self, report_dir: str) -> BenchResult:
        """Parse the tool's report file into BenchResult."""
        raise NotImplementedError

    @abstractmethod
    def find_report_path(self, report_dir: str) -> str:
        """Locate the report JSON file within the output directory."""
        raise NotImplementedError

    async def execute(self, params: BenchParams) -> BenchResult:
        report_dir = self._make_report_dir(params)
        args = self.build_args(params, report_dir)
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

    @property
    @abstractmethod
    def tool_name(self) -> str: ...
```

Key design decisions (from Dynamo/aiperf pattern):
- `list[str]` argv form — no shell, no injection risk.
- `asyncio.create_subprocess_exec` — non-blocking, integrates with profiler's async loop.
- Per-point report directory — no collision between sweep points.
- Non-zero exit → raise `BenchExecutionError` (caller catches and skips point).
- No Python-side timeout on communicate; the CLI tool handles per-request timeouts internally (inference-perf has its own circuit breakers).

### 3.5 HTTPBenchExecutor

Wraps the existing `BaseBenchBackend.run_benchmark()`. No subprocess — direct async function call.

```python
class HTTPBenchExecutor(BenchExecutor):
    """HTTP-based bench using the existing BaseBenchBackend.

    The bench_backend determines the API format (sglang completions,
    sglang chat, vllm, vllm chat).
    """

    def __init__(self, bench_backend: BaseBenchBackend):
        self.bench = bench_backend

    async def execute(self, params: BenchParams) -> BenchResult:
        metrics = await self.bench.run_benchmark(
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

The existing `BenchBackend` / `BaseBenchBackend` / `SGLangBenchBackend` / `VLLMBenchBackend` hierarchy stays **unchanged**. It becomes an internal detail of `HTTPBenchExecutor`.

### 3.6 InferencePerfExecutor

Implements `CLIBenchExecutor`. Generates CLI args for `inference-perf`, parses `summary_lifecycle_metrics.json`.

#### CLI Args Construction

inference-perf accepts dot-notation CLI flags mapped from its Pydantic Config. Example for a prefill point (ISL=1024, concurrency=1):

```python
def build_args(self, params: BenchParams, report_dir: str) -> list[str]:
    return [
        "inference-perf",
        "--server.type", self.server_type,           # "vllm" or "sglang"
        "--server.base_url", params.endpoint.base_url,
        "--server.model_name", params.model,
        "--server.ignore_eos", "true",
        "--api.type", self.api_type,                  # "completion" or "chat"
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
```

#### Report Parsing

inference-perf writes `summary_lifecycle_metrics.json` in the report directory. The report structure (from inference-perf docs + aiperf mapping doc):

```jsonc
{
  "successes": {
    "count": 480,
    "latency": {
      "request_latency": { "mean": 3.31, "median": 2.11, "p90": 5.94 },
      "time_to_first_token": { "mean": 0.80, "median": 0.20, "p90": 2.26 },
      "inter_token_latency": { "mean": 0.02, ... },
      "normalized_time_per_output_token": { ... },
      "time_per_output_token": { ... }
    },
    "throughput": {
      "requests_per_sec": 1.02,
      "output_tokens_per_sec": 676.12,
      "total_tokens_per_sec": 676.12
    }
  },
  "failures": { "count": 3, ... },
  "benchmark_time_seconds": 60.0
}
```

Parsing maps inference-perf fields to `BenchResult`:

```python
def parse_report(self, report_path: str) -> BenchResult:
    with open(report_path) as f:
        report = json.load(f)

    successes = report["successes"]
    latency = successes["latency"]
    throughput = successes["throughput"]

    # Throughput fields may be bare floats or stat dicts depending on version.
    # Handle both: if dict, extract .mean; if float, use directly.
    def _extract(val):
        return val["mean"] if isinstance(val, dict) else val

    return BenchResult(
        mean_ttft_ms=_extract(latency["time_to_first_token"]) * 1000,  # s → ms
        mean_itl_ms=_extract(latency["inter_token_latency"]) * 1000,
        mean_tpot_ms=_extract(latency["time_per_output_token"]) * 1000,
        request_throughput=_extract(throughput["requests_per_sec"]),
        output_throughput=_extract(throughput["output_tokens_per_sec"]),
        mean_e2e_latency_ms=_extract(latency["request_latency"]) * 1000,
    )
```

Key mapping notes (from aiperf format doc):
- inference-perf reports latency in **seconds**; rbg-planner uses **ms** → multiply by 1000.
- inference-perf uses `mean` (not `avg`), `median` (not `p50`).
- Throughput fields may be bare floats or stat dicts — `_extract()` handles both.
- `failures.count > 0` should log a warning but not fail the point (partial failures are normal under load).

#### Server Type and API Type Mapping

The `--backend` value (e.g. `sglang-oai`, `vllm-chat`) maps to inference-perf's `--server.type` and `--api.type`:

| rbg-planner backend | inference-perf server.type | inference-perf api.type |
|---------------------|---------------------------|------------------------|
| `sglang-oai`        | `sglang`                  | `completion`           |
| `sglang-oai-chat`   | `sglang`                  | `chat`                 |
| `vllm`              | `vllm`                    | `completion`           |
| `vllm-chat`         | `vllm`                    | `chat`                 |

### 3.7 ProfilingPipeline

Owns the sweep logic and adaptation. Replaces the default implementations currently in `BenchBackend.bench_prefill()` / `bench_decode()` / `bench_unified()`.

```python
class ProfilingPipeline:
    """Orchestrates bench sweeps using a pluggable executor.

    The sweep logic (iterate ISL/concurrency points) and adaptation
    (BenchResult → PrefillProfileData/DecodeProfileData) are shared
    across all executor types.
    """

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
        results = []
        for isl in isl_points:
            if isl >= max_context_length - 512:
                continue
            try:
                result = await self.executor.execute(BenchParams(
                    endpoint=endpoint, model=model,
                    isl=isl, osl=osl,
                    num_prompts=num_prompts,
                ))
            except ContextLengthExceededError:
                break
            except Exception:
                logger.exception("ISL=%d: benchmark failed, skipping", isl)
                continue
            if not (result.mean_ttft_ms > 0) or endpoint.num_gpus <= 0:
                continue
            thpt = isl / result.mean_ttft_ms * 1000 / endpoint.num_gpus
            results.append(PrefillProfileData(
                isl=isl, ttft_ms=result.mean_ttft_ms, thpt_per_gpu=round(thpt, 2)
            ))
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
        # Same sweep logic as current BenchBackend.bench_decode(),
        # but calls self.executor.execute() instead of self.run_benchmark()
        ...

    async def bench_unified(...) -> list[UnifiedProfileData]:
        # Same pattern
        ...
```

The sweep logic is **moved** from `BenchBackend` (where it was default implementations) to `ProfilingPipeline` (where it's the only implementation). The adaptation formulas are unchanged:

- Prefill: `thpt_per_gpu = isl / mean_ttft_ms * 1000 / num_gpus`
- Decode: `thpt_per_gpu = output_throughput / num_gpus`, `kv_usage = (isl + osl/2) * concurrency / max_kv_tokens`

### 3.8 Factory

```python
def create_bench_executor(
    method: str,
    backend: str,
    **kwargs,
) -> BenchExecutor:
    """Create a bench executor for the given method and backend.

    Args:
        method: Bench execution method — "http" or "inference-perf".
        backend: Engine/API backend — "sglang-oai", "vllm-chat", etc.
            For "http": selects the HTTP bench backend (API format).
            For "inference-perf": maps to server.type + api.type.
    """
    if method == "http":
        bench_backend = create_bench_backend(backend)
        return HTTPBenchExecutor(bench_backend)
    elif method == "inference-perf":
        return InferencePerfExecutor(backend=backend, **kwargs)
    else:
        raise ValueError(f"Unknown bench method: {method}")
```

### 3.9 Profiler Changes

`profiler.py:run_profiling()` changes from:

```python
bench_backend = create_bench_backend(backend)
prefill_results = await bench_backend.bench_prefill(...)
decode_results = await bench_backend.bench_decode(...)
```

to:

```python
executor = create_bench_executor(method=args.bench_method, backend=backend)
pipeline = ProfilingPipeline(executor)
prefill_results = await pipeline.bench_prefill(...)
decode_results = await pipeline.bench_decode(...)
```

The rest of `run_profiling()` (converting results to `prefill_data`/`decode_data` dicts, writing to ConfigMap, printing JSON) stays **unchanged**.

### 3.10 CLI Changes

`main.py` gets a new `--bench-method` flag:

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

The existing `--backend` flag retains its meaning (engine/API format) for both methods.

### 3.11 Controller Changes (Future)

The Go controller can pass `BENCH_METHOD` env var to the profiling Job. For now, the default `http` method is used. Adding `inference-perf` support to the CRD is a follow-up — the profiler container image needs `inference-perf` installed.

## 4. File Structure

### New files

```
python/profiler/rbg_profiler/bench/
├── executor.py              # BenchExecutor ABC, BenchParams, BenchResult, BenchExecutionError
├── cli_executor.py          # CLIBenchExecutor ABC (template method)
├── http_executor.py         # HTTPBenchExecutor (wraps BaseBenchBackend)
├── inference_perf_executor.py  # InferencePerfExecutor (CLI + report parser)
├── pipeline.py              # ProfilingPipeline (sweep + adaptation)
└── factory.py               # create_bench_executor()
```

### Modified files

```
python/profiler/rbg_profiler/bench/__init__.py   # export new classes + factory
python/profiler/rbg_profiler/bench/base.py        # no code changes — BenchBackend's
                                                   # bench_prefill/decode/unified remain
                                                   # as backward-compat defaults but are
                                                   # no longer called by the profiler
                                                   # (sweep logic moves to ProfilingPipeline)
python/profiler/rbg_profiler/profiler.py          # use ProfilingPipeline instead of BenchBackend
python/profiler/rbg_profiler/main.py              # add --bench-method flag
```

### Unchanged files

```
python/profiler/rbg_profiler/bench/sglang.py      # SGLangBenchBackend — unchanged
python/profiler/rbg_profiler/bench/vllm.py        # VLLMBenchBackend — unchanged
python/planner/                                   # entirely unchanged
internal/controller/                              # unchanged (for now)
```

### Backward Compatibility

The existing `BenchBackend` ABC and its default `bench_prefill()`/`bench_decode()`/`bench_unified()` implementations remain in `base.py`. They are still used by `HTTPBenchExecutor` internally (via `BaseBenchBackend.run_benchmark()`). The `create_bench_backend()` factory remains for backward compatibility.

The default `--bench-method http` preserves existing behavior exactly. No existing tests should break.

## 5. Error Handling

| Scenario | Behavior |
|----------|----------|
| Subprocess exits non-zero | Raise `BenchExecutionError` with stderr snippet. Pipeline catches, logs, skips point. |
| Report file not found | Raise `FileNotFoundError`. Pipeline catches, logs, skips point. |
| Report JSON missing expected field | `KeyError` during parsing. Pipeline catches, logs, skips point. |
| All requests failed (inference-perf `failures.count == successes.count`) | Log warning. If TTFT/ITL are 0 or absent, the point is skipped by the pipeline's validation. |
| HTTP 400 (context length exceeded) | `ContextLengthExceededError` raised by `HTTPBenchExecutor`. Pipeline stops the sweep (same as current). |
| CLI tool: ISL too large (all requests fail) | No `ContextLengthExceededError` — subprocess succeeds but report shows `failures.count > 0` and `successes` has 0 or absent metrics. Pipeline skips the point (metrics validation fails). Sweep continues to next point. |
| HTTP 429 (rate limited) | `RuntimeError` raised. Pipeline catches, logs, skips point. |

## 6. Testing Strategy

### Unit Tests

- **`test_executor.py`**: BenchParams/BenchResult dataclass tests.
- **`test_http_executor.py`**: HTTPBenchExecutor wraps run_benchmark correctly, maps BenchMetrics → BenchResult.
- **`test_inference_perf_executor.py`**:
  - `build_args()`: verify CLI args construction for prefill/decode params, all fields present, correct types.
  - `parse_report()`: parse sample `summary_lifecycle_metrics.json` fixtures → verify BenchResult fields, unit conversion (s → ms).
  - `parse_report()` edge cases: bare float throughput vs stat dict, missing optional fields, all-failures report.
  - `find_report_path()`: locate report in nested directory.
- **`test_cli_executor.py`**: CLIBenchExecutor template — subprocess mock, error handling (non-zero exit, timeout), report discovery.
- **`test_pipeline.py`**: ProfilingPipeline sweep logic — ISL sweep, concurrency sweep, context length exceeded stops sweep, invalid metrics skipped, adaptation formulas correct. Uses a stub executor.
- **`test_factory.py`**: `create_bench_executor()` — returns correct type for "http"/"inference-perf", raises on unknown method.

### Integration Tests

- **`test_profiler_integration.py`**: End-to-end with `--bench-method http` (existing behavior preserved). Mock executor for `--bench-method inference-perf` — verify pipeline calls executor per sweep point, results converted to prefill_data/decode_data format.

### Fixtures

- `tests/fixtures/inference_perf_report.json` — sample inference-perf `summary_lifecycle_metrics.json` for parsing tests.

## 7. Future: AIPerf Integration

Adding aiperf follows the same `CLIBenchExecutor` pattern:

```python
class AIPerfExecutor(CLIBenchExecutor):
    def build_args(self, params, report_dir) -> list[str]:
        return [
            "aiperf", "profile",
            "--model", params.model,
            "--tokenizer", params.endpoint.tokenizer,
            "--url", params.endpoint.base_url,
            "--synthetic-input-tokens-mean", str(params.isl),
            "--synthetic-input-tokens-stddev", "0",
            "--output-tokens-mean", str(params.osl),
            "--output-tokens-stddev", "0",
            "--concurrency", str(params.num_prompts),
            "--request-count", str(params.num_prompts),
            "--artifact-dir", report_dir,
            "--request-timeout-seconds", "1800",
        ]

    def parse_report(self, report_path) -> BenchResult:
        # Parse profile_export_aiperf.json
        # time_to_first_token.avg → mean_ttft_ms
        # inter_token_latency.avg → mean_itl_ms
        # output_token_throughput.avg → output_throughput
        ...

    def find_report_path(self, report_dir) -> str:
        # os.walk for profile_export_aiperf.json
        ...
```

Factory: add `"aiperf"` to `create_bench_executor()` choices. No other changes needed.

## 8. Open Questions

None — design is complete for the inference-perf priority.
