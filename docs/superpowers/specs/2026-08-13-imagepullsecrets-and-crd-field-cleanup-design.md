# imagePullSecrets Support and CRD Field Cleanup for rbg-planner

**Date:** 2026-08-13
**Status:** Design (pending implementation plan)

## Motivation

After configuring planner and profiler images, the reconciled planner Deployment
and profiling Job cannot specify `imagePullSecrets` — blocking private-registry
pulls. If the profiler image lives behind authentication (e.g., a private GHCR
or internal registry), the profiling Job fails with `ImagePullBackOff`, the
AutoScaler enters a terminal `Failed` phase, and the only recovery path today is
delete-and-recreate.

Additionally, a review of the CRD field design while still at `v1alpha1`
surfaced naming and structure issues that are cheap to fix now but expensive
later:

- `modelCache` implies caching behavior; the field is actually a read-only PVC
  reference for pre-downloaded model files.
- Three discriminated-union variant fields use PascalCase in YAML, violating
  Kubernetes API conventions (camelCase).
- `noCorrection` uses a double-negative boolean anti-pattern.
- `ttft` and `itl` are flat fields that would benefit from SLA grouping.
- `metricsEndpoint` is slightly misleading (it configures collection *and*
  exposition, not just an endpoint).

## Goals

- Users specify `imagePullSecrets` on both the profiling Job and planner
  Deployment via the AutoScaler CR.
- Profiling failure is recoverable by editing the spec (generation-guarded
  retry) — no delete/recreate required.
- CRD field names follow K8s conventions: camelCase JSON tags, accurate
  semantics, consistent grouping.

## Non-goals

- Re-profiling after successful profiling (spec-hash-gated CM regeneration +
  env-var rollout). Deferred — see Future Work.
- Per-workload pod config (`nodeSelector`, `resources`, `tolerations`).
  Separate future work; the field layout designed here does not block it.
- Operator-level `--image-pull-secrets` flag or Helm value. CR-only.
- Validating admission webhook. Runtime validation is sufficient at this stage.
- Restructuring `ModelSourceSpec` sub-fields into a discriminated union
  (`pvc:` / `image:`). Flat `pvcName` is fine for now; the parent name
  `modelSource` is source-agnostic and won't block future restructuring.

## Background

### Current architecture

The AutoScaler reconciler manages two workloads per RBG (1:1 by name equality):

1. **Profiling Job** — one-shot `batchv1.Job` that benchmarks the model and
   writes results to a ConfigMap. Created by `createProfilingJob`
   (line 536). PodSpec sets `ServiceAccountName`, `Volumes` (modelCache PVC),
   and a single `profiler` container with `Image` + `Args`. No
   `imagePullSecrets`.

2. **Planner Deployment** — long-running `appsv1.Deployment` that reads the
   profiling ConfigMap (mounted as a volume) and makes periodic scaling
   decisions. Created/updated by `ensurePlannerDeployment` (line 596). Drift
   detection uses `equality.Semantic.DeepEqual` on `Spec.Template.Spec`
   (PodSpec, line 675). No `imagePullSecrets`.

The reconcile flow is strictly sequential: `handleProfiling` (line 191) must
return `ready=true` (ConfigMap exists / Job succeeded) before
`ensurePlannerDeployment` (line 208) runs. The planner Deployment is never
created without a valid profiling ConfigMap.

### The Failed short-circuit

Line 187: `if phase == "Failed" { return ctrl.Result{}, nil }` — once profiling
fails, the reconciler stops. The user must delete and recreate the AutoScaler to
retry. This is the recovery gotcha: if `imagePullSecrets` were missing on first
submit, the Job fails, phase becomes `Failed`, and adding `imagePullSecrets` to
the spec has no effect.

### Why not adopt Dynamo's model

NVIDIA Dynamo's DGDR embeds the planner as a service *inside* the generated
DynamoGraphDeployment (DGD). Re-profiling regenerates the DGD, which naturally
redeploys the planner with new data — no separate "rollout the planner to read a
new ConfigMap" problem.

rbg-planner **cannot** adopt this model because it scales an *existing*
RoleBasedGroup with a separate planner Deployment, rather than generating a
deployment graph. The ConfigMap-bridge architecture (profiling Job writes CM →
planner Deployment reads CM) is inherent to this design. The lifecycle problems
it creates (Failed terminality, CM-pickup/rollout) are the cost of this
deliberate architectural difference.

Dynamo's own DGDR controller also makes `Failed` terminal (retry logic is
commented as "if desired" but unimplemented) and *rejects* mid-lifecycle spec
changes via a generation guard. rbg-planner's generation-guarded retry (below)
is intentionally more update-friendly than Dynamo's model.

## Design

### 1. imagePullSecrets field

**Placement:** `spec.implementation.dynamoPlanner.imagePullSecrets`

This is a shared credential — both the profiling Job and planner Deployment
pull from the same registry in the same namespace. It is not profiling-specific
(not under `profiling`) or planner-specific (not under a future `planner`
section). Placing it at the `dynamoPlanner` level matches the existing "simple
shared fields are flat" pattern (`modelName`, `dryRun`, etc.).

```go
// DynamoPlannerConfig
// +optional
// +listType=map
// +listMapKey=name
ImagePullSecrets []corev1.LocalObjectReference `json:"imagePullSecrets,omitempty"`
```

**Mechanism:** The controller sets `PodSpec.ImagePullSecrets` directly in both
`createProfilingJob` and `ensurePlannerDeployment`.

- **Planner Deployment:** Adding `ImagePullSecrets` to the desired PodSpec
  changes the DeepEqual comparison (line 675) → triggers `Update` → Kubernetes
  rolls out new pods. No new drift-detection logic needed.
- **Profiling Job:** Jobs are immutable, but the profiling Job is one-shot
  (`BackoffLimit: 1`). On failure, `handleProfiling` deletes the Job (line 487)
  and the controller re-creates it on the next reconcile with the updated spec.
  No immutability problem in practice.

**No operator flag, no default.** `imagePullSecrets` is CR-only. The operator's
own `--planner-image` / `--profiler-image` flags remain flag-only (asymmetric
layering is a separate concern). No Helm changes.

### 2. Generation-guarded Failed recovery

**New status field:** `status.lastProfilingFailureGeneration` (`int64`)

When profiling fails, the controller records `spec.generation` at the time of
failure. The Failed short-circuit (line 187) is relaxed: if `spec.generation`
has advanced since the failure, the reconciler falls through to normal
reconcile instead of returning early.

```
// Pseudo-logic at line 187:
if phase == "Failed" {
    lastFailGen := status.lastProfilingFailureGeneration
    if generation <= lastFailGen {
        return nil  // no spec change since failure — stay Failed
    }
    // generation advanced — fall through to retry
}
```

This is self-limiting: each retry records a new `lastProfilingFailureGeneration`.
If the fix doesn't work, the AutoScaler returns to `Failed` and stays there
until the user edits the spec again. No infinite retry loop.

**handleProfiling adjustment:** When the old failed Job is still being deleted
(`deletionTimestamp` set, `Status.Failed > 0`), return `(not-ready, nil)`
instead of `(error)` — this lets the controller set phase=`Initializing` and
requeue until the Job is gone, rather than re-entering `Failed`. The first
failure (no deletionTimestamp) still returns error → `Failed` as before.

**Recovery flow (forgot imagePullSecrets → fix):**
1. First submit without `imagePullSecrets` → Job fails → phase=`Failed`,
   `lastProfilingFailureGeneration` = gen 1.
2. User adds `imagePullSecrets` → spec generation advances to 2.
3. Reconcile: phase=`Failed` but gen 2 > gen 1 → fall through.
4. `handleProfiling`: CM not found → Job being deleted (or gone) → not-ready.
5. Job gone → `handleProfiling` creates new Job with `imagePullSecrets`.
6. Job succeeds → CM created → `ready=true` → planner Deployment created.

### 3. CRD field cleanup

All changes are `v1alpha1` breaking renames — cheap now, expensive after
graduation.

#### 3a. `modelCache` → `modelSource`

The field references a read-only PVC containing pre-downloaded model weights and
tokenizer files. "Cache" implies temporary storage with fill/eviction semantics;
this is a static model file source. `modelSource` parallels the existing
`metricSource` field ("where X comes from") and is source-agnostic for future
extension.

```yaml
# Before                          # After
profiling:                        profiling:
  modelCache:                       modelSource:
    pvcName: "model-weights"          pvcName: "model-weights"
    modelPath: "deepseek-r1"          modelPath: "deepseek-r1"
    mountPath: "/opt/model-cache"     mountPath: "/opt/model"
```

- `ModelCacheSpec` → `ModelSourceSpec`
- Default `mountPath`: `/opt/model-cache` → `/opt/model`
- Sub-fields (`pvcName`, `modelPath`, `mountPath`) unchanged.

#### 3b. PascalCase → camelCase (JSON tags only)

K8s API conventions require camelCase JSON field names. Three discriminated
union variant selectors are currently PascalCase:

| Current JSON tag | New JSON tag | Go field name (unchanged) |
|---|---|---|
| `PDDisaggregated` | `pdDisaggregated` | `PDDisaggregated` |
| `Unified` | `unified` | `Unified` |
| `DynamoPlanner` | `dynamoPlanner` | `DynamoPlanner` |

Go struct/type names stay PascalCase. `zz_generated.deepcopy.go` is unaffected.
Controller unstructured-access keys (`po["PDDisaggregated"]`,
`se["DynamoPlanner"]`) and error messages update to match.

#### 3c. `ttft` / `itl` → `sla` group

SLA targets are grouped under a new `sla` sub-struct, matching Dynamo's
`SLASpec` pattern and leaving room for future targets (e2eLatency,
optimizationType).

```yaml
# Before                    # After
dynamoPlanner:              dynamoPlanner:
  ttft: 200.0                 sla:
  itl: 20.0                     ttft: 200.0
                                itl: 20.0
```

```go
type SLAConfig struct {
    // +optional
    // +kubebuilder:default=500
    TTFT float64 `json:"ttft,omitempty"`
    // +optional
    // +kubebuilder:default=50
    ITL float64 `json:"itl,omitempty"`
}
```

New `SLA *SLAConfig` field on `DynamoPlannerConfig`. Defaults unchanged (ttft=500,
itl=50 ms).

#### 3d. `metricsEndpoint` → `metrics`

The field configures both metric *collection* (`metricSource`, `metricConfig`)
and metric *exposition* (`port`). "Endpoint" is misleading. Renamed to `metrics`
for brevity and accuracy.

- Field: `metricsEndpoint` → `metrics` on `DynamoPlannerConfig`
- Type: `MetricsEndpointConfig` → `MetricsConfig`

#### 3e. `noCorrection` — unchanged

Left as-is for now. The double-negative (`noCorrection: true`) is a minor API
smell but not a future-blocking issue.

### Final DynamoPlannerConfig shape (after all changes)

```yaml
spec:
  implementation:
    dynamoPlanner:                    # was: DynamoPlanner
      modelName: "Qwen/Qwen3-0.6B"
      sla:                            # was: flat ttft/itl
        ttft: 200.0
        itl: 20.0
      loadPredictor: arima
      predictionWindow: 50
      noCorrection: false
      dryRun: false
      imagePullSecrets:               # NEW
        - name: regcred
      profiling:
        image: "ghcr.io/rolebasedgroup/rbg-profiler:latest"
        modelSource:                  # was: modelCache
          pvcName: "model-weights"
          modelPath: "deepseek-r1"
          mountPath: "/opt/model"     # was: /opt/model-cache
      metrics:                        # was: metricsEndpoint
        metricSource: sglang
        port: 9091
        metricConfig:
          metrics:
            ttft: "sglang_time_to_first_token_seconds"
          metricShapes:
            isl: counter_ratio
```

### Final AutoScalerStatus shape (after all changes)

```yaml
status:
  phase: Ready
  conditions: []
  prefillReplicas: 2
  decodeReplicas: 3
  profilingConfigMap: sglang-pd-inference-profiling
  plannerDeployment: sglang-pd-inference-planner
  lastTransitionTime: "2026-08-13T10:00:00Z"
  lastProfilingFailureGeneration: 0   # NEW
```

## Future work

### Re-profiling system (deferred)

When a user changes profiling-relevant fields (backend, modelName, roles,
profiling.image, modelSource) on an already-Ready AutoScaler, the profiling
ConfigMap content should be regenerated and the planner should pick it up. This
is orthogonal to `imagePullSecrets` (a PodSpec field that propagates via
existing mechanisms) and is deferred:

1. **profilingSpecHash** in status — hash of profiling-relevant fields.
2. **Hash-gated "done" check** in `handleProfiling` — gate on hash-match, not
   CM-existence (line 457).
3. **CM-overwrite** — profiler creates-or-updates the CM (not delete-recreate).
4. **Env-var rollout** — `PROFILING_CM_RESOURCE_VERSION` env var on the planner
   container, tied to CM `resourceVersion`. DeepEqual detects the env var change
   → Deployment rollout. Necessary because the planner loads profiling data once
   at startup (`planner.py:190-191`), with no watch/reload.

### Per-workload pod config

Future `nodeSelector` / `resources` / `tolerations` for the profiling Job (GPU
nodes) and planner Deployment (CPU nodes) would go under `profiling.*` and a new
`planner.*` section respectively. The shared `imagePullSecrets` at
`dynamoPlanner` level stays unchanged — zero blast radius.

### Pre-existing: MetricShapes enum marker

`+kubebuilder:validation:Enum` on `MetricShapes map[string]string` may not
apply correctly to map values in CRD validation. Separate follow-up.

## Testing

1. **parseRASSpec** — verify parsing of `imagePullSecrets`, `modelSource`,
   `sla`, `metrics` (new field names); verify defaults (mountPath=`/opt/model`,
   sla.ttft=500, sla.itl=50).
2. **createProfilingJob** — assert `PodSpec.ImagePullSecrets` is set from CR.
3. **ensurePlannerDeployment** — assert `PodSpec.ImagePullSecrets` is set;
   DeepEqual detects add/remove/change of pull secrets → triggers Update.
4. **Failed recovery** — phase=Failed + generation unchanged → stay Failed;
   generation advanced → retry (Job recreated with new spec).
5. **Job being deleted** — on retry, old Job with `deletionTimestamp` →
   `handleProfiling` returns not-ready (not error) → phase=Initializing.
6. **camelCase keys** — controller unstructured access uses new keys
   (`pdDisaggregated`, `dynamoPlanner`); old PascalCase keys rejected.
7. **Sample YAML** — `config/samples/roleautoscaler_full.yaml` updated with all
   new field names; validates against regenerated CRD.
8. **CRD manifest** — `make manifests` + `make generate` produce correct schema
   with new field names, defaults, and listType markers.
