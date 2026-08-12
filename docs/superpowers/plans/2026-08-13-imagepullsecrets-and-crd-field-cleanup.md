# imagePullSecrets Support and CRD Field Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `imagePullSecrets` support to profiling Job and planner Deployment, add generation-guarded Failed recovery, and clean up CRD field naming — all while v1alpha1 makes breaking renames cheap.

**Architecture:** Three independent tasks: (1) CRD field renames (mechanical refactor of existing types/controller/tests), (2) `imagePullSecrets` field on `dynamoPlanner` applied to both PodSpecs, (3) `lastProfilingFailureGeneration` status field relaxing the Failed short-circuit. Each task is self-contained and independently testable.

**Tech Stack:** Go, controller-runtime, kubebuilder/controller-gen, Kubernetes batchv1/apps/v1/corev1 APIs.

## Global Constraints

- Go module: `github.com/rolebasedgroup/rbg-planner`
- CRD API version: `v1alpha1` (breaking renames allowed)
- Manifest generation: `make generate` (deepcopy) + `make manifests` (CRD YAML)
- Tests: `go test ./... -v`
- controller-gen path: `$(go env GOPATH)/bin/controller-gen`
- The controller uses **unstructured** access (string keys), not typed CR access — JSON tag renames require updating string keys in `parseRASSpec` and test fixtures.
- `parsedSpec` is an internal struct — its field names (`modelCachePVCName`, `ttft`, etc.) are NOT part of the API and do not need renaming.

---

### Task 1: CRD Field Renames

Renames `modelCache`→`modelSource`, PascalCase→camelCase JSON tags, `ttft`/`itl`→`sla` group, `metricsEndpoint`→`metrics`, and `mountPath` default. This is a mechanical refactor — existing tests are updated to match.

**Files:**
- Modify: `api/v1alpha1/roleautoscaler_types.go` (type definitions)
- Modify: `api/v1alpha1/zz_generated.deepcopy.go` (regenerated)
- Modify: `internal/controller/roleautoscaler_controller.go:262-313` (parseRASSpec string keys + buildProfilingVolumes default)
- Modify: `internal/controller/roleautoscaler_controller_test.go` (test fixtures + buildProfilingVolumes tests)
- Modify: `config/samples/roleautoscaler_full.yaml` (sample YAML)
- Modify: `config/crd/inference-extension.rolebasedgroup.io_autoscalers.yaml` (regenerated)

**Interfaces:**
- Consumes: none (first task)
- Produces: all renamed JSON field names used by Tasks 2 and 3 (`dynamoPlanner`, `pdDisaggregated`, `sla`, `modelSource`, `metrics`)

- [ ] **Step 1: Update types — PascalCase JSON tags**

In `api/v1alpha1/roleautoscaler_types.go`, change three JSON tags (Go field names stay PascalCase):

```go
// Pattern struct (line ~49)
PDDisaggregated *PDDisaggregatedPattern `json:"pdDisaggregated,omitempty"`

// Pattern struct (line ~54)
Unified *UnifiedPattern `json:"unified,omitempty"`

// Implementation struct (line ~103)
DynamoPlanner *DynamoPlannerConfig `json:"dynamoPlanner,omitempty"`
```

- [ ] **Step 2: Update types — add SLAConfig, restructure ttft/itl**

In `api/v1alpha1/roleautoscaler_types.go`, remove the flat `TTFT` and `ITL` fields from `DynamoPlannerConfig` (lines ~112-120) and add a new `SLAConfig` struct + `SLA` field:

```go
// DynamoPlannerConfig — replace the TTFT and ITL fields with:
	// SLA defines the target SLA thresholds.
	// +optional
	SLA *SLAConfig `json:"sla,omitempty"`
```

Add the new struct (place it before `ProfilingConfig`):

```go
// SLAConfig defines SLA target thresholds for the planner.
type SLAConfig struct {
	// TTFT is the target Time to First Token SLA in milliseconds.
	// +optional
	// +kubebuilder:default=500
	TTFT float64 `json:"ttft,omitempty"`

	// ITL is the target Inter-Token Latency SLA in milliseconds.
	// +optional
	// +kubebuilder:default=50
	ITL float64 `json:"itl,omitempty"`
}
```

- [ ] **Step 3: Update types — rename modelCache → modelSource**

In `api/v1alpha1/roleautoscaler_types.go`:

Rename `ProfilingConfig.ModelCache` field (line ~161):
```go
	// ModelSource provides PVC configuration for the profiler to access
	// model weights and tokenizer files. When provided, the operator mounts
	// the PVC and passes the tokenizer path to the profiler via --tokenizer.
	// +optional
	ModelSource *ModelSourceSpec `json:"modelSource,omitempty"`
```

Rename the struct `ModelCacheSpec` → `ModelSourceSpec` (line ~166) and change the `MountPath` default:
```go
// ModelSourceSpec references a PVC containing pre-downloaded model weights
// and tokenizer files for the profiling job.
type ModelSourceSpec struct {
	// PVCName is the name of the PersistentVolumeClaim containing model weights.
	// The PVC must exist in the same namespace as the AutoScaler.
	// +optional
	PVCName string `json:"pvcName,omitempty"`

	// ModelPath is the path to the model/tokenizer directory within the PVC
	// (e.g. "deepseek-r1" or "models/Llama-3.1-405B"). May also be a
	// container-visible absolute path already under MountPath. Such an absolute
	// path is interpreted as container-visible; use the relative form without a
	// leading slash to address the same path prefix within the PVC.
	// +optional
	ModelPath string `json:"modelPath,omitempty"`

	// MountPath is the mount path for the PVC inside the profiler container.
	// +optional
	// +kubebuilder:default="/opt/model"
	MountPath string `json:"mountPath,omitempty"`
}
```

- [ ] **Step 4: Update types — rename metricsEndpoint → metrics**

In `api/v1alpha1/roleautoscaler_types.go`:

Rename `DynamoPlannerConfig.MetricsEndpoint` field (line ~147):
```go
	// Metrics configures metrics collection and exposition.
	// +optional
	Metrics *MetricsConfig `json:"metrics,omitempty"`
```

Rename the struct `MetricsEndpointConfig` → `MetricsConfig` (line ~187):
```go
// MetricsConfig configures metrics collection and exposition.
type MetricsConfig struct {
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
```

- [ ] **Step 5: Regenerate deepcopy**

Run:
```bash
make generate
```

Expected: `zz_generated.deepcopy.go` updated with new type names (`ModelSourceSpec`, `MetricsConfig`, `SLAConfig`).

- [ ] **Step 6: Update controller — parseRASSpec string keys**

In `internal/controller/roleautoscaler_controller.go`, update `parseRASSpec` (lines 262-313):

Line 264 — change key and error message:
```go
	pd, _ := po["pdDisaggregated"].(map[string]interface{})
	if pd == nil {
		return nil, fmt.Errorf("pattern.pdDisaggregated is required")
	}
```

Line 279 — change key and error message:
```go
	dp, _ := se["dynamoPlanner"].(map[string]interface{})
	if dp == nil {
		return nil, fmt.Errorf("implementation.dynamoPlanner is required")
	}
```

Lines 288-289 — replace flat ttft/itl with sla sub-map parsing:
```go
	// sla
	slaMap, _ := dp["sla"].(map[string]interface{})
	s.ttft = getFloatField(slaMap, "ttft", 500.0)
	s.itl = getFloatField(slaMap, "itl", 50.0)
```

Line 304 — change `modelCache` to `modelSource`:
```go
	if mc, ok := prof["modelSource"].(map[string]interface{}); ok && mc != nil {
```

Line 307 — change default mountPath:
```go
		s.modelCacheMountPath = getStrField(mc, "mountPath", "/opt/model")
```

Line 312 — change `metricsEndpoint` to `metrics`:
```go
	me, _ := dp["metrics"].(map[string]interface{})
```

- [ ] **Step 7: Update controller — buildProfilingVolumes default**

In `internal/controller/roleautoscaler_controller.go`, line 507, change the default mountPath:

```go
	if mountPath == "" {
		mountPath = "/opt/model"
	}
```

- [ ] **Step 8: Update test fixtures — new field names**

In `internal/controller/roleautoscaler_controller_test.go`, update the three `TestParseRASSpec_*` test fixtures (lines ~633, ~689, ~743). In each fixture:

- `"PDDisaggregated"` → `"pdDisaggregated"`
- `"DynamoPlanner"` → `"dynamoPlanner"`
- Replace flat `"ttft": 200.0, "itl": 20.0` with:
```go
							"sla": map[string]interface{}{
								"ttft": 200.0,
								"itl":  20.0,
							},
```
- `"metricsEndpoint"` → `"metrics"`

For `TestParseRASSpec_MetricConfig` (line ~626), the updated `implementation` section:
```go
					"implementation": map[string]interface{}{
						"dynamoPlanner": map[string]interface{}{
							"modelName": "Qwen",
							"sla": map[string]interface{}{
								"ttft": 200.0,
								"itl":  20.0,
							},
							"metrics": map[string]interface{}{
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
```

Apply the same pattern to `TestParseRASSpec_NoMetricConfig_BackwardCompat` (line ~682) and `TestParseRASSpec_MetricConfig_GoldenFixture` (line ~730).

- [ ] **Step 9: Update buildProfilingVolumes tests — new default mountPath**

In `internal/controller/roleautoscaler_controller_test.go`, replace all `"/opt/model-cache"` with `"/opt/model"` in the `TestBuildProfilingVolumes_*` tests (lines 39, 54, 61, 67, 75, 89, 105).

- [ ] **Step 10: Update sample YAML**

In `config/samples/roleautoscaler_full.yaml`:

```yaml
  pattern:
    pdDisaggregated:              # was: PDDisaggregated
      prefill:
        roleName: prefill
        maxReplicas: 10
        minReplicas: 1
      decode:
        roleName: decode
        maxReplicas: 10
        minReplicas: 1

  implementation:
    dynamoPlanner:                # was: DynamoPlanner
      modelName: "Qwen/Qwen3-0.6B"

      sla:                        # was: flat ttft/itl
        ttft: 200.0
        itl: 20.0

      loadPredictor: arima
      predictionWindow: 50
      noCorrection: false
      dryRun: false

      profiling:
        image: "ghcr.io/rolebasedgroup/rbg-profiler:latest"
        modelSource:              # was: modelCache
          pvcName: "model-weights"
          modelPath: "deepseek-r1"
          mountPath: "/opt/model" # was: /opt/model-cache

      metrics:                    # was: metricsEndpoint
        metricSource: sglang
        port: 9091
        metricConfig:
          metrics:
            ttft: "sglang_time_to_first_token_seconds"
          metricShapes:
            isl: counter_ratio
```

- [ ] **Step 11: Regenerate CRD manifest**

Run:
```bash
make manifests
```

Expected: `config/crd/inference-extension.rolebasedgroup.io_autoscalers.yaml` updated with new field names.

- [ ] **Step 12: Run tests and verify build**

Run:
```bash
go build ./... && go test ./... -v
```

Expected: all tests PASS.

- [ ] **Step 13: Commit**

```bash
git add api/v1alpha1/ internal/controller/ config/samples/ config/crd/
git commit -m "refactor(api): rename CRD fields for K8s convention compliance

- modelCache → modelSource (accurate semantics, not caching)
- PDDisaggregated/Unified/DynamoPlanner → camelCase JSON tags
- ttft/itl → sla group (extensible for future SLA targets)
- metricsEndpoint → metrics (covers collection + exposition)
- mountPath default /opt/model-cache → /opt/model"
```

---

### Task 2: imagePullSecrets Support

Adds `imagePullSecrets` to `dynamoPlannerConfig`, parsed by `parseRASSpec`, and set on both the profiling Job and planner Deployment PodSpecs.

**Files:**
- Modify: `api/v1alpha1/roleautoscaler_types.go` (add `ImagePullSecrets` field + `corev1` import)
- Modify: `api/v1alpha1/zz_generated.deepcopy.go` (regenerated)
- Modify: `internal/controller/roleautoscaler_controller.go` (parsedSpec, parseRASSpec, createProfilingJob, ensurePlannerDeployment)
- Modify: `internal/controller/roleautoscaler_controller_test.go` (new tests)
- Modify: `config/crd/inference-extension.rolebasedgroup.io_autoscalers.yaml` (regenerated)

**Interfaces:**
- Consumes: `dynamoPlanner` JSON key from Task 1
- Produces: `parsedSpec.imagePullSecrets` field consumed by `createProfilingJob` and `ensurePlannerDeployment`

- [ ] **Step 1: Write failing test — parseRASSpec parses imagePullSecrets**

Add to `internal/controller/roleautoscaler_controller_test.go`:

```go
func TestParseRASSpec_ImagePullSecrets(t *testing.T) {
	ras := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"spec": map[string]interface{}{
				"scalingInterval": 180,
				"backend":         "sglang-oai",
				"pattern": map[string]interface{}{
					"pdDisaggregated": map[string]interface{}{
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
					"dynamoPlanner": map[string]interface{}{
						"modelName": "Qwen",
						"sla": map[string]interface{}{
							"ttft": 200.0,
							"itl":  20.0,
						},
						"imagePullSecrets": []interface{}{
							map[string]interface{}{"name": "regcred"},
							map[string]interface{}{"name": "ghcr-secret"},
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
	if len(spec.imagePullSecrets) != 2 {
		t.Fatalf("expected 2 imagePullSecrets, got %d", len(spec.imagePullSecrets))
	}
	if spec.imagePullSecrets[0].Name != "regcred" {
		t.Errorf("expected first secret 'regcred', got %q", spec.imagePullSecrets[0].Name)
	}
	if spec.imagePullSecrets[1].Name != "ghcr-secret" {
		t.Errorf("expected second secret 'ghcr-secret', got %q", spec.imagePullSecrets[1].Name)
	}
}
```

- [ ] **Step 2: Write failing test — createProfilingJob sets ImagePullSecrets**

Add to `internal/controller/roleautoscaler_controller_test.go`:

```go
func TestCreateProfilingJob_SetsImagePullSecrets(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := batchv1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add batch scheme: %v", err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add core scheme: %v", err)
	}

	ras := &unstructured.Unstructured{}
	ras.SetName("test-rbg")
	ras.SetNamespace("test-ns")
	ras.SetUID("test-uid")

	cl := fake.NewClientBuilder().WithScheme(scheme).Build()
	r := &AutoScalerReconciler{Client: cl, Scheme: scheme}

	spec := &parsedSpec{
		profilingImage: "ghcr.io/rolebasedgroup/rbg-profiler:latest",
		backend:        "sglang-oai",
		imagePullSecrets: []corev1.LocalObjectReference{
			{Name: "regcred"},
		},
	}

	err := r.createProfilingJob(context.Background(), ras, spec, "test-rbg-profiling", "test-rbg-profiling")
	if err != nil {
		t.Fatalf("createProfilingJob failed: %v", err)
	}

	created := &batchv1.Job{}
	if err := cl.Get(context.Background(),
		types.NamespacedName{Name: "test-rbg-profiling", Namespace: "test-ns"}, created); err != nil {
		t.Fatalf("failed to get created job: %v", err)
	}

	if len(created.Spec.Template.Spec.ImagePullSecrets) != 1 {
		t.Fatalf("expected 1 imagePullSecret, got %d", len(created.Spec.Template.Spec.ImagePullSecrets))
	}
	if created.Spec.Template.Spec.ImagePullSecrets[0].Name != "regcred" {
		t.Errorf("expected 'regcred', got %q", created.Spec.Template.Spec.ImagePullSecrets[0].Name)
	}
}
```

- [ ] **Step 2b: Write failing test — ensurePlannerDeployment sets ImagePullSecrets**

Add `appsv1 "k8s.io/api/apps/v1"` to the test file imports, then add to `internal/controller/roleautoscaler_controller_test.go`:

```go
func TestEnsurePlannerDeployment_SetsImagePullSecrets(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := appsv1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add apps scheme: %v", err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add core scheme: %v", err)
	}

	ras := &unstructured.Unstructured{}
	ras.SetName("test-rbg")
	ras.SetNamespace("test-ns")
	ras.SetUID("test-uid")

	cl := fake.NewClientBuilder().WithScheme(scheme).Build()
	r := &AutoScalerReconciler{Client: cl, Scheme: scheme}

	spec := &parsedSpec{
		plannerImage: "ghcr.io/rolebasedgroup/rbg-planner:latest",
		imagePullSecrets: []corev1.LocalObjectReference{
			{Name: "regcred"},
		},
	}

	err := r.ensurePlannerDeployment(context.Background(), ras, spec, "test-profiling-cm", 1, 1)
	if err != nil {
		t.Fatalf("ensurePlannerDeployment failed: %v", err)
	}

	created := &appsv1.Deployment{}
	if err := cl.Get(context.Background(),
		types.NamespacedName{Name: "test-rbg-planner", Namespace: "test-ns"}, created); err != nil {
		t.Fatalf("failed to get created deployment: %v", err)
	}

	if len(created.Spec.Template.Spec.ImagePullSecrets) != 1 {
		t.Fatalf("expected 1 imagePullSecret, got %d", len(created.Spec.Template.Spec.ImagePullSecrets))
	}
	if created.Spec.Template.Spec.ImagePullSecrets[0].Name != "regcred" {
		t.Errorf("expected 'regcred', got %q", created.Spec.Template.Spec.ImagePullSecrets[0].Name)
	}
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run:
```bash
go test ./internal/controller/ -run "TestParseRASSpec_ImagePullSecrets|TestCreateProfilingJob_SetsImagePullSecrets|TestEnsurePlannerDeployment_SetsImagePullSecrets" -v
```

Expected: FAIL — `spec.imagePullSecrets` undefined, `ImagePullSecrets` not set on PodSpec.

- [ ] **Step 4: Add ImagePullSecrets to CRD type**

In `api/v1alpha1/roleautoscaler_types.go`, add the `corev1` import:

```go
import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)
```

Add the field to `DynamoPlannerConfig` (after `DryRun`, before `Profiling`):

```go
	// ImagePullSecrets is a list of references to Kubernetes secrets used for
	// pulling the profiling and planner images from private registries.
	// Applied to both the profiling Job and planner Deployment PodSpecs.
	// +optional
	// +listType=map
	// +listMapKey=name
	ImagePullSecrets []corev1.LocalObjectReference `json:"imagePullSecrets,omitempty"`
```

- [ ] **Step 5: Add imagePullSecrets to parsedSpec and parseRASSpec**

In `internal/controller/roleautoscaler_controller.go`, add the field to the `parsedSpec` struct (after `modelCacheMountPath`):

```go
	imagePullSecrets []corev1.LocalObjectReference
```

In `parseRASSpec`, after the `dryRun` parsing (line ~293) and before the `profiling` parsing (line ~296), add:

```go
	// imagePullSecrets
	if ips, ok := dp["imagePullSecrets"].([]interface{}); ok {
		for _, item := range ips {
			if m, ok := item.(map[string]interface{}); ok {
				if name, ok := m["name"].(string); ok {
					s.imagePullSecrets = append(s.imagePullSecrets, corev1.LocalObjectReference{Name: name})
				}
			}
		}
	}
```

- [ ] **Step 6: Set ImagePullSecrets in createProfilingJob**

In `internal/controller/roleautoscaler_controller.go`, in `createProfilingJob` (line ~574), add `ImagePullSecrets` to the PodSpec:

```go
				Spec: corev1.PodSpec{
					ServiceAccountName: saName,
					RestartPolicy:      corev1.RestartPolicyNever,
					ImagePullSecrets:   spec.imagePullSecrets,
					Volumes:            volumes,
					Containers: []corev1.Container{
```

- [ ] **Step 7: Set ImagePullSecrets in ensurePlannerDeployment**

In `internal/controller/roleautoscaler_controller.go`, in `ensurePlannerDeployment` (line ~628), add `ImagePullSecrets` to the PodSpec:

```go
				Spec: corev1.PodSpec{
					ServiceAccountName: saName,
					ImagePullSecrets:   spec.imagePullSecrets,
					Containers: []corev1.Container{
```

- [ ] **Step 8: Regenerate deepcopy and CRD manifest**

Run:
```bash
make generate && make manifests
```

- [ ] **Step 9: Run tests to verify they pass**

Run:
```bash
go build ./... && go test ./internal/controller/ -run "TestParseRASSpec_ImagePullSecrets|TestCreateProfilingJob_SetsImagePullSecrets|TestEnsurePlannerDeployment_SetsImagePullSecrets" -v
```

Expected: PASS.

- [ ] **Step 10: Run full test suite**

Run:
```bash
go test ./... -v
```

Expected: all tests PASS.

- [ ] **Step 11: Commit**

```bash
git add api/v1alpha1/ internal/controller/ config/crd/
git commit -m "feat(api): add imagePullSecrets to dynamoPlanner config

Sets ImagePullSecrets on both profiling Job and planner Deployment
PodSpecs. Deployment updates propagate via existing DeepEqual; Job is
one-shot and recreated on failure."
```

---

### Task 3: Generation-Guarded Failed Recovery

Adds `lastProfilingFailureGeneration` to status, relaxes the Failed short-circuit to retry when `spec.generation` has advanced, and adjusts `handleProfiling` to return not-ready (not error) when a failed Job is still being deleted.

**Files:**
- Modify: `api/v1alpha1/roleautoscaler_types.go` (add `LastProfilingFailureGeneration` to status)
- Modify: `api/v1alpha1/zz_generated.deepcopy.go` (regenerated)
- Modify: `internal/controller/roleautoscaler_controller.go:184-197` (Reconcile Failed check + failure status), `:486-493` (handleProfiling)
- Modify: `internal/controller/roleautoscaler_controller_test.go` (update existing test + new tests)
- Modify: `config/crd/inference-extension.rolebasedgroup.io_autoscalers.yaml` (regenerated)

**Interfaces:**
- Consumes: `dynamoPlanner` JSON key from Task 1
- Produces: `shouldShortCircuitFailed` helper, updated `handleProfiling` return semantics

- [ ] **Step 1: Write failing test — shouldShortCircuitFailed helper**

Add to `internal/controller/roleautoscaler_controller_test.go`:

```go
func TestShouldShortCircuitFailed(t *testing.T) {
	tests := []struct {
		name     string
		phase    string
		lastFail int64
		gen      int64
		want     bool
	}{
		{"not failed", "Initializing", 0, 1, false},
		{"failed no spec change", "Failed", 2, 2, true},
		{"failed gen behind", "Failed", 2, 1, true},
		{"failed gen advanced", "Failed", 2, 3, false},
		{"failed no lastFailGen recorded", "Failed", 0, 1, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ras := &unstructured.Unstructured{}
			ras.SetGeneration(tt.gen)
			ras.Object["status"] = map[string]interface{}{
				"phase":                           tt.phase,
				"lastProfilingFailureGeneration": tt.lastFail,
			}
			if got := shouldShortCircuitFailed(ras); got != tt.want {
				t.Errorf("shouldShortCircuitFailed() = %v, want %v", got, tt.want)
			}
		})
	}
}
```

- [ ] **Step 2: Write failing test — handleProfiling returns not-ready for deleting Job**

Update the existing test `TestHandleProfiling_FailedJobAlreadyDeletingSkipsDelete` (line ~396). Change the expectation from error to not-ready:

```go
func TestHandleProfiling_FailedJobAlreadyDeleting_ReturnsNotReady(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := batchv1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add batch scheme: %v", err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add core scheme: %v", err)
	}

	ras := &unstructured.Unstructured{}
	ras.SetName("test-rbg")
	ras.SetNamespace("test-ns")

	deletionTime := metav1.Now()
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "test-rbg-profiling",
			Namespace:         "test-ns",
			DeletionTimestamp: &deletionTime,
			Finalizers:        []string{"foregroundDeletion"},
		},
		Status: batchv1.JobStatus{Failed: 1},
	}

	cl := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(job).
		Build()

	r := &AutoScalerReconciler{Client: cl, Scheme: scheme}

	_, ready, err := r.handleProfiling(context.Background(), ras, &parsedSpec{})
	if err != nil {
		t.Fatalf("expected no error for deleting job, got: %v", err)
	}
	if ready {
		t.Error("expected not-ready for deleting job")
	}

	// Job should still exist, unchanged — no second delete issued.
	got := &batchv1.Job{}
	if err := cl.Get(context.Background(),
		types.NamespacedName{Name: "test-rbg-profiling", Namespace: "test-ns"}, got); err != nil {
		t.Fatalf("expected job to still exist, got error: %v", err)
	}
	if got.GetDeletionTimestamp() == nil {
		t.Error("expected deletionTimestamp to still be set")
	}
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run:
```bash
go test ./internal/controller/ -run "TestShouldShortCircuitFailed|TestHandleProfiling_FailedJobAlreadyDeleting" -v
```

Expected: FAIL — `shouldShortCircuitFailed` undefined; existing test expects error but will need code change.

- [ ] **Step 4: Add LastProfilingFailureGeneration to status type**

In `api/v1alpha1/roleautoscaler_types.go`, add to `AutoScalerStatus` (after `LastTransitionTime`):

```go
	// LastProfilingFailureGeneration records the spec generation at which
	// profiling last failed. Used to gate retry: the controller retries
	// profiling only when spec.generation has advanced past this value.
	// +optional
	LastProfilingFailureGeneration int64 `json:"lastProfilingFailureGeneration,omitempty"`
```

- [ ] **Step 5: Add shouldShortCircuitFailed helper**

In `internal/controller/roleautoscaler_controller.go`, add a new helper function (place it near the other helpers, e.g., after `boolToConditionStatus`):

```go
// shouldShortCircuitFailed returns true when the AutoScaler is in Failed phase
// and the spec generation has not advanced since the failure was recorded.
// When false (generation advanced), the reconciler falls through to retry.
func shouldShortCircuitFailed(ras *unstructured.Unstructured) bool {
	phase, _, _ := unstructured.NestedString(ras.Object, "status", "phase")
	if phase != "Failed" {
		return false
	}
	lastFailGen, _, _ := unstructured.NestedInt64(ras.Object, "status", "lastProfilingFailureGeneration")
	return ras.GetGeneration() <= lastFailGen
}
```

- [ ] **Step 6: Relax Failed short-circuit in Reconcile**

In `internal/controller/roleautoscaler_controller.go`, replace lines 184-189:

Before:
```go
	// 6. Handle Profiling.
	// If profiling previously failed, stop reconciling to avoid loops.
	// The user must delete and recreate the AutoScaler to retry.
	if phase, _, _ := unstructured.NestedString(ras.Object, "status", "phase"); phase == "Failed" {
		return ctrl.Result{}, nil
	}
```

After:
```go
	// 6. Handle Profiling.
	// If profiling previously failed, stop reconciling unless the spec
	// generation has advanced (user edited the spec to fix the issue).
	if shouldShortCircuitFailed(ras) {
		return ctrl.Result{}, nil
	}
```

- [ ] **Step 7: Set lastProfilingFailureGeneration on profiling failure**

In `internal/controller/roleautoscaler_controller.go`, replace the profiling failure block (lines 192-198):

Before:
```go
	profilingCMName, ready, err := r.handleProfiling(ctx, ras, spec)
	if err != nil {
		logger.Error(err, "profiling failed")
		if err := r.setPhase(ctx, ras, "Failed", "ProfilingFailed", err.Error()); err != nil {
			logger.Error(err, "failed to set phase")
		}
		return ctrl.Result{}, nil
	}
```

After:
```go
	profilingCMName, ready, err := r.handleProfiling(ctx, ras, spec)
	if err != nil {
		logger.Error(err, "profiling failed")
		// Record the generation at failure time and set Failed phase.
		// Inline status update (not setPhase) to bypass dedup and ensure
		// lastProfilingFailureGeneration is always persisted.
		status, _ := ras.Object["status"].(map[string]interface{})
		if status == nil {
			status = map[string]interface{}{}
		}
		status["phase"] = "Failed"
		status["lastProfilingFailureGeneration"] = ras.GetGeneration()
		status["conditions"] = []interface{}{
			map[string]interface{}{
				"type":               "Ready",
				"status":             "False",
				"lastTransitionTime": metav1.Now().Format(time.RFC3339),
				"reason":             "ProfilingFailed",
				"message":            err.Error(),
			},
		}
		ras.Object["status"] = status
		if updateErr := r.Status().Update(ctx, ras); updateErr != nil {
			logger.Error(updateErr, "failed to set phase")
		}
		return ctrl.Result{}, nil
	}
```

- [ ] **Step 8: Adjust handleProfiling — Job being deleted returns not-ready**

In `internal/controller/roleautoscaler_controller.go`, in `handleProfiling` (lines 486-493), change the `deletionTimestamp != nil` branch:

Before:
```go
	if job.Status.Failed > 0 {
		if job.GetDeletionTimestamp() == nil {
			if delErr := r.Delete(ctx, job, client.PropagationPolicy(metav1.DeletePropagationForeground)); delErr != nil && !apierrors.IsNotFound(delErr) {
				return "", false, fmt.Errorf("profiling job %s failed and cleanup failed: %w", jobName, delErr)
			}
		}
		return "", false, fmt.Errorf("profiling job %s failed", jobName)
	}
```

After:
```go
	if job.Status.Failed > 0 {
		if job.GetDeletionTimestamp() == nil {
			if delErr := r.Delete(ctx, job, client.PropagationPolicy(metav1.DeletePropagationForeground)); delErr != nil && !apierrors.IsNotFound(delErr) {
				return "", false, fmt.Errorf("profiling job %s failed and cleanup failed: %w", jobName, delErr)
			}
			return "", false, fmt.Errorf("profiling job %s failed", jobName)
		}
		// Job is being deleted — wait for it to be gone before recreating.
		return "", false, nil
	}
```

- [ ] **Step 9: Regenerate deepcopy and CRD manifest**

Run:
```bash
make generate && make manifests
```

- [ ] **Step 10: Run tests to verify they pass**

Run:
```bash
go test ./internal/controller/ -run "TestShouldShortCircuitFailed|TestHandleProfiling_FailedJobAlreadyDeleting" -v
```

Expected: PASS.

- [ ] **Step 11: Run full test suite**

Run:
```bash
go test ./... -v
```

Expected: all tests PASS.

- [ ] **Step 12: Commit**

```bash
git add api/v1alpha1/ internal/controller/ config/crd/
git commit -m "feat(controller): generation-guarded Failed recovery for profiling

Adds status.lastProfilingFailureGeneration. The Failed short-circuit
now retries when spec.generation has advanced, so users can fix
imagePullSecrets (or other issues) and the controller auto-retries
without delete/recreate. handleProfiling returns not-ready (not error)
when a failed Job is still being deleted, preventing premature
re-Failed during retry."
```
