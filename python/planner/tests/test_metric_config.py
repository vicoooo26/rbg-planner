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

    def test_raw_template_camelcase_key_mapped(self, mock_prom):
        mock_prom.custom_query.return_value = [{"value": [0, "0.5"]}]
        cfg = {"rawTemplates": {"requestDuration": "avg(custom{labelFilter})"}}
        adapter = create_metrics_adapter("vllm", "http://prom:9090", cfg)
        adapter.get_avg_request_duration("180s", "Qwen")
        query = mock_prom.custom_query.call_args.kwargs["query"]
        assert query == 'avg(custom{model_name="Qwen"})'

    def test_invalid_shape_raises(self, mock_prom):
        cfg = {"metricShapes": {"ttft": "bogus"}}
        with pytest.raises(ValueError, match="Invalid shape values"):
            create_metrics_adapter("vllm", "http://prom:9090", cfg)


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

    def test_raw_template_unknown_key_passthrough(self, mock_prom):
        cfg = {"rawTemplates": {"myCustomMetric": "avg(custom{labelFilter})"}}
        adapter = create_metrics_adapter("vllm", "http://prom:9090", cfg)
        assert "myCustomMetric" in adapter.raw_templates
        assert adapter.raw_templates["myCustomMetric"] == "avg(custom{labelFilter})"
