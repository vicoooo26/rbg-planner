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
            "sglang", "http://prom:9090", metric_config,
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
