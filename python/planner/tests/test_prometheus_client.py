"""Unit tests for the metrics adapter module."""

from unittest.mock import MagicMock, patch

import pytest

from rbg_planner.metrics import SUPPORTED_SOURCES, create_metrics_adapter
from rbg_planner.metrics.base import MetricsAdapter
from rbg_planner.metrics.defaults import BUILTIN_ENGINES
from rbg_planner.metrics.prometheus import MetricNames, PrometheusAdapter


@pytest.fixture
def mock_prom():
    with patch("rbg_planner.metrics.prometheus.PrometheusConnect") as mock_cls:
        mock_conn = MagicMock()
        mock_cls.return_value = mock_conn
        yield mock_conn


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


class TestLabelFilter:
    def test_no_filters(self, mock_prom):
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        assert adapter._build_label_filter() == ""

    def test_model_filter_sglang(self, mock_prom):
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        assert adapter._build_label_filter(model_name="Qwen/Qwen3-0.6B") == '{model_name="Qwen/Qwen3-0.6B"}'

    def test_model_filter_dynamo(self, mock_prom):
        adapter = create_metrics_adapter("dynamo", "http://prom:9090")
        assert adapter._build_label_filter(model_name="Qwen/Qwen3-0.6B") == '{model="Qwen/Qwen3-0.6B"}'


class TestQueryMetrics:
    def test_avg_ttft_with_data(self, mock_prom):
        mock_prom.custom_query.return_value = [
            {"value": [1234567890, "0.5"]},
            {"value": [1234567890, "0.7"]},
        ]
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        result = adapter.get_avg_ttft("180s")
        assert result == pytest.approx(0.6, rel=0.01)

    def test_avg_ttft_no_data(self, mock_prom):
        mock_prom.custom_query.return_value = []
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        result = adapter.get_avg_ttft("180s")
        assert result == 0.0

    def test_avg_ttft_nan_filtered(self, mock_prom):
        mock_prom.custom_query.return_value = [
            {"value": [1234567890, "0.5"]},
            {"value": [1234567890, "NaN"]},
        ]
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        result = adapter.get_avg_ttft("180s")
        assert result == pytest.approx(0.5)

    def test_request_count(self, mock_prom):
        mock_prom.custom_query.return_value = [
            {"value": [1234567890, "100"]},
            {"value": [1234567890, "50"]},
        ]
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        result = adapter.get_request_count("180s")
        assert result == 150.0

    def test_error_returns_zero(self, mock_prom):
        mock_prom.custom_query.side_effect = Exception("connection error")
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        result = adapter.get_avg_ttft("180s")
        assert result == 0.0

    def test_avg_isl(self, mock_prom):
        mock_prom.custom_query.return_value = [
            {"value": [1234567890, "512.0"]},
        ]
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        result = adapter.get_avg_isl("180s")
        assert result == 512.0

    def test_avg_osl(self, mock_prom):
        mock_prom.custom_query.return_value = [
            {"value": [1234567890, "128.0"]},
        ]
        adapter = create_metrics_adapter("sglang", "http://prom:9090")
        result = adapter.get_avg_osl("180s")
        assert result == 128.0


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
