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
        with patch(
            "sys.argv",
            [
                "planner",
                "--metric-source",
                "acme-llm",
                "--metric-mapping-json",
                json_str,
            ],
        ):
            args = parse_args()
        config = build_config(args)
        assert config.metric_source == "acme-llm"
        assert config.metric_mapping_json == json_str
