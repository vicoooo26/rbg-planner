# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the built-in engine defaults registry."""

from rbg_planner.metrics.defaults import BUILTIN_ENGINES
from rbg_planner.metrics.prometheus import DEFAULT_SHAPES


class TestBuiltinEngines:
    def test_sglang_names(self):
        n = BUILTIN_ENGINES["sglang"]["names"]
        assert n.ttft == "sglang_time_to_first_token_seconds"
        assert n.itl == "sglang_inter_token_latency_seconds"
        assert n.request_duration == "sglang_e2e_request_latency_seconds"
        assert n.requests_total == "sglang_num_requests_total"
        assert n.prompt_tokens == "sglang_prompt_tokens_total"
        assert n.generation_tokens == "sglang_generation_tokens_total"
        assert n.model_label == "model_name"

    def test_vllm_names(self):
        n = BUILTIN_ENGINES["vllm"]["names"]
        assert n.ttft == "vllm:time_to_first_token_seconds"
        assert n.requests_total == "vllm:num_requests_total"
        assert n.model_label == "model_name"

    def test_dynamo_names_and_shapes(self):
        e = BUILTIN_ENGINES["dynamo"]
        n = e["names"]
        assert n.ttft == "dynamo_frontend_time_to_first_token_seconds"
        assert n.prompt_tokens == "dynamo_frontend_input_sequence_tokens"
        assert n.generation_tokens == "dynamo_frontend_output_sequence_tokens"
        assert n.model_label == "model"
        # Dynamo overrides ISL/OSL to histogram_avg (replaces the old subclass override).
        assert e["shapes"]["isl"] == "histogram_avg"
        assert e["shapes"]["osl"] == "histogram_avg"

    def test_patio_names(self):
        n = BUILTIN_ENGINES["patio"]["names"]
        assert n.ttft == "patio:time_to_first_token_seconds"
        assert n.requests_total == "patio:requests_total"

    def test_builtin_shapes_default_empty(self):
        # sglang/vllm/patio use all default shapes (no overrides).
        for engine in ("sglang", "vllm", "patio"):
            assert BUILTIN_ENGINES[engine]["shapes"] == {}

    def test_default_shapes_complete(self):
        assert set(DEFAULT_SHAPES.keys()) == {
            "ttft", "itl", "request_duration", "requests_total", "isl", "osl",
        }
