# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in engine defaults: metric names + shape overrides per engine.

Replaces the former per-engine adapter subclasses (SGLangAdapter, VLLMAdapter,
DynamoAdapter, PatioAdapter). The factory merges user-provided metricConfig
over these defaults.
"""

from rbg_planner.metrics.prometheus import MetricNames

# Engine name -> {"names": MetricNames, "shapes": dict[str, str]}.
# An empty "shapes" dict means all metrics use DEFAULT_SHAPES.
BUILTIN_ENGINES: dict[str, dict] = {
    "sglang": {
        "names": MetricNames(
            ttft="sglang_time_to_first_token_seconds",
            itl="sglang_inter_token_latency_seconds",
            request_duration="sglang_e2e_request_latency_seconds",
            requests_total="sglang_num_requests_total",
            prompt_tokens="sglang_prompt_tokens_total",
            generation_tokens="sglang_generation_tokens_total",
            model_label="model_name",
        ),
        "shapes": {},
    },
    "vllm": {
        "names": MetricNames(
            ttft="vllm:time_to_first_token_seconds",
            itl="vllm:time_per_output_token_seconds",
            request_duration="vllm:e2e_request_latency_seconds",
            requests_total="vllm:num_requests_total",
            prompt_tokens="vllm:prompt_tokens_total",
            generation_tokens="vllm:generation_tokens_total",
            model_label="model_name",
        ),
        "shapes": {},
    },
    "dynamo": {
        "names": MetricNames(
            ttft="dynamo_frontend_time_to_first_token_seconds",
            itl="dynamo_frontend_inter_token_latency_seconds",
            request_duration="dynamo_frontend_request_duration_seconds",
            requests_total="dynamo_frontend_requests_total",
            prompt_tokens="dynamo_frontend_input_sequence_tokens",
            generation_tokens="dynamo_frontend_output_sequence_tokens",
            model_label="model",
        ),
        # Dynamo exposes ISL/OSL as histograms, not counter ratios.
        "shapes": {"isl": "histogram_avg", "osl": "histogram_avg"},
    },
    "patio": {
        "names": MetricNames(
            ttft="patio:time_to_first_token_seconds",
            itl="patio:inter_token_latency_seconds",
            request_duration="patio:request_duration_seconds",
            requests_total="patio:requests_total",
            prompt_tokens="patio:prompt_tokens_total",
            generation_tokens="patio:generation_tokens_total",
            model_label="model_name",
        ),
        "shapes": {},
    },
}
