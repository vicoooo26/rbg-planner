# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profiling pipeline: sweep orchestration + result adaptation.

Owns the sweep logic (iterate ISL/concurrency points) and adaptation
(BenchResult → PrefillProfileData/DecodeProfileData/UnifiedProfileData).
Calls executor.execute() per sweep point — the executor is the only
pluggable part.
"""

import logging

from rbg_profiler.bench.base import (
    ContextLengthExceededError,
    DecodeProfileData,
    PrefillProfileData,
    UnifiedProfileData,
)
from rbg_profiler.bench.executor import BenchExecutor, BenchParams
from rbg_profiler.rbg_discoverer import RoleEndpoint

logger = logging.getLogger(__name__)


class ProfilingPipeline:
    """Orchestrates bench sweeps using a pluggable executor."""

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
        """Profile prefill across ISL points."""
        results: list[PrefillProfileData] = []
        for isl in isl_points:
            if isl >= max_context_length - 512:
                logger.info("Skipping ISL=%d (exceeds max_context_length)", isl)
                continue
            try:
                result = await self.executor.execute(
                    BenchParams(
                        endpoint=endpoint,
                        model=model,
                        isl=isl,
                        osl=osl,
                        num_prompts=num_prompts,
                    )
                )
            except ContextLengthExceededError:
                logger.info(
                    "ISL=%d rejected with HTTP 400, stopping prefill sweep", isl
                )
                break
            except Exception:
                logger.exception("ISL=%d: benchmark failed, skipping", isl)
                continue
            if not (result.mean_ttft_ms > 0) or endpoint.num_gpus <= 0:
                logger.warning(
                    "ISL=%d: invalid metrics (ttft=%s, gpus=%d), skipping",
                    isl,
                    result.mean_ttft_ms,
                    endpoint.num_gpus,
                )
                continue
            thpt = isl / result.mean_ttft_ms * 1000 / endpoint.num_gpus
            results.append(
                PrefillProfileData(
                    isl=isl, ttft_ms=result.mean_ttft_ms, thpt_per_gpu=round(thpt, 2)
                )
            )
            logger.info(
                "ISL=%d: TTFT=%.2fms, thpt/gpu=%.1f tok/s",
                isl,
                result.mean_ttft_ms,
                thpt,
            )
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
        """Profile decode across ISL x concurrency grid."""
        results: list[DecodeProfileData] = []
        for isl in isl_points:
            if isl + osl >= max_context_length:
                logger.info("Skipping ISL=%d (isl+osl exceeds context length)", isl)
                continue
            context_length = isl + osl // 2
            max_concurrency = min(max_kv_tokens // (isl + osl), 512)
            if max_concurrency < 1:
                logger.warning("ISL=%d: max_concurrency=0, skipping", isl)
                continue
            if max_concurrency <= concurrency_granularity:
                concurrency_points = list(range(1, max_concurrency + 1))
            else:
                step = (max_concurrency - 1) / (concurrency_granularity - 1)
                concurrency_points = sorted(
                    set(
                        max(1, int(1 + i * step))
                        for i in range(concurrency_granularity)
                    )
                )
            logger.info(
                "ISL=%d: context_len=%d, max_conc=%d, sweep=%s",
                isl,
                context_length,
                max_concurrency,
                concurrency_points,
            )
            stop_sweep = False
            for conc in concurrency_points:
                try:
                    result = await self.executor.execute(
                        BenchParams(
                            endpoint=endpoint,
                            model=model,
                            isl=isl,
                            osl=osl,
                            num_prompts=conc,
                        )
                    )
                except ContextLengthExceededError:
                    logger.info(
                        "ISL=%d rejected with HTTP 400, stopping decode sweep", isl
                    )
                    stop_sweep = True
                    break
                except Exception:
                    logger.exception(
                        "ISL=%d conc=%d: benchmark failed, skipping", isl, conc
                    )
                    continue
                if endpoint.num_gpus <= 0:
                    logger.warning(
                        "ISL=%d conc=%d: invalid num_gpus, skipping", isl, conc
                    )
                    continue
                kv_usage = (isl + osl / 2) * conc / max_kv_tokens
                thpt_per_gpu = result.output_throughput / endpoint.num_gpus
                results.append(
                    DecodeProfileData(
                        isl=isl,
                        concurrency=conc,
                        kv_usage=round(kv_usage, 6),
                        context_length=context_length,
                        itl_ms=round(result.mean_itl_ms, 3),
                        thpt_per_gpu=round(thpt_per_gpu, 2),
                    )
                )
                logger.info(
                    "  conc=%d: kv_usage=%.4f, ITL=%.2fms, thpt/gpu=%.1f tok/s",
                    conc,
                    kv_usage,
                    result.mean_itl_ms,
                    thpt_per_gpu,
                )
            if stop_sweep:
                break
        return results

    async def bench_unified(
        self,
        endpoint: RoleEndpoint,
        model: str,
        isl_points: list[int],
        concurrency_grid: list[int],
        osl: int = 500,
    ) -> list[UnifiedProfileData]:
        """Profile unified prefill+decode."""
        results: list[UnifiedProfileData] = []
        for isl in isl_points:
            stop_sweep = False
            for conc in concurrency_grid:
                try:
                    result = await self.executor.execute(
                        BenchParams(
                            endpoint=endpoint,
                            model=model,
                            isl=isl,
                            osl=osl,
                            num_prompts=conc,
                        )
                    )
                except ContextLengthExceededError:
                    logger.info(
                        "ISL=%d rejected with HTTP 400, stopping unified sweep", isl
                    )
                    stop_sweep = True
                    break
                if endpoint.num_gpus <= 0:
                    logger.warning(
                        "ISL=%d conc=%d: invalid num_gpus, skipping", isl, conc
                    )
                    continue
                thpt_per_gpu = result.output_throughput / endpoint.num_gpus
                results.append(
                    UnifiedProfileData(
                        isl=isl,
                        concurrency=conc,
                        ttft_ms=round(result.mean_ttft_ms, 2),
                        itl_ms=round(result.mean_itl_ms, 3),
                        thpt_per_gpu=round(thpt_per_gpu, 2),
                    )
                )
            if stop_sweep:
                break
        return results
