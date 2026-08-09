# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI-based bench executor template.

CLIBenchExecutor is a template method base for bench tools invoked as
subprocesses (inference-perf, aiperf). Subclasses implement build_args()
and parse_report(); the execute() method handles subprocess lifecycle,
error handling, and report discovery.
"""

import asyncio
import logging
import os
import tempfile

from abc import abstractmethod

from rbg_profiler.bench.executor import (
    BenchExecutionError,
    BenchExecutor,
    BenchParams,
    BenchResult,
)

logger = logging.getLogger(__name__)


class CLIBenchExecutor(BenchExecutor):
    """Template for CLI-based bench tools.

    Subclasses implement build_args() and parse_report().
    The execute() method handles subprocess lifecycle, error handling,
    and report discovery.
    """

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Name of the CLI tool, used in error messages."""
        raise NotImplementedError

    @abstractmethod
    def build_args(self, params: BenchParams, report_dir: str) -> list[str]:
        """Build CLI args list from params. Pure function."""
        raise NotImplementedError

    @abstractmethod
    def find_report_path(self, report_dir: str) -> str:
        """Locate the report JSON file within the output directory."""
        raise NotImplementedError

    @abstractmethod
    def parse_report(self, report_path: str) -> BenchResult:
        """Parse the tool's report file into BenchResult."""
        raise NotImplementedError

    def _make_report_dir(self, params: BenchParams) -> str:
        """Create a unique report directory for this benchmark point."""
        dir_name = f"{self.tool_name}_isl{params.isl}_osl{params.osl}_n{params.num_prompts}"
        report_dir = os.path.join(tempfile.gettempdir(), "rbg-bench-reports", dir_name)
        os.makedirs(report_dir, exist_ok=True)
        return report_dir

    async def execute(self, params: BenchParams) -> BenchResult:
        """Run the CLI tool and parse its report."""
        report_dir = self._make_report_dir(params)
        args = self.build_args(params, report_dir)

        logger.info("Running %s: %s", self.tool_name, " ".join(args))

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
