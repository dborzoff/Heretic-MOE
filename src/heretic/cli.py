# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lightweight command dispatcher for the HereticMOE executable."""

from __future__ import annotations

import os
import sys
from importlib.metadata import version


def main() -> None:
    if any(argument in {"-V", "--version"} for argument in sys.argv[1:]):
        print(f"Heretic-MOE {version('heretic-llm')}")
        return
    if len(sys.argv) > 1 and sys.argv[1] in {"geometry-map", "language-map"}:
        from .language_map_cli import main as language_map_main

        language_map_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "srg-benchmark":
        from .srg_benchmark import main as srg_benchmark_main

        srg_benchmark_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "prepare-multilingual":
        from .multilingual_prepare_cli import main as multilingual_prepare_main

        multilingual_prepare_main(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "prepare-final-holdout":
        from .multilingual_final_holdout_cli import main as final_holdout_main

        final_holdout_main(sys.argv[2:])
        return

    if os.environ.get("HERETIC_MOE_INTERNAL") == "1":
        from .main import main as worker_main

        worker_main()
        return

    from .supervisor import main as supervisor_main

    supervisor_main(sys.argv[1:])
