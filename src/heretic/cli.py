# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lightweight command dispatcher for the HereticMOE executable."""

from __future__ import annotations

import os
import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"geometry-map", "language-map"}:
        from .language_map_cli import main as language_map_main

        language_map_main(sys.argv[2:])
        return

    if os.environ.get("HERETIC_MOE_INTERNAL") == "1":
        from .main import main as worker_main

        worker_main()
        return

    from .supervisor import main as supervisor_main

    supervisor_main(sys.argv[1:])
