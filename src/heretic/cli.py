# SPDX-License-Identifier: AGPL-3.0-or-later

"""Lightweight command dispatcher for the HereticMOE executable."""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1].lower() == "adaptive":
        from .supervisor import main as supervisor_main

        supervisor_main(sys.argv[2:])
        return

    from .main import main as legacy_main

    legacy_main()
