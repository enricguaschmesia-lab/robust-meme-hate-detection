"""Cluster-friendly entrypoint for the perturbation inspection script.

Wraps ``scripts/inspect_perturbations.py`` so the cluster launcher
``scripts/cluster_entrypoint.sh`` (which always invokes ``python3 -m
<module>``) can run it. The script body is path-loaded so we don't
duplicate logic.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / 'scripts' / 'inspect_perturbations.py'
    runpy.run_path(str(script_path), run_name='__main__')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
