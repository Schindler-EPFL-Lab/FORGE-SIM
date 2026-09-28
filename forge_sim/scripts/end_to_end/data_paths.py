from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[3]
"""
Repository root, derived from this file's location.

`main_scripts/` needs it to set the working directory of the stages it spawns and
to locate the conda-side ones, neither of which a relative path can express. The
stage configs do not: their defaults are relative and resolve against that same
working directory.
"""
