import os
from pathlib import Path
from typing import ClassVar


class CondaPython:
    """
    Resolves the DOLFINx conda environment's python interpreter.

    Every FEM-side stage across both drivers needs this interpreter, and each asks
    for it differently: `resolve_with_override` takes an explicit CLI value with
    `ENV_VAR` as its fallback, `resolve_from_environment` reads `ENV_VAR` alone for
    a driver that offers no override of its own. Both share the one variable name
    here rather than each redeclaring it.
    """

    ENV_VAR: ClassVar[str] = "REBEL_SHAPE_DOLFINX_PYTHON"
    """
    Environment variable pointing at the DOLFINx conda environment's python
    interpreter.
    """

    @classmethod
    def resolve_with_override(cls, explicit: Path | None) -> Path:
        """
        Resolves the interpreter: `explicit` if given, else `ENV_VAR`.

        :raises ValueError: If `explicit` is `None` and `ENV_VAR` is not set, or if
            the resolved interpreter path does not exist.
        """
        if explicit is None:
            value = os.environ.get(cls.ENV_VAR)
            if value is None:
                raise ValueError(
                    f"--conda-python not given and {cls.ENV_VAR} is not set; "
                    "point it at the DOLFINx conda environment's python interpreter."
                )
            explicit = Path(value)
        return cls._require_exists(explicit)

    @classmethod
    def resolve_from_environment(cls) -> Path:
        """
        Resolves the interpreter from `ENV_VAR` alone.

        :raises ValueError: If `ENV_VAR` is not set, or names a path that does not
            exist.
        """
        value = os.environ.get(cls.ENV_VAR)
        if value is None:
            raise ValueError(
                f"{cls.ENV_VAR} is not set; point it at the DOLFINx conda "
                "environment's python interpreter."
            )
        return cls._require_exists(Path(value))

    @staticmethod
    def _require_exists(interpreter: Path) -> Path:
        """
        Checks that `interpreter` exists on disk.

        :raises ValueError: If it does not.
        """
        if not interpreter.exists():
            raise ValueError(
                f"conda_python interpreter does not exist: {interpreter}"
            )
        return interpreter
