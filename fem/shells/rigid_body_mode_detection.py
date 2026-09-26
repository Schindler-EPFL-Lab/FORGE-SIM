from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class RigidBodyModeDetection:
    """
    Where `detect_rigid_body_modes` found the rigid-body band to end.

    `count` is read off the spectrum itself, not asserted from the expected
    count the scan window was sized by; see that function's docstring.
    """

    count: int
    """The number of modes before the largest relative gap; 0 when the scan
    had fewer than two eigenvalues to form a ratio from."""
    gap: float
    """The ratio across that gap, floored per `_NEAR_ZERO_FLOOR`; `nan` when
    `count` is 0."""
