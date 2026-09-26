from dataclasses import dataclass

from fem.shells.rigid_body_mode import RigidBodyMode


@dataclass(frozen=True, kw_only=True)
class RigidBodyModeEnergy:
    """
    The Rayleigh-quotient "frequency" [Hz] of one analytic rigid-body field.

    A closed shell has exactly six zero-energy motions, so `frequency` should be
    ~0 for every mode; it is evaluated on the interpolated analytic field rather
    than on a computed eigenvector, which isolates the strain measures from the
    eigensolver. `frequency` is `nan` when `mode` carries no modal mass, which
    happens once the rotary inertia of the drilling rotation is projected out.

    Pairs the quotient with the `RigidBodyMode` it was computed from, rather
    than a bare float keyed by a hand-written label: `mode.label` is the one
    place the six tags (`"Tx"`, ..., `"Rz"`) are spelled out.
    """

    mode: RigidBodyMode
    """The rigid-body field this quotient was computed from."""
    frequency: float
    """The Rayleigh-quotient frequency, in Hz."""
