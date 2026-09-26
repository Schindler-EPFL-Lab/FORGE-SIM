from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, kw_only=True)
class TriangleQuality:
    """
    The standard FEM per-triangle quality indicators, one entry per face, in
    face order.
    """

    _EPS: ClassVar[float] = 1e-30
    """
    Floor under every quantity that appears in a denominator (area, semi-perimeter,
    an angle's edge-length product), so a degenerate triangle yields a large finite
    indicator instead of a division by zero. Far below any real triangle's area on
    these meshes, whose target edge lengths are ~1e-2 in model units.
    """

    skewness: npt.NDArray[np.float64]
    """Equiangle skewness; 0 is ideal, 1 degenerate."""
    aspect_ratio: npt.NDArray[np.float64]
    """Circumradius / (2 * inradius); 1 is ideal."""
    slenderness: npt.NDArray[np.float64]
    """Reciprocal shape regularity; 1 is ideal, large is degenerate."""
    min_angle: npt.NDArray[np.float64]
    """Smallest interior angle [degrees]."""
    max_angle: npt.NDArray[np.float64]
    """Largest interior angle [degrees]."""
    edge_min: npt.NDArray[np.float64]
    """Shortest edge length."""
    edge_max: npt.NDArray[np.float64]
    """Longest edge length."""
    area: npt.NDArray[np.float64]
    """Triangle area."""

    @classmethod
    def from_mesh(
        cls, vertices: npt.NDArray[np.float64], faces: npt.NDArray[np.int64]
    ) -> "TriangleQuality":
        """
        Computes per-triangle quality quantities for a triangle mesh, from the
        vertex positions `vertices` (shape `(n, 3)`) and the triangle vertex
        indices `faces` (shape `(m, 3)`).
        """
        p0, p1, p2 = (vertices[faces[:, i]] for i in range(3))
        # Edge lengths: a opposite p0, b opposite p1, c opposite p2.
        a = np.linalg.norm(p1 - p2, axis=1)
        b = np.linalg.norm(p2 - p0, axis=1)
        c = np.linalg.norm(p0 - p1, axis=1)

        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        area_safe = np.maximum(area, cls._EPS)
        s = 0.5 * (a + b + c)  # semi-perimeter

        # Interior angles via the law of cosines (clip for numerical safety).
        def angle(
            opp: npt.NDArray[np.float64],
            x: npt.NDArray[np.float64],
            y: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.float64]:
            cos = (x**2 + y**2 - opp**2) / np.maximum(2.0 * x * y, cls._EPS)
            return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

        angles = np.stack([angle(a, b, c), angle(b, c, a), angle(c, a, b)], axis=1)
        min_angle = angles.min(axis=1)
        max_angle = angles.max(axis=1)
        skewness = np.maximum((max_angle - 60.0) / 120.0, (60.0 - min_angle) / 60.0)

        circumradius = (a * b * c) / (4.0 * area_safe)
        inradius = area_safe / np.maximum(s, cls._EPS)
        aspect_ratio = circumradius / np.maximum(2.0 * inradius, cls._EPS)

        # Shape regularity = 4*sqrt(3)*A / (a^2+b^2+c^2) in (0, 1];
        # slenderness = 1/that.
        shape_regularity = (4.0 * np.sqrt(3.0) * area) / np.maximum(
            a**2 + b**2 + c**2, cls._EPS
        )
        slenderness = 1.0 / np.maximum(shape_regularity, cls._EPS)

        edges = np.stack([a, b, c], axis=1)
        return cls(
            skewness=skewness,
            aspect_ratio=aspect_ratio,
            slenderness=slenderness,
            min_angle=min_angle,
            max_angle=max_angle,
            edge_min=edges.min(axis=1),
            edge_max=edges.max(axis=1),
            area=area,
        )
