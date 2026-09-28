import abc
from enum import Enum

import mitsuba as mi


class Patch(abc.ABC):
    """
    Abstract base class for rectangular 2D surface segments in 3D space - patches.

    Patches can originate from different underlying primitives, but their shape is
    always rectangular. All explicit surfaces are defined by some `nodes` (e.g.
    vertices for meshes, control points for splines) so they should be able to
    provide their nodes and a method to get the indices of the nodes that belong to
    a specific side of the patch.

    Patches are the building blocks for more complex mesh or spline structures and
    are consumed by the `PatchManager` that provides a unified interface for
    building more complex structures from abstract patches.
    """

    class Side(Enum):
        """
        Represents the four sides of a rectangular patch.
        """

        LEFT = 0
        RIGHT = 1
        TOP = 2
        BOTTOM = 3

        @classmethod
        def all_sides(cls) -> list["Patch.Side"]:
            """
            Returns a list of all sides of the patch.

            Useful for operations that require iterating over all sides.

            :returns: A list containing all four sides of the patch.
            """
            return [cls.LEFT, cls.RIGHT, cls.BOTTOM, cls.TOP]

    @abc.abstractmethod
    def get_node_indices_for(self, side: Side) -> "mi.ArrayXu":
        """
        Returns the node indices at the specified side of the patch.

        `side` is an instance of the `Patch.Side` enum that specifies which side
        of the patch to query.

        :returns: A Mitsuba array of vertex indices that belong to the specified side.
        :raises NotImplementedError: If a child class does not implement this method.
        """
        raise NotImplementedError(
            "Classes that inherit from `Patch` must implement the "
            "`get_node_indices_for` method."
        )

    @property
    @abc.abstractmethod
    def nodes(self) -> "mi.TensorXf":
        """
        Nodes of the patch.

        Nodes are a generic term that refers to 3D points in space connected in some
        manner. For triangles those are the vertices, while for splines those are
        the control points.

        :returns: A Mitsuba tensor containing the nodes of the patch.
        :raises NotImplementedError: If a child class does not implement this property.
        """
        raise NotImplementedError(
            "Classes that inherit from `Patch` must implement the `nodes` property."
        )

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Name of the patch.

        Mitsuba uses strings as keys to identify patches, so each patch should have a
        unique name.

        :returns: The name of the patch.
        :raises NotImplementedError: If a child class does not implement this property.
        """
        raise NotImplementedError(
            "Classes that inherit from `Patch` must implement the `name` property."
        )
