import drjit as dr
import mitsuba as mi
import numpy as np

from forge_sim.meshing.tessellated_spline_surface import TessellatedSplineSurface
from forge_sim.regularization.regularization import Regularization


class SplineG1Loss(Regularization):
    """
    Loss that enforces tangent continuity (G1 continuity) between adjacent
    spline patches in a tessellated spline surface.

    This loss is based on the paper "Differentiable rendering of parametric geometry" by
    Markus Worchel and Marc Alexa (see the paper and implementation on GitHub).
    They propose to enforce continuity between tangent vectors of adjacent patches
    by effectively forcing the normals of adjacent patches to be equal at shared edges.

    In their formulation, they take the cross product of parallel and perpendicular
    (which they call "tangent") vectors of the patches at the shared edges. Here, we
    simplify this by directly comparing the normals of the patches at the shared edges.
    This made problems with direction of the perpendicular vectors go away and
    simplified the implementation without any noted drawbacks.
    """

    def __init__(
        self, composite_spline: TessellatedSplineSurface, weight: float = 1.0
    ) -> None:
        """
        Initializes the tangent continuity loss for a tessellated spline surface.

        :raises ValueError: If the `weight` is negative.
        """
        # Check for negative weight done in `Regularization` interface
        super().__init__(weight=weight)
        self._tessellated_spline = composite_spline

        self._coord_lists: list[tuple[int, "mi.TensorXf", int, "mi.TensorXf"]] = []
        """
        List of tuples containing the patch indices and their corresponding
        parametric coordinates for each shared edge. Each tuple is of the form:
        (patch_index_1, parametric_coords_1, patch_index_2, parametric_coords_2).
        """
        self._build_coord_lists()

    def _build_coord_lists(self) -> None:
        """
        Builds the list of parametric coordinates for each patch and side.

        The method iterates over all shared edges of the tessellated spline surface
        and computes the parametric coordinates for the vertices along each edge.
        It takes care that the edge orientation is consistent between the two
        adjacent patches. The resulting list of tuples is stored in `self._coord_lists`.
        Each tuple contains:

            - `patch_index_1`: Index of the first patch.
            - `parametric_coords_1`: Parametric coordinates of the first patch's edge.
            - `patch_index_2`: Index of the second patch.
            - `parametric_coords_2`: Parametric coordinates of the second patch's edge.

        "Caching" the parametric coordinates allows for efficient evaluation of the
        G1 continuity loss without needing to recompute them every time the loss is
        evaluated. The parametric coordinates are the midpoints of the edges
        between the vertices of the adjacent patches, making it an O(h^2) approximation
        of an integral over the entire edge (h being length / num_vertices).
        """
        self._coord_lists.clear()

        for edge in self._tessellated_spline.composite_spline.shared_edges:
            mesh_1 = self._tessellated_spline[edge.first_index]
            mesh_2 = self._tessellated_spline[edge.second_index]

            indices_1 = mesh_1.get_node_indices_for(edge.first_side)
            indices_2 = mesh_2.get_node_indices_for(edge.second_side)

            edge_vertices_1 = mesh_1.vertices[indices_1]
            edge_vertices_2 = mesh_2.vertices[indices_2]

            edge_vector_1 = (edge_vertices_1[-1] - edge_vertices_1[0]).numpy()
            edge_vector_2 = (edge_vertices_2[-1] - edge_vertices_2[0]).numpy()

            # If the edge vectors are in opposite directions, reverse the order
            if np.dot(edge_vector_1, edge_vector_2) < 0:
                temp_indices = mi.TensorXu(indices_2)
                indices_2 = dr.ravel(temp_indices[::-1])

            texcoords_1 = mesh_1.texcoords[indices_1]
            texcoords_1 = 0.5 * (texcoords_1[:-1] + texcoords_1[1:])

            texcoords_2 = mesh_2.texcoords[indices_2]
            texcoords_2 = 0.5 * (texcoords_2[:-1] + texcoords_2[1:])

            parametric_1 = self._tessellated_spline.composite_spline[
                edge.first_index
            ].texcoords_to_parametric_coords(texcoords_1)

            parametric_2 = self._tessellated_spline.composite_spline[
                edge.second_index
            ].texcoords_to_parametric_coords(texcoords_2)

            self._coord_lists.append(
                (edge.first_index, parametric_1, edge.second_index, parametric_2)
            )

    def evaluate(self) -> "mi.TensorXf":
        """
        Returns a Tensor with a single element representing the G1 continuity loss.

        Sums over the squared differences of normals at shared edges of adjacent
        spline patches. The loss is computed as follows:

            1. For each pair of adjacent patches, evaluate the normals at the
               parametric coordinates of vertices along the shared edge.
            2. Compute the average squared difference between the normals.
            3. Scale by the weight and normalize by the number of edges.

        :returns: Tensor of shape (1,) containing the regularization loss value.
        """
        spline_surface = self._tessellated_spline.composite_spline

        loss = mi.TensorXf(0.0)

        for id_1, parametric_1, id_2, parametric_2 in self._coord_lists:
            spline_1 = spline_surface[id_1]
            spline_2 = spline_surface[id_2]

            normals_1 = spline_1.evaluate_normals(parametric_1)
            normals_2 = spline_2.evaluate_normals(parametric_2)

            diff = normals_1 - normals_2

            loss += dr.mean(dr.sum(diff * diff, axis=1))

        return loss * self._weight / len(self._coord_lists)
