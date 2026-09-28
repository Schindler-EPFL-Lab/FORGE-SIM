import drjit as dr
import mitsuba as mi

from forge_sim.meshing.mesh import Mesh
from forge_sim.regularization.regularization import Regularization
from forge_sim.regularization.sparse_laplacian_matrix import SparseLaplacianMatrix


class PositionLoss(Regularization):
    """
    A regularization loss that encourages the vertex positions of a mesh to be
    similar to the average of their neighboring vertex positions.

    The mesh is encouraged to be smooth and isotropic, which can help to reduce noise
    in the position field. By default, the Laplacian loss is averaged over vertices,
    so increasing the number of vertices will reduce the difference between the
    average position and the actual position of the vertex. This is countered by
    dividing by
    """

    def __init__(self, mesh: Mesh, weight: float = 1.0) -> None:
        """
        Returns an instance of position loss for the provided `mesh`.

        Mesh faces are used to construct the combinatorial Laplacian matrix, and the
        vertex positions are used as the values to average.

        This loss tries to make the mesh smooth and isotropic by computing the
        difference between the average position of neighboring vertices and the
        actual position of each vertex.

        :raises ValueError: If the weight is negative.
        """
        # Check for negative weight done in the `Regularization` base class
        super().__init__(weight=weight)
        self._mesh = mesh
        self._laplacian = SparseLaplacianMatrix(faces=self._mesh.faces.numpy())

    def evaluate(self) -> "mi.TensorXf":
        laplacian_product = self._laplacian.apply_to(
            values_to_average=self._mesh.vertices,
            positions=self._mesh.vertices,
        )
        dot_product = dr.sum(laplacian_product * laplacian_product, axis=1)

        return self.weight * dr.mean(dot_product)
