import logging

import drjit as dr
import mitsuba as mi
import numpy as np

from forge_sim import logger
from forge_sim.meshing.composite_spline_surface_factory import (
    CompositeSplineSurfaceFactory,
)
from forge_sim.meshing.patch import Patch
from forge_sim.meshing.tessellated_spline_surface import TessellatedSplineSurface
from forge_sim.plugin_wrappers.diffuse_bsdf import DiffuseBSDF
from forge_sim.regularization.spline_g1_loss import SplineG1Loss
from forge_sim.splines.bspline_patch import BSplinePatch


def unit_cube(
    name: str,
    degree: int = 3,
    spline_subdivisions: int = 16,
    tessellation_subdivisions: int = 64,
    bsdf: DiffuseBSDF | None = None,
) -> TessellatedSplineSurface:
    """
    Creates a cube mesh with unit dimensions centered at the origin.

    The cube consists of six B-spline patches representing each face, with each patch
    having the specified `degree` and `spline_subdivision` knot spans. The patches are
    linked at their edges to form a continuous surface. Finally, the cube is tessellated
    into a composite mesh with `tessellation_subdivisions` per patch.

    The following layout is used for the cube faces:

                  +---------+
                  |    B    |
                  |L  top  R|
                  |    T    |
        +---------+---------+---------+---------+
        |    T    |    T    |    T    |    T    |
        |R left  L|R front L|R right L|R  back L|
        |    B    |    B    |    B    |    B    |
        +---------+---------+---------+---------+
                  |    B    |
                  |R  Bot  L|
                  |    T    |
                  +---------+

    Each patch starts from the xy plane (z = 0) with the normal pointing in the
    direction of the z axis. They have 4 edges:
        - T (top) edge: y = 0.5 and B (bottom) edge: y = -0.5
        - L (left) edge: x = -0.5 and R (right) edge: x = 0.5

    Through rotations and reflections, the patches are arranged to form the cube. The
    faces lie in the following planes:
        - Left: yz plane, x = -0.5, normal pointing in the -x direction (reflected)
        - Right: yz plane, x = 0.5, normal pointing in the +x direction
        - Back: xz plane, y = -0.5, normal pointing in the -y direction
        - Front: xz plane, y = 0.5, normal pointing in the +y direction (reflected)
        - Bottom: xy plane, z = -0.5, normal pointing in the -z direction (reflected)
        - Top: xy plane, z = 0.5, normal pointing in the +z direction

    How each patch should be connected to others is quite a confusing task, and the
    image provided above can only do so much - drawing it our on a 3D cube with
    the provided labels is the best way to understand how the patches are connected.
    There are visualize functions for splines and meshes to help.

    A warning is issued if `tessellation_subdivisions` is set to a low value as
    in this case it will not capture the spline surface well enough, resulting in
    potential errors in optimization or rendering.

    The `bsdf` parameter is optional and can be used to specify a BSDF for the
    downstream mesh. It is applied to all patches of the cube. If not provided,
    a default `DiffuseBSDF` is used.

    :returns: A tessellated spline surface representing unit cube with bounding box
              [-0.5, -0.5, -0.5] to [0.5, 0.5, 0.5].

    :raises ValueError: If `name` is an empty string, if `degree` is less than 1,
        if `spline_subdivisions` is less than 1, or if `tessellation_subdivisions` is
        less than 1.
    """
    if len(name) == 0:
        raise ValueError("`name` must be a non-empty string.")
    if degree < 1:
        raise ValueError("`degree` must be at least 1.")
    if spline_subdivisions < 1:
        raise ValueError("`spline_subdivisions` must be at least 1.")
    if tessellation_subdivisions < 1:
        raise ValueError("`tessellation_subdivisions` must be at least 1.")

    if tessellation_subdivisions < 16:
        logging.warning(
            f"Tessellation subdivisions are set to a low value "
            f"({tessellation_subdivisions}). This may result in a low-quality mesh. "
        )

    if bsdf is None:
        bsdf = DiffuseBSDF()

    # Left - yz plane, x = -0.5
    left_patch = BSplinePatch.unit_square(
        patch_name=name + "_left",
        degree=degree,
        u_subdivisions=spline_subdivisions,
        v_subdivisions=spline_subdivisions,
    )
    # Right - yz plane, x = 0.5
    right_patch = BSplinePatch.unit_square(
        patch_name=name + "_right",
        degree=degree,
        u_subdivisions=spline_subdivisions,
        v_subdivisions=spline_subdivisions,
    )
    # Back - xy plane, y = -0.5
    back_patch = BSplinePatch.unit_square(
        patch_name=name + "_back",
        degree=degree,
        u_subdivisions=spline_subdivisions,
        v_subdivisions=spline_subdivisions,
    )
    # Front - xy plane, y = 0.5
    front_patch = BSplinePatch.unit_square(
        patch_name=name + "_front",
        degree=degree,
        u_subdivisions=spline_subdivisions,
        v_subdivisions=spline_subdivisions,
    )
    # Bottom - xz plane, z = -0.5
    bottom_patch = BSplinePatch.unit_square(
        patch_name=name + "_bottom",
        degree=degree,
        u_subdivisions=spline_subdivisions,
        v_subdivisions=spline_subdivisions,
    )
    # Top - xz plane, z = 0.5
    top_patch = BSplinePatch.unit_square(
        patch_name=name + "_top",
        degree=degree,
        u_subdivisions=spline_subdivisions,
        v_subdivisions=spline_subdivisions,
    )

    # All of them start from a plane in xy, z = 0, and are then transformed
    left_patch.control_points = mi.TensorXf(
        (np.eye(3)[[2, 0, 1]] @ left_patch.control_points.numpy().T).T
        * np.array([[1.0, -1.0, 1.0]])
        + np.array([[-0.5, 0.0, 0.0]])
    )
    right_patch.control_points = mi.TensorXf(
        (np.eye(3)[[2, 0, 1]] @ right_patch.control_points.numpy().T).T
        * np.array([[1.0, 1.0, 1.0]])
        + np.array([[0.5, 0.0, 0.0]])
    )

    back_patch.control_points = mi.TensorXf(
        (np.eye(3)[[0, 2, 1]] @ back_patch.control_points.numpy().T).T
        * np.array([[1.0, 1.0, 1.0]])
        + np.array([[0.0, -0.5, 0.0]])
    )
    front_patch.control_points = mi.TensorXf(
        (np.eye(3)[[0, 2, 1]] @ front_patch.control_points.numpy().T).T
        * np.array([[-1.0, 1.0, 1.0]])
        + np.array([[0.0, 0.5, 0.0]])
    )

    bottom_patch.control_points = mi.TensorXf(
        (np.eye(3)[[0, 1, 2]] @ bottom_patch.control_points.numpy().T).T
        * np.array([[-1.0, 1.0, 1.0]])
        + np.array([[0.0, 0.0, -0.5]])
    )
    top_patch.control_points = mi.TensorXf(
        (np.eye(3)[[0, 1, 2]] @ top_patch.control_points.numpy().T).T
        * np.array([[1.0, 1.0, 1.0]])
        + np.array([[0.0, 0.0, 0.5]])
    )

    composite_factory = CompositeSplineSurfaceFactory(manager_name=name)

    composite_factory.add_patch(left_patch)
    composite_factory.add_patch(right_patch)
    composite_factory.add_patch(back_patch)
    composite_factory.add_patch(front_patch)
    composite_factory.add_patch(bottom_patch)
    composite_factory.add_patch(top_patch)

    for second_patch, side in [
        (left_patch, Patch.Side.LEFT),
        (right_patch, Patch.Side.RIGHT),
        (front_patch, Patch.Side.TOP),
        (back_patch, Patch.Side.BOTTOM),
    ]:
        composite_factory.link_sides(
            first_patch_name=top_patch.name,
            first_patch_side=side,
            second_patch_name=second_patch.name,
            second_patch_side=Patch.Side.TOP,
        )

    for second_patch, side in [
        (left_patch, Patch.Side.RIGHT),
        (right_patch, Patch.Side.LEFT),
        (front_patch, Patch.Side.TOP),
        (back_patch, Patch.Side.BOTTOM),
    ]:
        composite_factory.link_sides(
            first_patch_name=bottom_patch.name,
            first_patch_side=side,
            second_patch_name=second_patch.name,
            second_patch_side=Patch.Side.BOTTOM,
        )

    for first_patch, second_patch in [
        (right_patch, front_patch),
        (front_patch, left_patch),
        (left_patch, back_patch),
        (back_patch, right_patch),
    ]:
        composite_factory.link_sides(
            first_patch_name=first_patch.name,
            first_patch_side=Patch.Side.RIGHT,
            second_patch_name=second_patch.name,
            second_patch_side=Patch.Side.LEFT,
        )

    composite_mesh = composite_factory.build()

    return TessellatedSplineSurface(
        composite_spline_surface=composite_mesh,
        tessellation_subdivisions_per_patch=tessellation_subdivisions,
        per_patch_bsdf=[bsdf] * len(composite_mesh),
    )


def _get_normalizing_tf_matrix(points: "mi.TensorXf") -> "mi.Transform4f":
    """
    Computes the translation and scale factors required to transform the input `points`
    so that their bounding box is exactly [-1, 1]^3.

    returns:
        tuple[list[float], list[float]]: A tuple containing the scale and translation
        vectors to apply to the points for normalization.
    """

    if points.ndim != 2:
        raise RuntimeError(f"points ndim = {points.ndim} but it must be 2")

    if points.shape[1] != 3:
        raise RuntimeError(f"points shape[1] = {points.shape[1]} but it must be 3")

    # Find the bounding box of the object
    bbox_min = dr.min(points, axis=0)
    bbox_max = dr.max(points, axis=0)

    # Scale the object such that the bounding box is [-radius, radius]^3
    translation = -(bbox_max + bbox_min) / 2
    translation = list(translation.array)
    scale = 2 / (bbox_max - bbox_min)  # go to [-1, -1]
    scale = list(scale.array)

    return mi.Transform4f().scale(mi.Point3f(scale)).translate(mi.Point3f(translation))


def cubed_sphere(
    name: str,
    degree: int = 3,
    spline_subdivisions: int = 16,
    tessellation_subdivisions: int = 64,
    radius: float = 1.0,
    bsdf: DiffuseBSDF | None = None,
    optimize_g1: bool = True,
) -> TessellatedSplineSurface:
    """
    Creates a cubed sphere mesh with the specified `radius`.

    The cubed sphere is a tessellated version of a unit cube, where each vertex of the
    cube (which is centered at the origin) is scaled by the `radius` parameter.
    The cube is created using the `unit_cube` function and then the control points
    are scaled accordingly to their distance from the origin.

    Note that this will not yield a perfect sphere due to the nature of B-splines. To
    make an exact sphere one would need to use NURBS instead of B-splines, and the
    node placement would need to be adjusted accordingly.

    As a consequence, the constructed sphere does not have smooth normals by default,
    so an additional regularization step is performed to ensure that the normals
    are smooth across the surface. This is done by minimizing the G1 continuity loss
    across the patches of the cubed sphere. Optimization could proceed without this
    step, but this can sometimes cause artifacts at the start of the optimization
    process. This is done only if `optimize_g1` is set to `True`.

    The unit sphere is named `name` and uses `degree` B-splines with
    `spline_subdivisions` knot spans. The resulting mesh is tessellated into a
    composite mesh with `tessellation_subdivisions` subdivisions per patch.

    :returns: A tessellated spline surface representing a cubed sphere with bounding
              box [-radius, -radius, -radius] to [radius, radius, radius].

    :raises ValueError: If `name` is an empty string, if `degree` is less than 1,
        if `spline_subdivisions` is less than 1, if `tessellation_subdivisions` is
        less than 1, or if `radius` is not a positive real number.
    """
    if radius <= 0.0:
        raise ValueError("`radius` must be a positive real number.")
    # unit cube will take care of checking the rest of the parameters
    cube = unit_cube(
        name=name,
        degree=degree,
        spline_subdivisions=spline_subdivisions,
        tessellation_subdivisions=tessellation_subdivisions,
        bsdf=bsdf,
    )

    control_points = cube.global_control_points.numpy()
    vertex_magnitudes = np.linalg.norm(control_points, axis=1)
    # Normalize to unit sphere
    cube.global_control_points = mi.TensorXf(
        control_points / vertex_magnitudes[:, np.newaxis]
    )

    if optimize_g1:
        logger.info("Optimizing G1 continuity of the cubed sphere.")
        regularization = SplineG1Loss(composite_spline=cube, weight=1.0)

        # Learning rate is chosen heuristically to be small enough to avoid
        # oscillations, but large enough to converge in a reasonable time.
        # quadratic scaling was determined to work well empirically.
        adam = mi.ad.Adam(lr=1.5e-2 / spline_subdivisions**2, uniform=True)
        adam["cps"] = cube.global_control_points

        for i in range(151):
            # Update the control points
            cube.global_control_points = adam["cps"]

            loss = regularization.evaluate()
            dr.backward(loss)
            adam.step()

            if i % 10 == 0:
                logger.debug(f"Iteration {i:3d}: G1 loss = {loss.numpy():.4g}")

    transform_to_normalized_bbox = _get_normalizing_tf_matrix(
        cube._tessellated_mesh._global_mesh.vertices
    )
    cube.transform_global_control_points(transform_to_normalized_bbox)

    # Scale the sphere to the desired radius
    cube.global_control_points = cube.global_control_points * radius

    return cube
