# Rebel Shape

Rebel shape is a library for spline-based shape and texture optimization as well as simulation based on Isogeometric
Analysis (IGA).
It builds on top of the Mitsuba 3 differentiable renderer and Dr.Jit Automatic Differentiation framework.

Its main capabilities include:

- Simplifies and streamlines the use of Mitsuba 3 for differentiable rendering and optimization tasks.
- Provides building blocks for image based 3D reconstruction pipelines including optimizers, losses, and regularization terms
- Supports both triangle mesh and spline based shape and texture optimization of multi-patch surfaces
- Implements isogeometric analysis (IGA) tools for simulation tasks with the optimized spline surfaces
- Facilitates data processing for both synthetic and real datasets, including compatibility with nerfstudio scenes.
- Supplies scripts for data preprocessing, optimization, simulation, and Azure job management.

## Installation

This project is built using the [uv package manager](https://docs.astral.sh/uv/).

`torch`, `torchvision`, `setuptools` have to be installed before the rest as
[Segment Anything Model 3 (SAM 3)](https://github.com/facebookresearch/sam3)
require it as a dependency.

Running the following commands at the top directory of rebel-shape installs the project:

```bash
uv venv &&\
uv pip install torch torchvision setuptools && \
uv sync --all-extras
```

### Quick-start

Go to `rebel_shape/scripts` directory and run `optimize_suzanne_mesh.py` or `optimize_suzanne_spline.py` scripts.
These scripts can be run locally on the CPU, but the default setting are tuned for GPU execution on Azure, so adjusting the parameters is necessary.
Reducing the number of samples per pixel to 4-9 can yield reasonable execution time without compromising the optimization procedure.
The fact that the images are quite noisy does not really impact the optimization until fine details have to be resolved.
For executing a full optimization procedure, you will need patience or a faster computer :)

To get familiar with rendering, it is recommended to study Mitsuba 3 and its' nice [tutorials](https://mitsuba.readthedocs.io/en/stable/src/quickstart/mitsuba_quickstart.html)
They teach everything from making a small renderer to performing inverse design, all the way to voxel-based volumetric novel view synthesis.

For an introduction to splines, there are many good resources online, but it can also be useful to go into the tests and call the `visualize` method for curves and surfaces.
Knot vectors can be changed and control points moved around to see how the resulting shape changes.

## Project Structure

The project is structured as follows:

```
rebel_shape/
├── rebel_shape/            # Main package
│   ├── plugin_wrappers/        # Wrappers around select Mitsuba plugins that are used in rebel_shape
│   ├── scene_optimizers/       # Optimizers 
│   ├── loss_evaluators/        # Image loss functions that are the primary optimization targets
│   ├── regularization/         # Regularization penalty terms
│   ├── schedulers/             # Learning rate schedulers
│   ├── splines/                # Spline interfaces and a differentiable implementation of B-splines
│   ├── meshing/                # Composite mesh and spline building and handling functionality
│   ├── property_optimizers/    # High-level specification of the optimization problem
│   ├── data_processing/        # Functionality for preparing the data for optimization
│   ├── isogeometric_analysis/  # Simulation functionality based on IGA
│   ├── utils/                  # Miscallaneous utility functions
│   ├── scripts/                # Scripts for running shape optimization and simulation
│       ├── segmentation/           # Scripts for image segmentation
├── azure_scripts/          # Scripts to launch Azure jobs (execute code from rebel_shape/scripts)
├── tests/                  # Unit tests
```

### Plugin Wrappers

[Mitsuba 3](https://mitsuba.readthedocs.io/en/stable/) is a modular differentiable renderer and as such it is based on dictionaries to specify the scene.
Dictionaries make the code and all the available options more difficult to understand, so a basic interface that represents a `MitsubaPlugin` is defined.  

The interface prescribes a single method that has to be implemented:  `mitsuba_dict`.
It returns the dictionary representation that can be directly inputted into a Mitsuba scene, which is also specified by a dictionary.
The advantage of this approach is that all configuration that is usually based on strings can be made explicit and documented, without consulting the [Mitsuba docs](https://mitsuba.readthedocs.io/en/stable/src/plugin_reference.html) (which are very good!)

### Scene Optimizers

Rendering workloads exhibit properties that are not suitable for conventional Automatic Differentiation (AD) frameworks like PyTorch and Jax.
So, Mitsuba 3 uses a specialized Just-In Time (JIT) compiler [Dr.Jit](https://rgl.epfl.ch/publications/Jakob2022DrJit).
It is somewhat more delicate to use than similar frameworks as it is lower level, so its functionality is wrapped in higher level optimizer objects.
Mitsuba 3 already provides low-level wrappers for SGD and Adam that are flexible, but lack nice-to-haves like batching.

Another basic optimizer kind, L-BFGS, is also implemented.
[L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) is not typically used in shape and texture optimization, but in the context of splines it was found to offer comparable convergence properties while making the optimization process much more stable.
It also obviates the need for guessing a good learning rate, and benefits from more samples per pixel as it allows a more accurate curvature estimate.

To facilitate simpler scene optimization and correct scene property updates, the `SceneOptimizer` interface is created and aims to add a layer of specialization on top of the basic optimizers.
It is assumed that the loss is based on a combination on image loss and regularization.
Additionally, batching support and learning rate scheduling are added, as well as scene updates based on the provided callback.

Composite optimization is also supported, where different parameters are assigned to an optimizer with different properties (e.g. batch size).
This is particularly useful for spline and texture optimization where the shape optimization is best handled with L-BFGS, while Adam with a batch size of 1 performs the best in texture optimization

### Loss Evaluators

The optimization problem that this library aims to solve is based on matching the rendered representation of the object with the input images.
As such, the fundamental loss is an image based one and the following losses are supported:

- L2 loss: The mean squared pixel-wise error
- L1 loss: The mean average pixel-wise error (commonly referred to as MAE)
- SSIM loss: Patched based loss that compares the similarity of the distributions
- MS-SSIM loss: patch based perception inspired loss that extends SSIM concepts to multiple resolutions
- Composite loss: A combination of any number of the aforementioned image losses

A good default is simply choosing the L1 loss; the L2 loss tends to produce overly smooth results that lack detail.
The combination of the two can be useful in some cases.
Best results are typically obtained by combining the L1 loss with MS-SSIM  loss with weights 0.8 and 0.2, respectivey.

### Regularization

The space of possible solutions to an optimization problem can be constrained in three ways:

1. **Hard constraints**: projecting the real number line into a range of admissible values (determined a priori). \
Example: Inequality constraints
2. **Pre-conditioning**: optimization is carrier out in a latent space instead of directly optimizing the parameters, providing an opportunity to change how the gradients propagate. \
Example: gradient pre-conditioning that smoothens out strong gradients and propagates them to neighboring vertices.
3. **Soft constraints (penalty terms)**: another term is added to the optimization problem loss and it biases the gradients, helping to avoid bad configurations. \
Example: Laplacian penalty term that penalizes high curvature.

The first type is typically a strong physical constraint, e.g. the color of an object is a triplet of floating point values in range `[0.0, 1.0]`.
Values outside of this range would not make physical sense as they would imply that the outgoing light, reflecting from a surface, becomes more intense than it was when it arrived.
This kind of a constraint can be enforced at the solver level or by changing the optimization space from the real numbers to a finite interval.

The second type is effectively a re-parameterization that leaves the original problem unchanged but still implicitly constrains how gradient updates could be carried out.
This approach is typically a product of aspiring to build Hessian information into the optimization problem to increase the robustness of the optimization procedure.
This is exactly what the [Large Steps paper](https://rgl.epfl.ch/publications/Nicolet2021Large) proposes to stabilize the optimization of triangular meshes.

The third type is very widely used for a range of applications, from finite element analysis and inverse design to data science and machine learning.
It is a soft penalty on the magnitude of some value that constrains the space of admissible configurations to avoid getting stuck in bad local minima.
For shape optimization this mostly reduces to Laplacian regularization terms or other types of curvature penalty.
Another example is are multi-patch spline surfaces that require an additional normal continuity loss to ensure visual smoothness at the edge where pairs of patches meet.

### Schedulers

As the optimization procedure relies mostly on stochastic optimizers, adjusting the learning rate as optimization progresses often leads to better results.
The typical pattern is to reduce the learning rate towards the end of the optimization to avoid oscillation and converge to the minimum.
However, sometimes the opposite is true, and increasing the learning rate towards the end can yield better results.

An example of this is texture optimization while both the shape and texture are being optimized;
Large texture learning rate on the initial shape can lead to getting stuck in local minima or reduce the detail in the geometry.
This is an unfortunate consequence of shape-radiance (texture) ambiguity.

### B-Splines

B-splines are the fundamental primitive of the rebel-shape library, used for both optimization and simulation.

Unlike triangular meshes, spline surfaces are smooth, assuming the degree of the spline is greater than 2.
Higher smoothness comes from basis functions are more spread out in space, as well as more basis functions influencing the value at a particular point.
Increasing the degree results in higher continuity and the representation exhibits less variation overall (visually smoother).
Combined with the fact that a small number of control points can capture complex geometry makes splines a good candidate for shape optimization.

Basic interfaces that are needed for shape optimization and isogeometric analysis are provided.
The B-spline surface is the main implementor of all this functionality.
For details on how B-splines are defined and their properties, it is advised to consult the documentation of classes in this directory or consult the [The NURBS Book](https://home.zcu.cz/~bastl/GM1/the-nurbs-book.pdf) for an exhaustive treatment.

### Composite Spline Surfaces and Meshes

The primary responsibility of this module is managing multiple rectangular patches that are either represented by splines or patches.
Furthermore, as splines cannot be optimized directly, nor can Mitsuba directly represent B-spline surfaces, a mesh representation of spline surfaces is required for shape optimization.
Thus, when carrying out shape optimization with splines two representations have to coexist: a spline-based representation and a tessellated (triangulated) representation of the spline.

Since the goal is to optimize closed surfaces, more than one spline patch is needed;
The initial shape is sphere-like and has 6 patches; it is inspired by a cubed sphere.
The initial shape can be chosen arbitrarily and it is should matches the [genus](https://en.wikipedia.org/wiki/Genus_(mathematics)) of the target object to obtain a high-quality reconstruction.

The multi-patch representation requires keeping track of the global control points, control points assigned to each patch, as well as the
topology (connectivity) information of the patches.
Furthermore, each patch has an associated mesh and all meshes form a global mesh.

A 2-level hierarchy emerges for each of the 2 representations:

- Composite Spline surface
  - Spline patches: each patch has its own parametrization, control points, and quads.
  - Global representation: global control points and quads, cannot be queried directly.
- Composite Mesh
  - Mesh patches: each patch has its own vertices, faces, normals, and texture coordinates.
  - Global mesh: the collection of all unique vertices and averaged normals as well as global faces, but no global texture coordinates.

Shape optimization requires consistently keeping all 4 components in sync and this is the responsibility of the `TessellatedSplineSurface` class.
Spline representations can be exported using [jsonpickle](https://jsonpickle.github.io/), but the Mitsuba Mesh representation cannot be preserved.
Associated textures are saved separately and have to be set manually after loading the composite spline representation.

### Property Optimizers

While scene optimizers focus on exposing a more streamlined interface for optimizing an arbitrary scene property, the `PropertyOptimizer` interface embodies the logic needed to optimize a specific property.
For example, optimizing a triangle mesh requires updating vertex positions which is inherently ill-conditioned so gradient pre-conditioning needs to be applied
This is the kind of functionality handled by the implementors of the interface.

The interface exposes a small number of methods that have to be implemented and this allows a common optimization loop to be executed for different properties.
If the optimization procedure has some particularities, the `optimize` method can easily be overridden.

In general, the optimization proceeds for a predefined number of steps with each step Executing the following:

1. Set learning rate for the iteration (epoch).
2. Update the scene using all training images, typically in multiple batches, with the regularization loss applied after each batch (the job of the `SceneOptimizer`).
3. Render evaluation views and save the resulting images to visually track the progress of optimization.
4. Process special events, e.g. increasing the number of spline basis functions to increase the resolution of
       the reconstruction
5. Save intermediate optimization results to avoid losing the data if optimization is interrupted for some reason.

Multiple `PropertyOptimizers` can easily be composed and the combined properties optimized using the same interface.
In this case the update order for the parameters is determined by the `SceneOptimizer` that is attached to the main optimization loop.

A script to render camera poses for visualization and debugging purposes is provided in `rebel_shape/scripts`.

### Data Processing

When using synthetic datasets, e.g. Suzanne that is available as an explicit mesh, it is possible to manually render the reference views and then perform a reconstruction using known camera poses.

However, when using a real world dataset camera poses are typically not available.
This library does not support this step, but it can take as input [nerfstudio](https://docs.nerf.studio/) compatible scenes.
It provides the functionality to load this data and applies the correct pose transformations to be compatible with the [Mitsuba sensor convention](https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_sensors.html).

### Isogeometric Analysis

[Isogeometric Analysis (IGA)](https://www.sciencedirect.com/science/article/abs/pii/S0045782504005171) is a Galerkin method that uses spline basis functions to construct the solution while also assuming that these basis functions are used to represent the geometry.
Note that [Finite Element Analysis (FEM)](https://en.wikipedia.org/wiki/Finite_element_method) also typically uses the Galerkin method and a set of basis functions to represent the solution, but the geometry and the solution representation are separated.

In FEM, the geometry is defined by a 2D or 3D mesh that consists of finite elements that can have various shapes (e.g. tetrahedral, hexahedral, prismatic in 3D) and nodes at predefined points.
Lagrange polynomials are then used to represent the solution that is piecewise continuous from element to element, even for higher degrees of basis functions.

In contrast, the solution obtained with IGA is typically $p - 1$ continuous, with $p$ being the degree of the spline basis functions, which allows for a more compact representation without reducing accuracy.
Asymptotically, FEM uses $O((np)^d)$ memory, with $n$ being the number of elements per dimension and $d$ the number of dimensions, while IGA uses only $O((n+p)^d)$ memory.
The advantate becomes clear at high spline orders.

The Galerkin discretization method yields a system of equations that have to be solved either only once or for multiple time steps.
Fundamental building blocks for this process are provided, as well as basic building blocks for simulations on manifolds embedded in 3D space.
The Poisson problem is meant to serve as an example of how to write an IGA assembly code, while the `HeatSimulation` is the more comprehensive IGA implementation of the heat transport problem.

### Scripts

There are two flavors of scripts: Azure scripts and "training" scripts.
Since this library does not really train any neural networks, and sometimes the scripts do not even perform optimization, they will be referred to as simply scripts.

Scripts are used to perform shape optimization, preprocess data, visualize data, apply initial conditions to the optimized shape, and carry out IGA simulation.
They are the consumer of all functionality described above.
Since this library supports handling both meshes and splines, there are distinct scripts for carrying out shape optimization based on triangle meshes and composite splines.
These scripts can be executed locally or on Azure.

These scripts can also be executed on Azure using the Azure scripts.
Azure scripts are formally not part of the library and are thus situated outside, at the same level as unit tests.
Each Azure script should have the same name as the script that it launches.

## Pipeline

### Preparing The Data

The first step is preparing the data and making sure that its format is compatible with the provided data loader for the [ThermoScenes dataset](https://zenodo.org/records/10835108).
In case the data is not directly compatible with this format it is recommended to introduce a conversion script.
For a practical example, see the [`buildnet_to_thermoscenes.py`](./rebel_shape/scripts/buildnet_to_thermoscenes.py) script that converts the [BuildNet3D dataset](https://zenodo.org/records/15075790) to a ThermoScenes compatible dataset.

The baseline ThermoScenes datasets are not yet ready for optimization - the object that should be reconstructed needs to be masked.
We use [Segment Anything Model 3 (SAM 3)](https://github.com/facebookresearch/sam3) to generate precise masks using textual prompts.

This is done in the [`mask_with_sam3.py`](./rebel_shape/scripts/segmentation/mask_with_sam3.py) script.
At the time of writing, this part is expected to be done locally and there is no Azure script.

In the case of the BuildNet3D dataset, the semantic masks are used to directly obtain perfectly segmented images, sidestepping the aforementioned segmentation procedure.

### Shape and Texture Optimization

The quality of the final representation depends heavily on the initial pose and radius of the shape. Therefore, before running any optimization script, it is necessary to determine these parameters by following the [Manual Initialization Guide](./docs/manual_initialization/README.md).

There are 2 classes of optimization scripts:

1. Scripts that assume the input is an explicit mesh that is then optimized (scripts with suzanne in their name)
2. Scripts that receive a ThermoScenes compatible input dataset and optimize the provided images

The scripts from the first category were initially introduced for validating the implementation and assessing the effectiveness of various loss and regularization terms.
They can be used for synthetic datasets that provide a single explicit mesh that has to be reconstructed.

Scripts are further divided into mesh-based and spline-based shape (and texture) optimization.
The optimization procedure is very similar for both as spline-based optimization uses a mesh for representation in the Mitsuba scene.
On a high level, the difference lies in the fact that mesh optimization uses gradient preconditioning (as introduced in [Large Steps](https://rgl.epfl.ch/publications/Nicolet2021Large)) while spline optimization uses a coarse-to-fine scheme.

The output of mesh-based optimization scripts is a single mesh that was obtained by deforming the initial mesh and an appropriate texture.
It is also possible to use a composite mesh made up of 6 patches that each have an assigned texture.
The latter is very similar to the output of spline-based optimization.

Spline-based optimization returns a .json file that is a [jsonpickle](https://jsonpickle.github.io/) representation of `TessellatedSplineSurface` and 6 textures, one for each spline patch.
The tessellated spline surface typically contains a compact representation and a mesh representation can be obtained from it.

The optimized mesh/spline is then provided as an input to inpainting script.
It can either be stored as a data asset on Azure or put into the [optimized_scenes](data/optimized_scenes) folder.
The latter is a good solution for validating scripts and debugging purposes while the former is a more long term solution.

### Inpainting

After the shape has been reconstructed from RGB images, the initial condition for thermal simulation are obtained from thermal images.
This procedure is referred to as thermal inpainting.
Since simulation is carried out with vertices (in case of FEM) and control points (in case of IGA), thermal information has to be directly projected onto vertices/control points.

The first script takes as input just the spline surface obtained via thermal inpainting, while the second one also requires 6 masked facades.
The facades are masked the same way as the ThermoScenes input data; it is accomplished by calling the [`segment_facade.py`](rebel_shape/scripts/segmentation/segment_facade.py) script with the path to the facade images.

The numerical integration scheme is stable for all time step sizes [(Crank-Nicolson scheme)](https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method) and perfectly preserves the total energy if there are no sources or sinks.

If non-linear terms (e.g. black-body radiation) are introduced, an iterative solve has to be performed at each step to ensure stability and accuracy.
This can be done either via the [Newton–Raphson method](https://en.wikipedia.org/wiki/Newton%27s_method) or with a simple [fixed-point iteration](https://en.wikipedia.org/wiki/Fixed-point_iteration) scheme.

Mitsuba supports vertex-based textures while for the spline-based representation the texture is generated by evaluating the spline basis function at each pixel.
This difference again necessitates two different inpainting scripts.

Since the stanford PLY format is quite flexible it is possible to store the per vertex temperature data directly in the PLY format.
For splines, the output is a serialized `TessellatedIGASplineSurface` that contains the per control point thermal information and can be directly used for heat simulation.

Additionally, if other information, like material properties, need to be assigned to the shape, the same procedure can be applied.
A concrete example is the two-material heat simulation that requires assigning one of two materials to each control point.
Windows are segmented using the same procedure as for the RGB images and then the obtained masks are projected onto control points.

### Thermal Simulation (IGA)

Thermal simulation is currently only supported for splines.
The simulation is done using the isogeometric analysis (IGA) framework.

Two heat simulation options are available at the time of writing:

1. Heat simulation with a single material and no heat sources [(here)](rebel_shape/scripts/heat_simulation/run_heat_simulation.py)
2. Heat simulation with two materials and no heat sources [(here)](rebel_shape/scripts/heat_simulation/run_heat_simulation_two_materials.py)

The first script takes as input just the spline surface obtained via thermal inpainting, while the second one also requires 6 masked facades.
The facades are masked the same way as the ThermoScenes input data; it is accomplished by calling the [`segment_facade.py`](rebel_shape/scripts/segmentation/segment_facade.py) script with the path to the facade images.

The numerical integration scheme is stable for all time step sizes [(Crank-Nicolson scheme)](https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method) and perfectly preserves the total energy if there are no sources or sinks.

If non-linear terms (e.g. black-body radiation) are introduced, an iterative solve has to be performed at each step to ensure stability and accuracy.
This can be done either via the [Newton-Raphson method](https://en.wikipedia.org/wiki/Newton%27s_method) or with a simple [fixed-point iteration](https://en.wikipedia.org/wiki/Fixed-point_iteration) scheme.
