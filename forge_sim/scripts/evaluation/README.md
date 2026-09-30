# Evaluation

This file describes how to evaluate shape reconstruction and novel view synthesis quality on BuildNet3D, Synthetic Meshes, and ThermoScenes datasets.

Evaluation is two-step for every dataset: first render the reconstruction from the validation poses, then score those renders against the eval images. The rendering step is shared, so it is described once under [BuildNet3D](#buildnet3d) and referenced afterwards.

**Pose convention.** Scripts that read a scene take `--from-vggt` / `--no-from-vggt`, selecting which loader reads the poses. It defaults to `--no-from-vggt` (the ThermoScenes convention). Pass `--from-vggt` only for scenes whose poses came from VGGT; the wrong choice loads different poses and fails silently rather than with an error.

## Spline

### Synthetic Meshes

Run the script:

```bash
python3 forge_sim/scripts/evaluation/eval_ply.py \
    --reference-shape-path /path/to/original/shape \
    --spline-path /path/to/optimized/spline.json \
    --output-file-path /where/to/store/metrics/result.json
```

The script calculates Chamfer distance between the optimized spline and the original synthetic mesh and stores the result at `/where/to/store/metrics/result.json`.

### BuildNet3D

1. Render optimized spline from validation poses images using the script:

    ```bash
    python3 forge_sim/scripts/render_eval_images.py \
        --scene-path /path/to/scene/used/during/training \
        --environment-map-path /path/to/environment/path/used/during/training \
        --spline-path /path/to/optimized/spline.json \
        --textures-paths /path/to/texture_sides/texture_1.exr /path/to/texture_sides/texture_2.exr ... \
        --output-folder /where/to/store/the/result \
    ```

    In our case the optimized spline has 6 sides (it is a cubedsphere initially), therefore one should provide texture paths for all 6 sides.

    Instead of naming the spline and its textures, `--result-folder` can point at an optimization output folder and let the script find them: it takes the highest-iteration `<name>_<timestamp>_iteration-<N>.json` as the checkpoint and ignores any other JSON file. It is ignored if `--spline-path` or `--textures-paths` is given.

    The mesh reconstructions use `render_eval_images_mesh.py`, which takes `--result-path` in place of the spline and texture arguments.

2. Run the script:

    ```bash
    python3 forge_sim/scripts/evaluation/eval_buildnet3d.py \
        --thermoscenes-scene-path /path/to/scene/used/during/training \
        --environment-map-path /path/to/environment/path/used/during/training \
        --rendered-images-folder /path/to/rendered/images/folder/ \
        --spline-path /path/to/optimized/spline.json \
        --reference-shape-path /path/to/ground/truth/buildnet3d/building \
        --output-folder /where/to/store/metrics/and/intermediate/renderings/
    ```

    Note this script spells the scene argument `--thermoscenes-scene-path`, unlike the others.

    The script calculates Chamfer distance between the optimized spline and the original buildnet3d mesh as well as the novel view synthesis metrics (PSNR, SSIM, LPIPS).
    The script also saved two images containing all predicted and all ground truth rgb images.
    The images are helpful to observe the reconstruction quality and to debug metrics if necessary.
    The script saves the computed metrics and images at `/where/to/store/metrics/and/intermediate/renderings/`.

    For Chamfer alone, `eval_spline_chamfer.py` measures against the render-scale ground-truth mesh with a deterministic camera-derived transform rather than a fitted one, and `stitch_and_eval_cubed_sphere_mesh.py` is its counterpart for a cubed-sphere mesh reconstruction, stitching the six patches first.

### Real Scenes

1. Render optimized spline from validation poses following the instructions from [BuildNet3D](#buildnet3d).
2. Run the script:

    ```bash
    python3 forge_sim/scripts/evaluation/eval_real.py \
        --scene-path /path/to/scene/used/during/training \
        --environment-map-path /path/to/environment/path/used/during/training \
        --rendered-images-folder /path/to/rendered/images/folder/ \
        --output-folder /where/to/store/metrics/
    ```

    The script calculates the novel view synthesis metrics (PSNR, SSIM, LPIPS).
    The script also saves two images containing all predicted and all ground truth eval images.
    The images are helpful to observe the reconstruction quality and to debug metrics if necessary.
    The script saves the computed metrics and the images at `/where/to/store/metrics/`.

    `--ignore-background` excludes pixels that are white in the eval images from the comparison. Real-world eval images have occluding elements removed, so without it a method that fails to reconstruct a missing part can score higher than one that does not.

## Mesh

1. Create a folder where to store data required for evaluation, let's say it's EVAL_ROOT. Every driver below takes it as `--validation-inputs-root`.
1. Download the "Baseline-Large-Steps" folder and place it in the EVAL_ROOT with name "large-steps"
2. Download the "BuildNet3DCorrectedMeshes" folder and place it in the EVAL_ROOT with name "BuildNet3DCorrectedMeshes"
3. Place the scenes in EVAL_ROOT/data, one folder per scene.
4. Run the scripts:

    ```bash
    # evaluation on PLY meshes
    python3 main_scripts/evaluation/eval_all_ply_large_steps.py \
        --scenes-names ARMADILLO BUNNY SPOT SUZANNE
    
    # evaluation on buildnet3d scenes
    python3 main_scripts/evaluation/eval_all_buildnet3d_large_steps.py \
        --scenes-ids SCENE_1 SCENE_2 SCENE_3 SCENE_4 \
        --environment-map-path ./data/textures/checkerboard_lighter.exr

    # evaluation on real scenes
    python3 main_scripts/evaluation/eval_all_real_large_steps.py \
        --scenes-names BUILDINGA_SPRING CAR WOODSHED_SUN HEATER_WATER_KETTLE
    ```

    It evaluates the reconstruction on PLY meshes, buildnet3d scenes and real scenes.

    These three drivers score the Large Steps baseline. Each has a counterpart without the `_large_steps` suffix (`eval_all_ply.py`, `eval_all_buildnet3d.py`, `eval_all_real.py`, also under `main_scripts/evaluation/`) taking the same arguments and scoring our own reconstructions. They spawn the single-target scripts on this page, then [`aggregate_metrics.py`](#aggregating-metrics) once per scene to write that scene's metrics file. Every computation, the aggregation included, stays in `forge_sim/scripts/evaluation/`: the library provides the evaluation, and `main_scripts/` owns the list of targets, sweeps it, and does no arithmetic on what comes back.

    A driver writes into the folder named by `--output-folder-name`, and refuses to start if that folder already holds the metrics files it is about to write. Give each run a fresh name rather than clearing the old folder by hand, so a partial previous run cannot be mistaken for a complete one.

## Aggregating metrics

`aggregate_metrics.py` averages the metrics of several evaluated reconstructions into one file:

```bash
python3 forge_sim/scripts/evaluation/aggregate_metrics.py \
    --metrics-folders /metrics/of/reconstruction_0 /metrics/of/reconstruction_1 ... \
    --metrics-file-names eval_nvs.json eval_shape.json \
    --output-file-path /where/to/store/the/mean.json
```

Each folder in `--metrics-folders` is the output of one of the scripts above for one reconstruction. The files named by `--metrics-file-names` are read from every folder and merged, then each metric is averaged over the folders. With a single folder the metrics are written through unchanged, which is how the `_large_steps` drivers record their one reconstruction per scene.

The script refuses rather than write a mean that would mislead: an empty folder list, a folder missing one of the files, two files reporting the same metric, or folders that do not all report the same metrics.

`SplineTexturePaths` (`spline_texture_paths.py`) finds the six cube-face textures of every spline reconstruction in a folder, paired by sort order; the drivers use it to build the `--textures-paths` of `render_eval_images.py`.
