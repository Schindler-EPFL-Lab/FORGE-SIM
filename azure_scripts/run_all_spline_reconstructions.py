import subprocess


def main() -> None:
    """
    Run all spline reconstructions for the specified datasets.

    This script iterates over a predefined list of datasets, each with a specific JSON
    file and version number. For each dataset, it executes a subprocess to run the
    spline reconstruction script with the appropriate parameters.

    The datasets include various scenes such as realistic buildings, cars, woodsheds,
    and heating appliances. Each dataset is processed with its corresponding bounding
    box transforms and environment map.
    """
    datasets = {
        "extracted-buildnet3d-realistic-1": (
            "data/bbox_transforms/buildnet3d-realistic-1.json",
            2,
        ),
        "extracted-buildnet3d-realistic-2": (
            "data/bbox_transforms/buildnet3d-realistic-2.json",
            2,
        ),
        "extracted-buildnet3d-realistic-3": (
            "data/bbox_transforms/buildnet3d-realistic-3.json",
            2,
        ),
        "extracted-buildnet3d-realistic-4": (
            "data/bbox_transforms/buildnet3d-realistic-4.json",
            2,
        ),
        "extracted-car": ("data/bbox_transforms/car.json", 3),
        "extracted-woodshed": ("data/bbox_transforms/woodshed.json", 1),
        "extracted-buildingA-spring": ("data/bbox_transforms/buildingA-spring.json", 5),
        "extracted_heater_water_kettle": (
            "data/bbox_transforms/heater_water_kettle_v2.json",
            2,
        ),
        "extracted_heater_water_cup": (
            "data/bbox_transforms/heater_water_cup_v1.json",
            1,
        ),
    }

    for dataset, (json_file, version) in datasets.items():
        command = [
            "python",
            "azure_scripts/optimize_thermoscenes_spline_manual_initial_pose.py",
            "--script-params.thermoscenes-scene-path",
            dataset,
            "--scene-version",
            str(version),
            "--script-params.bbox-transforms-path",
            json_file,
            "--script-params.environment-map-path",
            "data/textures/overcast_soil_2_2k.exr",
        ]
        process = subprocess.run(command, check=True)
        # Check the return code to ensure it completed successfully
        if process.returncode == 0:
            print("Subprocess finished successfully.")
        else:
            print("Subprocess encountered an error.")


if __name__ == "__main__":
    main()
