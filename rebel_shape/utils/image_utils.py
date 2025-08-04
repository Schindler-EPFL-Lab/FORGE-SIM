from pathlib import Path

import matplotlib.pyplot as plt
import mitsuba as mi
import mlflow


def show_rendered_views(
    views: list[mi.Bitmap],
    save_path: Path | None = None,
    titles: list[str] | None = None,
    image_dpi: int = 120,
) -> None:
    """
    Displays a list of rendered `views` using matplotlib.

    Optionally saves the generated image to `save_path`. Since there might be many
    views, optional `titles` can be assigned to each one of them.

    :raises ValueError: if the `views` list is empty or if the `views` and
        `titles` lists do not have the same length (`titles` list is not empty).
    """
    if len(views) == 0:
        raise ValueError("The 'views' list cannot be empty.")

    if titles and (len(views) != len(titles)):
        raise ValueError(
            "There should be as many elements in 'views' as there are 'titles"
        )

    num_views = len(views)
    fig, axs = plt.subplots(
        nrows=1, ncols=num_views, figsize=(3 * num_views, 3 + 0.5), dpi=image_dpi
    )

    # Ensure axs is iterable if there's only one view
    if num_views == 1:
        axs = [axs]

    for i, (view, ax) in enumerate(zip(views, axs)):
        ax.imshow(view)
        ax.axis("off")
        if titles:
            ax.set_title(titles[i], fontsize=10)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

        mlflow.log_figure(figure=fig, artifact_file=f"logged_images/{save_path.name}")

    plt.show()
    plt.close(fig)


def show_comparison_with_reference(
    rendered_views: list[mi.Bitmap],
    reference_views: list[mi.Bitmap],
    save_path: Path | None = None,
) -> None:
    """
    Displays the rendered views next to the reference ones for visual comparison.

    Optionally saves the generated image to `save_path`.

    :raises ValueError: if the 'rendered_views' or 'reference_views' lists are
        empty or they do not have the same length
    """
    if len(rendered_views) == 0:
        raise ValueError("The 'views' list cannot be empty.")

    if len(rendered_views) != len(reference_views):
        raise ValueError("'views' and 'referene_views' must have the same length.")

    num_views = len(rendered_views)
    fig, axs = plt.subplots(
        nrows=2, ncols=num_views, figsize=(3 * num_views + 1, 2 * 3), dpi=150
    )

    axs[0][0].set_title("Reference", y=0.3, x=-0.1, rotation=90, fontsize=16)
    axs[1][0].set_title("Optimized", y=0.2, x=-0.1, rotation=90, fontsize=16)

    for i in range(num_views):
        axs[0][i].imshow(reference_views[i])
        axs[0][i].axis("off")

        axs[1][i].imshow(rendered_views[i])
        axs[1][i].axis("off")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.show()
    plt.close(fig)

    mlflow.log_figure(
        figure=fig, artifact_file=save_path.name if save_path else "comparison.png"
    )
