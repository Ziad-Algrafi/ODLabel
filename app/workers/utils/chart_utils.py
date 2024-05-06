import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def input_chart(images_folder, chart_colors):
    image_files = [
        f
        for f in os.listdir(images_folder)
        if f.endswith((".jpg", ".jpeg", ".png", "jfif"))
    ]
    resolutions = []
    blurred_images = 0
    grayed_images = 0
    black_white_images = 0
    corrupted_images = 0
    color_spaces = {}

    for image_file in image_files:
        img_path = os.path.join(images_folder, image_file)
        try:
            img = cv.imread(img_path)
            if img is not None:
                height, width = img.shape[:2]
                resolutions.append((width, height))

                # Check for blurred images
                if cv.Laplacian(img, cv.CV_64F).var() < 100:
                    blurred_images += 1

                # Check for grayscale images
                if len(img.shape) < 3 or img.shape[2] == 1:
                    grayed_images += 1

                # Check for black and white images
                if (
                    len(img.shape) == 3
                    and np.all(img[..., 0] == img[..., 1])
                    and np.all(img[..., 1] == img[..., 2])
                ):
                    black_white_images += 1

                # Check color space
                color_space = tuple(img.mean(axis=(0, 1)) / 255.0)
                color_spaces[color_space] = color_spaces.get(color_space, 0) + 1
        except Exception:
            corrupted_images += 1

    # Plot number of images and image format distribution
    fig_format = Figure(
        figsize=(4, 3),
        dpi=100,
        facecolor="#1c1c1c",
        edgecolor="#1c1c1c",
        tight_layout=True,
    )
    ax_format = fig_format.add_subplot(1, 1, 1)
    formats = [os.path.splitext(f)[1].lower() for f in image_files]
    format_counts = {}
    for fmt in formats:
        format_counts[fmt] = format_counts.get(fmt, 0) + 1

    _ = ax_format.bar(
        ["Images"], [len(image_files)], color=chart_colors["images"], width=0.4
    )
    _ = ax_format.bar(
        list(format_counts.keys()),
        list(format_counts.values()),
        color=chart_colors["format"],
        width=0.4,
    )
    ax_format.set_ylabel("Instances", color="white")
    ax_format.set_title("Instances and Format Distribution", color="white")
    ax_format.tick_params(axis="x", colors="white", rotation=0)
    ax_format.tick_params(axis="y", colors="white")
    fig_format.set_facecolor("#1c1c1c")  # dark gray

    # Plot image resolution distribution
    fig_resolution = Figure(
        figsize=(4, 3),
        dpi=100,
        facecolor="#1c1c1c",
        edgecolor="#1c1c1c",
        tight_layout=True,
    )
    ax_resolution = fig_resolution.add_subplot(1, 1, 1)
    widths = [res[0] for res in resolutions]
    min_width = min(widths)
    # max_width = max(widths)
    bins = np.arange(min_width, 4000, 600)

    ax_resolution.hist(widths, bins=bins, color=chart_colors["resolution"])
    ax_resolution.set_xlabel("Resolution", color="white")
    ax_resolution.set_ylabel("Instances", color="white")
    ax_resolution.set_title("Image Resolution Distribution", color="white")
    ax_resolution.tick_params(axis="x", colors="white")
    ax_resolution.tick_params(axis="y", colors="white")
    ax_resolution.ticklabel_format(axis="x", style="plain")
    ax_resolution.set_xticks(bins)
    ax_resolution.set_xticklabels(list(map(str, bins)), rotation=0, ha="center")
    fig_resolution.set_facecolor("#1c1c1c")  # dark gray

    # Plot image quality information
    fig_quality = Figure(
        figsize=(4, 3),
        dpi=100,
        facecolor="#1c1c1c",
        edgecolor="#1c1c1c",
        tight_layout=True,
    )
    ax_quality = fig_quality.add_subplot(1, 1, 1)
    quality_data = [blurred_images, grayed_images, black_white_images, corrupted_images]
    ax_quality.bar(
        ["Blurred", "Grayscale", "B&W", "Corrupted"],
        quality_data,
        color=chart_colors["quality"],
    )
    ax_quality.set_ylabel("Instances", color="white")
    ax_quality.set_title("Image Quality", color="white")
    ax_quality.tick_params(axis="x", colors="white")
    ax_quality.tick_params(axis="y", colors="white", labelleft=True, labelright=False)
    ax_quality.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig_quality.set_facecolor("#1c1c1c")  # dark gray

    # Plot color space distribution
    fig_colorspace = Figure(
        figsize=(4, 3),
        dpi=100,
        facecolor="#1c1c1c",
        edgecolor="#1c1c1c",
        tight_layout=True,
    )
    ax_colorspace = fig_colorspace.add_subplot(1, 1, 1, projection="3d")
    color_spaces_keys = list(color_spaces.keys())
    x = [color[0] for color in color_spaces_keys]
    y = [color[1] for color in color_spaces_keys]
    z = [color[2] for color in color_spaces_keys]
    sizes = [size * 50 for size in list(color_spaces.values())]

    ax_colorspace.scatter(x, y, z, c=color_spaces_keys, s=sizes)

    ax_colorspace.set_xlabel("Red", color="white")
    ax_colorspace.set_ylabel("Green", color="white")
    ax_colorspace.set_zlabel("Blue", color="white")
    ax_colorspace.set_title("Color Space Distribution", color="white")
    ax_colorspace.tick_params(axis="x", colors="white")
    ax_colorspace.tick_params(axis="y", colors="white")
    ax_colorspace.tick_params(axis="z", colors="white")
    ax_colorspace.set_facecolor((0, 0, 0, 0))
    fig_colorspace.set_facecolor("#1c1c1c")  # dark gray

    return fig_format, fig_resolution, fig_quality, fig_colorspace, image_files


def output_charts(detections, num_images):
    object_counts = {}
    confidences = []
    labels_sizes = {}
    bboxes = []
    class_names = []
    heatmap_coords = {}
    labelism_counts = {}

    for _, x, y, w, h, class_name, confidence, img_width, img_height in detections:
        object_counts[class_name] = object_counts.get(class_name, 0) + 1
        confidences.append(confidence)
        labels_sizes[class_name] = labels_sizes.get(class_name, 0) + 1

        # Normalizing bounding box coordinates to [0, 1] range
        normalized_x = x / img_width
        normalized_y = y / img_height
        normalized_w = w / img_width
        normalized_h = h / img_height

        # Adjusting width and x-coordinate to account for the bounding box origin
        normalized_x -= normalized_w / 2
        normalized_w = max(0, min(1 - normalized_x, normalized_w))
        normalized_y -= normalized_h / 2
        normalized_h = max(0, min(1 - normalized_y, normalized_h))

        # Flipping the y-coordinate to match the chart's coordinate system
        normalized_y = 1 - normalized_y - normalized_h

        bboxes.append((normalized_x, normalized_y, normalized_w, normalized_h))

        # Update heatmap and labelism counts
        heatmap_center_x = normalized_x + normalized_w / 2
        heatmap_center_y = normalized_y + normalized_h / 2
        heatmap_coords[(heatmap_center_x, heatmap_center_y)] = (
            heatmap_coords.get((heatmap_center_x, heatmap_center_y), 0) + 1
        )

        labelism_counts[(normalized_x, normalized_y, normalized_w, normalized_h)] = (
            labelism_counts.get(
                (normalized_x, normalized_y, normalized_w, normalized_h), 0
            )
            + 1
        )
        class_names.append(class_name)

    # Count of Detected Objects
    fig_count = Figure(figsize=(4, 3), dpi=100)
    ax_count = fig_count.add_subplot(1, 1, 1)

    if object_counts:
        x = np.arange(len(object_counts) + 1)
        width = 0.4
        ax_count.bar(x[0], num_images, width, color="blue", label="Images")
        ax_count.bar(
            x[1:], list(object_counts.values()), width, color="orange", label="Classes"
        )
        ax_count.set_xticks(x)
        ax_count.set_xticklabels(
            ["Images"] + list(object_counts.keys()), rotation=0, ha="center"
        )
    else:
        ax_count.bar(0, num_images, 0.4, color="blue", label="Images")
        ax_count.set_xticks([0])
        ax_count.set_xticklabels(["Images"], rotation=0, ha="center")

    ax_count.set_ylabel("Instance")
    ax_count.set_title("Instances of Detected Objects by Class", y=1.05)
    ax_count.legend()

    # Detection Confidence Histogram
    fig_conf = Figure(figsize=(4, 3), dpi=100)
    ax_conf = fig_conf.add_subplot(1, 1, 1)
    ax_conf.hist(confidences, bins=20, range=(0, 1))
    ax_conf.set_xlim(0, 1)
    ax_conf.set_xlabel("Confidence")
    ax_conf.set_ylabel("Frequency")
    ax_conf.set_title("Detection Confidence Histogram", y=1.05)

    # Labelism: Bounding Box Chart
    fig_labelism = Figure(figsize=(4, 3), dpi=100)
    ax_labelism = fig_labelism.add_subplot(1, 1, 1)

    max_count = max(labelism_counts.values())
    cmap = plt.get_cmap("Reds")

    for (x, y, w, h), count in labelism_counts.items():
        rect = Rectangle(
            (x, y),
            w,
            h,
            linewidth=0.7,
            edgecolor=cmap(count / max_count),
            facecolor="none",
        )
        ax_labelism.add_patch(rect)

    ax_labelism.set_xlim(0, 1)
    ax_labelism.set_ylim(0, 1)
    ax_labelism.set_title("Labelism: Bounding Boxes", y=1.05)
    ax_labelism.set_xlabel("X")
    ax_labelism.set_ylabel("Y")

    # Heatmap of Detection
    fig_heatmap = Figure(figsize=(4, 3), dpi=100)
    ax_heatmap = fig_heatmap.add_subplot(1, 1, 1)

    max_count = max(heatmap_coords.values())
    cmap = plt.get_cmap("Blues")

    for (x, y), count in heatmap_coords.items():
        color = cmap(count / max_count)
        ax_heatmap.plot(
            x, y, marker="o", markersize=1, color=color, markeredgecolor=color
        )

    ax_heatmap.set_xlim(0, 1)
    ax_heatmap.set_ylim(0, 1)
    ax_heatmap.set_title("Heatmap of Detection")
    ax_heatmap.set_xlabel("X")
    ax_heatmap.set_ylabel("Y")

    return fig_count, fig_conf, fig_labelism, fig_heatmap


def display_chart(fig, frame, row, col):
    if isinstance(fig, tuple):
        fig, _ = fig
    else:
        _ = fig.get_axes()[0]
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().grid(row=row, column=col, sticky="nsew")
    frame.grid_rowconfigure(row, weight=1)
    frame.grid_columnconfigure(col, weight=1)
    return canvas


def close_matplotlib_figures():
    plt.close("all")
