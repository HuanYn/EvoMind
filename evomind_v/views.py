"""Deterministic global + 2x2 local image views, before SigLIP preprocessing."""

from PIL import Image, ImageOps

VIEW_VERSION = "global-quadrants-exif-rgb-v1"
VIEW_NAMES = ("global", "top_left", "top_right", "bottom_left", "bottom_right")


def num_views(mode="single"):
    if mode not in {"single", "multi"}:
        raise ValueError(f"mode must be 'single' or 'multi', got {mode!r}")
    return 1 if mode == "single" else 5


def make_image_views(image, mode="single"):
    """Return RGB copies ordered global, TL, TR, BL, BR.

    EXIF orientation is applied first. Odd dimensions split at floor(size/2),
    assigning the extra row/column to the bottom/right. One-pixel dimensions
    reuse that row/column in both halves, never producing empty crops.
    Each view is subsequently resized independently by the official processor.
    """
    count = num_views(mode)
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")
    if image.width < 1 or image.height < 1:
        raise ValueError("image dimensions must be positive")
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    result = [rgb.copy()]
    if count == 1:
        return result
    width, height = rgb.size
    middle_x, middle_y = width // 2, height // 2
    left_end, top_end = max(1, middle_x), max(1, middle_y)
    boxes = (
        (0, 0, left_end, top_end),
        (middle_x, 0, width, top_end),
        (0, middle_y, left_end, height),
        (middle_x, middle_y, width, height),
    )
    result.extend(rgb.crop(box) for box in boxes)
    return result


def preprocess_views(image, processor, mode="single"):
    """Run the supplied official processor and return tensors [views, ...]."""
    views = make_image_views(image, mode)
    output = dict(processor(images=views, return_tensors="pt"))
    if "pixel_values" not in output:
        raise ValueError("image processor must return pixel_values")
    for name, value in output.items():
        if not hasattr(value, "shape") or value.shape[0] != len(views):
            raise ValueError(f"processor output {name!r} has an invalid view axis")
    return output
