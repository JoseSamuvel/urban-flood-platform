"""
create_sample_data.py - Generates a sample multi-class image dataset for testing
the training, evaluation, and single-image prediction pipeline.
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate_sample_image(category: str, width: int = 256, height: int = 256) -> Image.Image:
    """Generates synthetic sample images with distinct color palettes and patterns."""
    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)

    if category == "healthy_leaf":
        # Base green gradient with organic leaf veins
        base_color = (34, random.randint(130, 180), 34)
        img = Image.new("RGB", (width, height), color=base_color)
        draw = ImageDraw.Draw(img)
        # Draw green veins and contours
        for _ in range(15):
            x1, y1 = random.randint(0, width), random.randint(0, height)
            x2, y2 = x1 + random.randint(-50, 50), y1 + random.randint(-50, 50)
            draw.line([(x1, y1), (x2, y2)], fill=(50, random.randint(190, 240), 50), width=random.randint(2, 4))

    elif category == "rust_disease":
        # Yellowish-green base with orange/reddish powdery rust spots
        base_color = (random.randint(120, 160), random.randint(140, 170), 30)
        img = Image.new("RGB", (width, height), color=base_color)
        draw = ImageDraw.Draw(img)
        # Draw rust spots
        for _ in range(40):
            cx, cy = random.randint(20, width - 20), random.randint(20, height - 20)
            r = random.randint(4, 15)
            rust_color = (random.randint(180, 230), random.randint(60, 100), 20)
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=rust_color)

    elif category == "blight_disease":
        # Pale green base with large dark necrotic lesions/blotches
        base_color = (random.randint(90, 120), random.randint(120, 150), 60)
        img = Image.new("RGB", (width, height), color=base_color)
        draw = ImageDraw.Draw(img)
        # Draw dark necrotic blotches
        for _ in range(8):
            cx, cy = random.randint(30, width - 30), random.randint(30, height - 30)
            rx, ry = random.randint(15, 45), random.randint(15, 45)
            dark_color = (random.randint(40, 70), random.randint(30, 50), random.randint(20, 40))
            draw.ellipse([(cx - rx, cy - ry), (cx + rx, cy + ry)], fill=dark_color)

    # Add subtle Gaussian blur and noise for realistic texture
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
    np_img = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 8, np_img.shape)
    np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)


def create_dataset(
    output_dir: str = "sample_dataset",
    train_count: int = 15,
    val_count: int = 5
):
    categories = ["healthy_leaf", "rust_disease", "blight_disease"]
    print(f"Creating sample dataset in '{output_dir}'...")

    for split, count in [("train", train_count), ("val", val_count)]:
        for cat in categories:
            cat_dir = os.path.join(output_dir, split, cat)
            os.makedirs(cat_dir, exist_ok=True)
            for i in range(count):
                img = generate_sample_image(cat)
                img.save(os.path.join(cat_dir, f"{cat}_{split}_{i+1:03d}.jpg"))

    # Also generate a standalone test image for inference demo
    test_image_path = "test_upload.jpg"
    test_img = generate_sample_image("rust_disease")
    test_img.save(test_image_path)
    print(f"Dataset created with {train_count} train and {val_count} val images per class.")
    print(f"Generated sample inference image: '{test_image_path}'")


if __name__ == "__main__":
    create_dataset()
