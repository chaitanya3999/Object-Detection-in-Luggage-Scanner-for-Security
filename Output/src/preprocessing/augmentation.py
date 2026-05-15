"""
Training augmentation pipelines using albumentations.
Separate pipelines for training and validation.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from typing import Optional


def get_train_augmentation(
    image_size: int = 640,
    rotation_limit: int = 30,
    scale_range: tuple = (0.8, 1.2),
    brightness_limit: float = 0.15,
    contrast_limit: float = 0.15,
    gauss_noise_var_limit: tuple = (10.0, 50.0),
    cutmix_prob: float = 0.0,
) -> A.Compose:
    """
    Get training augmentation pipeline.

    Includes geometric, photometric, and X-ray-specific augmentations.
    Compatible with bounding box annotations (YOLO/Pascal VOC format).

    Args:
        image_size: Target image size.
        rotation_limit: Max rotation degrees.
        scale_range: Min/max scale factors.
        brightness_limit: Max brightness/contrast adjustment.
        contrast_limit: Max contrast adjustment.
        gauss_noise_var_limit: Gaussian noise variance range.
        cutmix_prob: Probability of CutOut augmentation (simulates occlusion).

    Returns:
        albumentations Compose pipeline.
    """
    transforms = [
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.Rotate(
            limit=rotation_limit,
            border_mode=0,  # cv2.BORDER_CONSTANT
            value=0,
            p=0.5,
        ),
        A.RandomScale(
            scale_limit=(scale_range[0] - 1.0, scale_range[1] - 1.0),
            p=0.3,
        ),
        A.ElasticTransform(
            alpha=50,
            sigma=5,
            p=0.1,
        ),

        # Photometric augmentations
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=0.5,
        ),
        A.GaussNoise(
            var_limit=gauss_noise_var_limit,
            p=0.3,
        ),
        A.RandomGamma(
            gamma_limit=(80, 120),
            p=0.3,
        ),

        # X-ray specific: simulate partial occlusion
        A.CoarseDropout(
            max_holes=3,
            max_height=int(image_size * 0.15),
            max_width=int(image_size * 0.15),
            fill_value=128,
            p=cutmix_prob,
        ),

        # Resize to target
        A.Resize(image_size, image_size),
    ]

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",  # [x1, y1, x2, y2]
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def get_val_augmentation(image_size: int = 640) -> A.Compose:
    """
    Get validation/test augmentation pipeline.
    Only resizing, no augmentation.
    """
    return A.Compose(
        [A.Resize(image_size, image_size)],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def get_mosaic_augmentation(
    images: list,
    bboxes_list: list,
    labels_list: list,
    image_size: int = 640,
    seed: Optional[int] = None,
) -> tuple:
    """
    Apply mosaic augmentation: combine 4 images into one.

    This is a powerful augmentation that increases diversity by
    showing multiple images in a single training sample.

    Args:
        images: List of 4 images (np.ndarray).
        bboxes_list: List of 4 bbox arrays.
        labels_list: List of 4 label arrays.
        image_size: Output image size.
        seed: Random seed.

    Returns:
        (mosaic_image, mosaic_bboxes, mosaic_labels)
    """
    rng = np.random.RandomState(seed)

    if len(images) < 4:
        # Repeat to fill 4
        while len(images) < 4:
            images.append(images[0])
            bboxes_list.append(bboxes_list[0])
            labels_list.append(labels_list[0])

    # Random center point
    cx = rng.randint(image_size // 4, 3 * image_size // 4)
    cy = rng.randint(image_size // 4, 3 * image_size // 4)

    mosaic = np.full((image_size, image_size, images[0].shape[2] if len(images[0].shape) == 3 else 1),
                     114, dtype=np.uint8)
    all_bboxes = []
    all_labels = []

    # Place 4 images in quadrants
    placements = [
        (0, 0, cx, cy),            # top-left
        (cx, 0, image_size, cy),   # top-right
        (0, cy, cx, image_size),   # bottom-left
        (cx, cy, image_size, image_size),  # bottom-right
    ]

    for i, (x1, y1, x2, y2) in enumerate(placements):
        img = images[i]
        bboxes = bboxes_list[i]
        labels = labels_list[i]

        rh, rw = y2 - y1, x2 - x1
        h, w = img.shape[:2]

        # Scale image to fit region
        scale = min(rw / w, rh / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h)) if new_w > 0 and new_h > 0 else img

        # Place in mosaic
        paste_x = x1
        paste_y = y1
        ph = min(new_h, rh)
        pw = min(new_w, rw)

        if len(img_resized.shape) == 2:
            img_resized = img_resized[:, :, np.newaxis]
        mosaic[paste_y:paste_y + ph, paste_x:paste_x + pw] = img_resized[:ph, :pw]

        # Adjust bboxes
        if len(bboxes) > 0:
            adjusted = bboxes.copy().astype(float)
            adjusted[:, [0, 2]] = adjusted[:, [0, 2]] * scale + paste_x
            adjusted[:, [1, 3]] = adjusted[:, [1, 3]] * scale + paste_y

            # Clip to mosaic bounds
            adjusted[:, [0, 2]] = np.clip(adjusted[:, [0, 2]], x1, x2)
            adjusted[:, [1, 3]] = np.clip(adjusted[:, [1, 3]], y1, y2)

            # Filter out boxes that became too small
            widths = adjusted[:, 2] - adjusted[:, 0]
            heights = adjusted[:, 3] - adjusted[:, 1]
            valid = (widths > 5) & (heights > 5)

            all_bboxes.append(adjusted[valid])
            all_labels.append(labels[valid] if isinstance(labels, np.ndarray) else np.array(labels)[valid])

    if all_bboxes:
        all_bboxes = np.concatenate(all_bboxes, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
    else:
        all_bboxes = np.zeros((0, 4))
        all_labels = np.array([])

    return mosaic, all_bboxes, all_labels


import cv2  # needed for mosaic
