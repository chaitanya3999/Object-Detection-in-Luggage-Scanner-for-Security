import cv2
import numpy as np
from typing import Dict, Any, Tuple

class BeerLambertTIPProjector:
    """
    Simulates Threat Image Projection (TIP) based on the Beer-Lambert Law of X-Ray attenuation:
    I_final = I_bg * (I_fg / 255)
    
    Supports dynamic scaling, rotation, transparency/thickness adjustments, and translation.
    """
    def __init__(self):
        pass

    def project_threat(
        self,
        bg_image: np.ndarray,
        fg_image: np.ndarray,
        scale: float = 1.0,
        angle_deg: float = 0.0,
        pos_x_pct: float = 50.0,
        pos_y_pct: float = 50.0,
        thickness: float = 1.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Overlays the threat image fg_image onto bg_image using strict alpha masking.
        
        Pipeline:
          1. Threshold → binary mask isolating object from white/black bg
          2. Crop to object bounding rect
          3. Resize threat + mask (mask uses INTER_NEAREST to stay binary)
          4. Rotate threat + mask (mask padded with 0, re-thresholded after)
          5. Beer-Lambert blend ONLY where mask > 0
        """
        bg_h, bg_w = bg_image.shape[:2]

        # ── Step 1: Create binary mask BEFORE any transforms ──────────
        fg_gray = cv2.cvtColor(fg_image, cv2.COLOR_BGR2GRAY)

        # Detect background color from the 4 corners
        corners = [int(fg_gray[0, 0]), int(fg_gray[0, -1]),
                    int(fg_gray[-1, 0]), int(fg_gray[-1, -1])]
        avg_corner = np.mean(corners)

        if avg_corner > 127:
            # White background → object is darker
            _, mask = cv2.threshold(fg_gray, 230, 255, cv2.THRESH_BINARY_INV)
        else:
            # Black background → object is brighter
            _, mask = cv2.threshold(fg_gray, 25, 255, cv2.THRESH_BINARY)

        # Morphological cleanup: erode to kill edge fuzz, dilate to restore shape
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # ── Step 2: Crop to tight bounding rect of the object ─────────
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Use the union bounding rect of ALL contours, not just the largest
            all_points = np.vstack(contours)
            tx, ty, tw, th = cv2.boundingRect(all_points)
            cropped_fg = fg_image[ty:ty+th, tx:tx+tw]
            cropped_mask = mask[ty:ty+th, tx:tx+tw]
        else:
            cropped_fg = fg_image
            cropped_mask = mask

        # ── Step 3: Resize threat + mask ──────────────────────────────
        orig_h, orig_w = cropped_fg.shape[:2]
        target_w = max(10, int(bg_w * 0.25 * scale))
        target_h = max(10, int(orig_h * (target_w / max(orig_w, 1))))

        resized_fg = cv2.resize(cropped_fg, (target_w, target_h), interpolation=cv2.INTER_AREA)
        # INTER_NEAREST keeps the mask binary — no grey interpolation artifacts
        resized_mask = cv2.resize(cropped_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        # Re-threshold to guarantee binary after resize
        _, resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)

        # ── Step 4: Rotate threat + mask ──────────────────────────────
        if angle_deg != 0:
            # Rotate the threat image — pad with neutral gray (won't matter, mask gates it)
            resized_fg = self._rotate_image(resized_fg, angle_deg, bg_fill=(128, 128, 128))
            # Rotate the mask — pad with BLACK (0) so padding is never "active"
            resized_mask = self._rotate_image(resized_mask, angle_deg, bg_fill=0)
            # Re-threshold after rotation to kill any interpolation bleed
            _, resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)

        tf_h, tf_w = resized_fg.shape[:2]

        # ── Step 5: Compute placement coordinates ─────────────────────
        center_x = int(bg_w * (pos_x_pct / 100.0))
        center_y = int(bg_h * (pos_y_pct / 100.0))

        x1 = center_x - tf_w // 2
        y1 = center_y - tf_h // 2
        x2 = x1 + tf_w
        y2 = y1 + tf_h

        # Clip to background bounds
        dx1, dy1 = max(0, x1), max(0, y1)
        dx2, dy2 = min(bg_w, x2), min(bg_h, y2)

        sx1, sy1 = dx1 - x1, dy1 - y1
        sx2, sy2 = sx1 + (dx2 - dx1), sy1 + (dy2 - dy1)

        # ── Step 6: Beer-Lambert blend, strictly gated by mask ────────
        projected = bg_image.copy().astype(np.float64)

        if (dx2 > dx1) and (dy2 > dy1):
            local_bg = projected[dy1:dy2, dx1:dx2]
            local_fg = resized_fg[sy1:sy2, sx1:sx2].astype(np.float64)
            local_mask = resized_mask[sy1:sy2, sx1:sx2]

            # Build a soft alpha from the binary mask (0.0 or 1.0)
            alpha = (local_mask.astype(np.float64) / 255.0)[:, :, np.newaxis]  # (H, W, 1)

            # Beer-Lambert attenuation with thickness modifier:
            #   I_attenuated = 255 - (255 - I_fg) * thickness
            #   I_final = I_bg * (I_attenuated / 255)
            fg_attenuated = np.clip(255.0 - (255.0 - local_fg) * thickness, 0.0, 255.0)
            fg_ratio = fg_attenuated / 255.0
            beer_lambert_result = local_bg * fg_ratio

            # np.where gated on alpha: only touch pixels where mask is active
            blended = np.where(alpha > 0.5, beer_lambert_result, local_bg)
            projected[dy1:dy2, dx1:dx2] = blended

        projected = np.clip(projected, 0, 255).astype(np.uint8)

        bbox = {
            "x1": max(0, x1),
            "y1": max(0, y1),
            "x2": min(bg_w, x2),
            "y2": min(bg_h, y2)
        }

        return projected, bbox

    def _rotate_image(self, image: np.ndarray, angle: float, bg_fill: Any = 0) -> np.ndarray:
        """Rotates an image by an angle in degrees, expanding bounds and keeping center."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Calculate rotation matrix
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bound sizes
        cos = np.abs(rot_mat[0, 0])
        sin = np.abs(rot_mat[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix to account for translation
        rot_mat[0, 2] += (new_w / 2) - center[0]
        rot_mat[1, 2] += (new_h / 2) - center[1]
        
        # Warp affine
        if isinstance(bg_fill, tuple):
            border_value = bg_fill
        else:
            border_value = (bg_fill, bg_fill, bg_fill)
            
        rotated = cv2.warpAffine(
            image, rot_mat, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value
        )
        return rotated
