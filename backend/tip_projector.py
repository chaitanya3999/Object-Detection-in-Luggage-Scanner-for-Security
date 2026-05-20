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
        pos_x_pct: float = 50.0,  # 0 to 100% of bg width
        pos_y_pct: float = 50.0,  # 0 to 100% of bg height
        thickness: float = 1.0,   # Attenuation multiplier
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Overlays the threat image fg_image onto bg_image.
        
        Args:
            bg_image: BGR suitcase image.
            fg_image: BGR threat item image (should have a light/white background).
            scale: Scaling factor for fg_image.
            angle_deg: Rotation angle in degrees.
            pos_x_pct: Center position X percentage (0-100).
            pos_y_pct: Center position Y percentage (0-100).
            thickness: Beer-Lambert attenuation multiplier (1.0 = normal, < 1 = thinner, > 1 = denser).
            
        Returns:
            - projected_image: Attenuated BGR result.
            - bbox: Coordinates of the projected item [x1, y1, x2, y2].
        """
        bg_h, bg_w = bg_image.shape[:2]
        
        # 1. Isolate the threat object from its white background
        fg_gray = cv2.cvtColor(fg_image, cv2.COLOR_BGR2GRAY)
        # Background is white, so the object is darker (< 240)
        _, mask = cv2.threshold(fg_gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # Crop to bounding box of threat to ease transformations
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            tx, ty, tw, th = cv2.boundingRect(cnt)
            cropped_fg = fg_image[ty:ty+th, tx:tx+tw]
            cropped_mask = mask[ty:ty+th, tx:tx+tw]
        else:
            cropped_fg = fg_image
            cropped_mask = mask
            
        # 2. Resize and rotate the threat and its mask
        orig_h, orig_w = cropped_fg.shape[:2]
        # Target size relative to bg width
        target_w = int(bg_w * 0.25 * scale)
        target_h = int(orig_h * (target_w / max(orig_w, 1)))
        target_w = max(target_w, 10)
        target_h = max(target_h, 10)
        
        resized_fg = cv2.resize(cropped_fg, (target_w, target_h), interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(cropped_mask, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Rotate fg and mask
        if angle_deg != 0:
            resized_fg = self._rotate_image(resized_fg, angle_deg, bg_fill=(255, 255, 255))
            resized_mask = self._rotate_image(resized_mask, angle_deg, bg_fill=0)
            
        tf_h, tf_w = resized_fg.shape[:2]
        
        # 3. Compute target coordinates
        center_x = int(bg_w * (pos_x_pct / 100.0))
        center_y = int(bg_h * (pos_y_pct / 100.0))
        
        x1 = center_x - tf_w // 2
        y1 = center_y - tf_h // 2
        x2 = x1 + tf_w
        y2 = y1 + tf_h
        
        # Clip coordinates within background image
        dx1, dy1 = max(0, x1), max(0, y1)
        dx2, dy2 = min(bg_w, x2), min(bg_h, y2)
        
        sx1, sy1 = dx1 - x1, dy1 - y1
        sx2, sy2 = sx1 + (dx2 - dx1), sy1 + (dy2 - dy1)
        
        # Create output image
        projected = bg_image.copy().astype(float)
        
        if (dx2 > dx1) and (dy2 > dy1):
            # Extract local background region
            local_bg = projected[dy1:dy2, dx1:dx2]
            
            # Get matching threat and mask slices
            local_fg = resized_fg[sy1:sy2, sx1:sx2].astype(float)
            local_mask = resized_mask[sy1:sy2, sx1:sx2].astype(float) / 255.0
            
            # Apply thickness modifier to local threat intensity (attenuation strength)
            # Normal: local_fg. Thickness increases absorption (makes it darker).
            # Darker = closer to 0, which multiplies the background even more.
            # I_attenuated = 255 - (255 - I_fg) * thickness
            local_fg_attenuated = np.clip(255.0 - (255.0 - local_fg) * thickness, 0.0, 255.0)
            
            # Apply Beer-Lambert projection equation: I_final = I_bg * (I_fg / 255)
            # Only apply this inside the mask region. Outside the mask, keep local background.
            local_fg_ratio = local_fg_attenuated / 255.0
            projected_region = local_bg * local_fg_ratio
            
            # Blend based on mask transparency to smooth edges
            blend_mask = np.expand_dims(local_mask, axis=2)
            blended_region = projected_region * blend_mask + local_bg * (1.0 - blend_mask)
            
            projected[dy1:dy2, dx1:dx2] = blended_region
            
        projected = np.clip(projected, 0, 255).astype(np.uint8)
        
        # Return projected BGR and the bounding box in absolute pixels
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
