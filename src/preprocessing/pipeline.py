"""
5-Step Preprocessing Pipeline for X-Ray images.

Steps:
1. Four-Channel Tensor Construction (simulated dual-energy from grayscale)
2. Bilateral Filter (edge-preserving denoising)
3. CLAHE Normalization (per-channel contrast enhancement)
4. Letterbox Resize (aspect-ratio preserving)
5. Instance Segmentation integration point
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class PreprocessingPipeline:
    """
    Complete 5-step preprocessing pipeline for X-ray images.

    Usage:
        pipeline = PreprocessingPipeline()
        tensor_4ch = pipeline(image)
    """

    def __init__(
        self,
        target_size: int = 640,
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75,
        bilateral_sigma_space: float = 75,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
    ):
        self.target_size = target_size
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid

        # Create CLAHE object
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid,
        )

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Run the full preprocessing pipeline.

        Args:
            image: Input X-ray image. Can be:
                   - Grayscale (H, W)
                   - BGR/RGB (H, W, 3)
                   - Dual-energy (H, W, 2) with HE and LE channels

        Returns:
            Preprocessed 4-channel tensor (H', W', 4) as float32 [0, 1].
            Shape is (target_size, target_size, 4).
        """
        # Step 1: Construct 4-channel tensor
        tensor_4ch = self.construct_four_channel(image)

        # Step 2: Bilateral filter (per channel)
        tensor_4ch = self.bilateral_filter(tensor_4ch)

        # Step 3: CLAHE normalization (per channel)
        tensor_4ch = self.clahe_normalize(tensor_4ch)

        # Step 4: Letterbox resize
        tensor_4ch, _, _ = self.letterbox_resize(tensor_4ch, self.target_size)

        return tensor_4ch

    def construct_four_channel(self, image: np.ndarray) -> np.ndarray:
        """
        Step 1: Construct a 4-channel tensor from the input.

        For true dual-energy (H, W, 2):
            Channel 0: -log(HE)     — linearized high-energy attenuation
            Channel 1: -log(LE)     — linearized low-energy attenuation
            Channel 2: HE + LE      — total density
            Channel 3: HE - LE      — material-specific differential absorption

        For single-energy grayscale (H, W) or RGB (H, W, 3):
            Simulates dual-energy channels using intensity decomposition.
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        if len(image.shape) == 2:
            # Grayscale: simulate dual-energy
            return self._simulate_dual_energy(image)
        elif image.shape[2] == 2:
            # True dual-energy: HE and LE channels
            return self._process_dual_energy(image)
        elif image.shape[2] == 3:
            # RGB/BGR: convert to grayscale first
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            return self._simulate_dual_energy(gray)
        elif image.shape[2] == 4:
            # Already 4-channel
            return image
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

    def _simulate_dual_energy(self, gray: np.ndarray) -> np.ndarray:
        """
        Simulate dual-energy channels from a single grayscale image.

        Uses Gaussian decomposition to approximate HE (high-frequency detail)
        and LE (low-frequency structure) channels.
        """
        h, w = gray.shape

        # Normalize to [0, 1]
        normalized = gray / 255.0

        # Approximate LE (low-energy): more absorption, shows denser materials
        # Use Gaussian blur to simulate LE response
        le = cv2.GaussianBlur(normalized, (15, 15), 3.0)

        # Approximate HE (high-energy): less absorption, penetrates more
        # Original image is closer to HE response in single-energy scanners
        he = normalized

        # Ensure no zeros for log (add small epsilon)
        eps = 1e-6
        he_safe = np.clip(he, eps, 1.0)
        le_safe = np.clip(le, eps, 1.0)

        # Construct 4 channels
        ch0 = -np.log(he_safe)            # -log(HE): linearized attenuation
        ch1 = -np.log(le_safe)            # -log(LE): linearized attenuation
        ch2 = he + le                      # Sum: total density
        ch3 = he - le                      # Difference: material discrimination

        # Normalize each channel to [0, 1]
        ch0 = self._normalize_channel(ch0)
        ch1 = self._normalize_channel(ch1)
        ch2 = self._normalize_channel(ch2)
        ch3 = self._normalize_channel(ch3)

        tensor = np.stack([ch0, ch1, ch2, ch3], axis=-1)
        return tensor

    def _process_dual_energy(self, dual: np.ndarray) -> np.ndarray:
        """Process true dual-energy input (HE, LE channels)."""
        he = dual[:, :, 0].astype(np.float32) / 255.0
        le = dual[:, :, 1].astype(np.float32) / 255.0

        eps = 1e-6
        he_safe = np.clip(he, eps, 1.0)
        le_safe = np.clip(le, eps, 1.0)

        ch0 = -np.log(he_safe)
        ch1 = -np.log(le_safe)
        ch2 = he + le
        ch3 = he - le

        ch0 = self._normalize_channel(ch0)
        ch1 = self._normalize_channel(ch1)
        ch2 = self._normalize_channel(ch2)
        ch3 = self._normalize_channel(ch3)

        return np.stack([ch0, ch1, ch2, ch3], axis=-1)

    @staticmethod
    def _normalize_channel(channel: np.ndarray) -> np.ndarray:
        """Normalize a channel to [0, 1] range."""
        cmin, cmax = channel.min(), channel.max()
        if cmax - cmin < 1e-8:
            return np.zeros_like(channel)
        return (channel - cmin) / (cmax - cmin)

    def bilateral_filter(self, tensor: np.ndarray) -> np.ndarray:
        """
        Step 2: Apply bilateral filter to each channel.
        Edge-preserving denoising — does NOT blur edges.
        """
        result = np.zeros_like(tensor)
        for c in range(tensor.shape[2]):
            ch = (tensor[:, :, c] * 255).astype(np.uint8)
            filtered = cv2.bilateralFilter(
                ch,
                d=self.bilateral_d,
                sigmaColor=self.bilateral_sigma_color,
                sigmaSpace=self.bilateral_sigma_space,
            )
            result[:, :, c] = filtered.astype(np.float32) / 255.0
        return result

    def clahe_normalize(self, tensor: np.ndarray) -> np.ndarray:
        """
        Step 3: Apply CLAHE to each channel independently.
        Improves visibility of low-contrast objects.
        """
        result = np.zeros_like(tensor)
        for c in range(tensor.shape[2]):
            ch = (tensor[:, :, c] * 255).astype(np.uint8)
            enhanced = self.clahe.apply(ch)
            result[:, :, c] = enhanced.astype(np.float32) / 255.0
        return result

    @staticmethod
    def letterbox_resize(
        image: np.ndarray,
        target_size: int,
        fill_value: float = 0.5,
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Step 4: Letterbox resize maintaining aspect ratio.

        Args:
            image: Input image (H, W, C).
            target_size: Target square size.
            fill_value: Padding fill value.

        Returns:
            Tuple of (resized_image, scale_factor, (pad_left, pad_top)).
        """
        h, w = image.shape[:2]
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        pad_h = target_size - new_h
        pad_w = target_size - new_w
        top = pad_h // 2
        left = pad_w // 2

        if len(image.shape) == 3:
            padded = np.full((target_size, target_size, image.shape[2]), fill_value, dtype=image.dtype)
        else:
            padded = np.full((target_size, target_size), fill_value, dtype=image.dtype)

        padded[top : top + new_h, left : left + new_w] = resized

        return padded, scale, (left, top)


def preprocess_xray(
    image: np.ndarray,
    target_size: int = 640,
    **kwargs,
) -> np.ndarray:
    """
    Convenience function: preprocess an X-ray image.

    Args:
        image: Input image (any format).
        target_size: Target size.

    Returns:
        Preprocessed 4-channel float32 tensor of shape (4, target_size, target_size).
        Note: Returns in CHW format for PyTorch compatibility.
    """
    pipeline = PreprocessingPipeline(target_size=target_size, **kwargs)
    result = pipeline(image)

    # Convert HWC -> CHW for PyTorch
    result = np.transpose(result, (2, 0, 1))
    return result


def preprocess_batch(
    images: list,
    target_size: int = 640,
    **kwargs,
) -> np.ndarray:
    """
    Preprocess a batch of images.

    Returns:
        NumPy array of shape (N, 4, target_size, target_size).
    """
    pipeline = PreprocessingPipeline(target_size=target_size, **kwargs)
    results = []
    for img in images:
        result = pipeline(img)
        result = np.transpose(result, (2, 0, 1))  # HWC -> CHW
        results.append(result)
    return np.stack(results, axis=0)
