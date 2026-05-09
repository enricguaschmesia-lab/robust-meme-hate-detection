"""Image perturbation functions for adversarial robustness evaluation.

This module implements label-preserving image transformations including:
- Gaussian noise
- Blur (Gaussian)
- Compression (JPEG)
- Brightness/contrast adjustment
- Translation and cropping
- Mild occlusions (small masks)
"""

from __future__ import annotations

import random
from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class ImagePerturbation:
    """Apply a single image perturbation with configurable probability and severity."""

    def __init__(
        self,
        mode: str = 'gaussian_noise',
        probability: float = 0.5,
        severity: float = 0.1,
    ) -> None:
        """
        Args:
            mode: Type of perturbation. One of:
                'gaussian_noise', 'blur', 'compression', 'brightness',
                'contrast', 'translation', 'crop', 'occlusion'
            probability: Chance (0-1) of applying this perturbation to a sample.
            severity: Strength of perturbation (0-1 or specific units):
                - 'gaussian_noise': std dev as fraction of [0, 255] range
                - 'blur': radius in pixels
                - 'compression': JPEG quality (0.1 → quality 90, 1.0 → quality 10)
                - 'brightness': scale factor (0.1 → 0.7x, 1.0 → 1.3x brightness)
                - 'contrast': scale factor (0.1 → 0.7x, 1.0 → 1.3x contrast)
                - 'translation': max pixel shift as fraction of image size
                - 'crop': crop size as fraction of image
                - 'occlusion': mask size as fraction of image (creates 1-3 masks)
        """
        if mode not in [
            'gaussian_noise',
            'blur',
            'compression',
            'brightness',
            'contrast',
            'translation',
            'crop',
            'occlusion',
        ]:
            raise ValueError(f"Unknown image perturbation mode: {mode}")

        self.mode = mode
        self.probability = probability
        self.severity = severity

    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply perturbation with given probability."""
        if random.random() > self.probability:
            return image

        if self.mode == 'gaussian_noise':
            return self._gaussian_noise(image)
        elif self.mode == 'blur':
            return self._blur(image)
        elif self.mode == 'compression':
            return self._compression(image)
        elif self.mode == 'brightness':
            return self._brightness(image)
        elif self.mode == 'contrast':
            return self._contrast(image)
        elif self.mode == 'translation':
            return self._translation(image)
        elif self.mode == 'crop':
            return self._crop(image)
        elif self.mode == 'occlusion':
            return self._occlusion(image)
        else:
            return image

    def _gaussian_noise(self, image: Image.Image) -> Image.Image:
        """Add Gaussian noise to image."""
        image_array = np.array(image, dtype=np.float32)
        
        # severity is std dev as fraction of [0, 255]
        std_dev = self.severity * 255.0
        noise = np.random.normal(0, std_dev, image_array.shape)
        
        noisy_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)

    def _blur(self, image: Image.Image) -> Image.Image:
        """Apply Gaussian blur."""
        # severity is blur radius in pixels (0-1 maps to 0-10 pixels)
        radius = max(1, int(self.severity * 10))
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def _compression(self, image: Image.Image) -> Image.Image:
        """Apply JPEG compression."""
        # severity maps to JPEG quality: 0.1 → quality 90 (less compression), 1.0 → quality 10 (more compression)
        quality = max(10, min(95, int(95 - self.severity * 85)))
        
        # Save to JPEG and reload
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')

    def _brightness(self, image: Image.Image) -> Image.Image:
        """Adjust brightness."""
        # severity is scale factor: 0.1 → 0.7 brightness, 1.0 → 1.3 brightness
        factor = 0.7 + self.severity * 0.6
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def _contrast(self, image: Image.Image) -> Image.Image:
        """Adjust contrast."""
        # severity is scale factor: 0.1 → 0.7 contrast, 1.0 → 1.3 contrast
        factor = 0.7 + self.severity * 0.6
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def _translation(self, image: Image.Image) -> Image.Image:
        """Translate image (shift pixels with padding)."""
        width, height = image.size
        image_array = np.array(image)
        
        # severity is max shift as fraction of image size
        max_shift = int(self.severity * min(width, height))
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        
        # Create output array with same size, padded with gray
        output = np.ones_like(image_array) * 128  # Gray padding
        
        # Determine source and destination regions
        src_x_start = max(0, shift_x)
        src_x_end = min(width, width + shift_x)
        src_y_start = max(0, shift_y)
        src_y_end = min(height, height + shift_y)
        
        dst_x_start = max(0, -shift_x)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        dst_y_start = max(0, -shift_y)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        
        # Copy the non-padded region
        output[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image_array[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return Image.fromarray(output.astype(np.uint8))

    def _crop(self, image: Image.Image) -> Image.Image:
        """Crop and restore to original size (simulates zoom)."""
        width, height = image.size
        
        # severity is crop as fraction of image (e.g., 0.1 means crop 10% from edges)
        crop_px = int(self.severity * min(width, height))
        
        # Crop from random corner
        left = random.randint(0, crop_px)
        top = random.randint(0, crop_px)
        right = width - random.randint(0, crop_px)
        bottom = height - random.randint(0, crop_px)
        
        # Ensure valid crop box
        if right <= left or bottom <= top:
            return image
        
        cropped = image.crop((left, top, right, bottom))
        return cropped.resize((width, height), Image.BILINEAR)

    def _occlusion(self, image: Image.Image) -> Image.Image:
        """Add mild occlusion mask (random rectangular patches)."""
        width, height = image.size
        image_array = np.array(image, dtype=np.uint8)
        
        # severity is mask size as fraction of image
        mask_size = max(1, int(self.severity * min(width, height)))
        
        # Number of masks: 1-3 depending on severity
        num_masks = max(1, int(self.severity * 3))
        
        for _ in range(num_masks):
            x = random.randint(0, max(0, width - mask_size))
            y = random.randint(0, max(0, height - mask_size))
            
            # Use gray color (neutral) or random color
            color = random.randint(100, 155)
            image_array[y:y + mask_size, x:x + mask_size] = color
        
        return Image.fromarray(image_array)
