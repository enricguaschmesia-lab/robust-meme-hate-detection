"""Text perturbation functions for adversarial robustness evaluation.

This module implements label-preserving text transformations including:
- Leetspeak conversion (4 → A, 3 → E, etc.)
- Character deletion
- Character swaps
- Spacing modifications
- Punctuation noise
- Case variations
- Character censoring
"""

from __future__ import annotations

import random

import numpy as np

# Leetspeak mapping
LEETSPEAK_MAP = {
    'a': ['a', '@', '4'],
    'e': ['e', '3'],
    'i': ['i', '1', '!'],
    'o': ['o', '0'],
    's': ['s', '$', '5'],
    't': ['t', '7'],
    'l': ['l', '1'],
    'z': ['z', '2'],
}


class TextPerturbation:
    """Apply a single text perturbation with configurable probability and severity."""

    def __init__(
        self,
        mode: str = 'leetspeak',
        probability: float = 0.5,
        severity: float = 0.1,
    ) -> None:
        """
        Args:
            mode: Type of perturbation. One of:
                'leetspeak', 'char_deletion', 'char_swap', 'spacing',
                'punctuation', 'case_noise', 'censoring'
            probability: Chance (0-1) of applying this perturbation to a sample.
            severity: Strength of perturbation (0-1):
                - All modes: fraction of characters affected (0.1 → 10%, 1.0 → 100%)
        """
        if mode not in [
            'leetspeak',
            'char_deletion',
            'char_swap',
            'spacing',
            'punctuation',
            'case_noise',
            'censoring',
        ]:
            raise ValueError(f"Unknown text perturbation mode: {mode}")

        self.mode = mode
        self.probability = probability
        self.severity = severity

    def __call__(self, text: str) -> str:
        """Apply perturbation with given probability."""
        if random.random() > self.probability:
            return text

        if self.mode == 'leetspeak':
            return self._leetspeak(text)
        elif self.mode == 'char_deletion':
            return self._char_deletion(text)
        elif self.mode == 'char_swap':
            return self._char_swap(text)
        elif self.mode == 'spacing':
            return self._spacing(text)
        elif self.mode == 'punctuation':
            return self._punctuation(text)
        elif self.mode == 'case_noise':
            return self._case_noise(text)
        elif self.mode == 'censoring':
            return self._censoring(text)
        else:
            return text

    def _leetspeak(self, text: str) -> str:
        """Convert characters to leetspeak variants."""
        result = list(text)
        num_chars = max(1, int(len(result) * self.severity))
        indices = random.sample(range(len(result)), min(num_chars, len(result)))

        for idx in indices:
            char = result[idx].lower()
            if char in LEETSPEAK_MAP:
                result[idx] = random.choice(LEETSPEAK_MAP[char])

        return ''.join(result)

    def _char_deletion(self, text: str) -> str:
        """Delete random characters (but keep at least 50% of text)."""
        if len(text) < 2:
            return text

        num_chars = max(1, int(len(text) * self.severity))
        num_chars = min(num_chars, len(text) - max(1, len(text) // 2))

        indices = sorted(
            random.sample(range(len(text)), num_chars),
            reverse=True,
        )
        result = list(text)
        for idx in indices:
            result.pop(idx)

        return ''.join(result)

    def _char_swap(self, text: str) -> str:
        """Swap random adjacent characters."""
        if len(text) < 2:
            return text

        result = list(text)
        num_swaps = max(1, int(len(result) * self.severity))

        for _ in range(num_swaps):
            idx = random.randint(0, len(result) - 2)
            result[idx], result[idx + 1] = result[idx + 1], result[idx]

        return ''.join(result)

    def _spacing(self, text: str) -> str:
        """Add, remove, or modify spacing."""
        result = list(text)
        num_spaces = max(1, int(len(result) * self.severity))

        for _ in range(num_spaces):
            idx = random.randint(0, len(result) - 1)
            if result[idx] == ' ':
                # Remove space
                result.pop(idx)
            else:
                # Add space after this character
                result.insert(idx + 1, ' ')

        return ''.join(result)

    def _punctuation(self, text: str) -> str:
        """Add random punctuation marks."""
        result = list(text)
        punctuation = '.,!?;:*&'
        num_insertions = max(1, int(len(result) * self.severity))

        insertion_points = sorted(
            random.sample(range(len(result) + 1), min(num_insertions, len(result) + 1)),
            reverse=True,
        )

        for point in insertion_points:
            result.insert(point, random.choice(punctuation))

        return ''.join(result)

    def _case_noise(self, text: str) -> str:
        """Randomly flip case of characters."""
        result = list(text)
        num_chars = max(1, int(len(result) * self.severity))
        indices = random.sample(range(len(result)), min(num_chars, len(result)))

        for idx in indices:
            if result[idx].isupper():
                result[idx] = result[idx].lower()
            elif result[idx].islower():
                result[idx] = result[idx].upper()

        return ''.join(result)

    def _censoring(self, text: str) -> str:
        """Replace characters with asterisks."""
        result = list(text)
        num_chars = max(1, int(len(result) * self.severity))
        indices = random.sample(range(len(result)), min(num_chars, len(result)))

        for idx in indices:
            if result[idx].isalpha() or result[idx].isdigit():
                result[idx] = '*'

        return ''.join(result)
