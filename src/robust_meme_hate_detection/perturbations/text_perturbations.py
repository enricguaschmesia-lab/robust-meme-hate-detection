"""Text perturbation functions for adversarial robustness evaluation.

Each ``TextPerturbation`` instance applies a single named transformation with a
configurable severity. Three named severity levels are provided via
``SEVERITY_PRESETS`` and the ``TextPerturbation.from_preset`` constructor;
these map to label-preserving severity floats chosen to match the examples
in the project roadmap (`ProjectIdeaHenrik_Feasibility_Roadmap.md` §"Text
perturbations").

For reproducibility, each instance optionally owns its own ``random.Random``
RNG seeded at construction time. When ``seed`` is ``None`` (default) the
class falls back to the module-level ``random`` module so older callers that
relied on ``random.seed`` at process start continue to work.

Supported modes:
    leetspeak, char_deletion, char_swap, spacing, punctuation,
    case_noise, censoring, keyboard_typo.
"""

from __future__ import annotations

import random
from typing import Literal

LEETSPEAK_MAP: dict[str, list[str]] = {
    'a': ['@', '4'],
    'e': ['3'],
    'i': ['1', '!'],
    'o': ['0'],
    's': ['$', '5'],
    't': ['7'],
    'l': ['1'],
    'z': ['2'],
}

# QWERTY-neighbour map for keyboard_typo. Symmetric — only one direction stored.
_QWERTY_NEIGHBOURS: dict[str, str] = {
    'q': 'wa',     'w': 'qeas',    'e': 'wrsd',    'r': 'etdf',
    't': 'ryfg',   'y': 'tugh',    'u': 'yihj',    'i': 'uojk',
    'o': 'ipkl',   'p': 'ol',
    'a': 'qwsz',   's': 'awedxz',  'd': 'serfcx',  'f': 'drtgvc',
    'g': 'ftyhbv', 'h': 'gyujnb',  'j': 'huiknm',  'k': 'jiolm',
    'l': 'kop',
    'z': 'asx',    'x': 'zsdc',    'c': 'xdfv',    'v': 'cfgb',
    'b': 'vghn',   'n': 'bhjm',    'm': 'njk',
}

TEXT_MODES = (
    'leetspeak',
    'char_deletion',
    'char_swap',
    'spacing',
    'punctuation',
    'case_noise',
    'censoring',
    'keyboard_typo',
)

SeverityLevel = Literal['low', 'medium', 'high']

# Per-mode severity presets. The ``severity`` float is interpreted per mode:
# for character-fraction modes it is the fraction of characters that may be
# touched; for ``char_swap`` it is the absolute swap count divided by 10
# (so 0.1 → 1 swap, 0.2 → 2, 0.3 → 3), which matches the roadmap's
# 1 / 2 / 3-swap severity examples. Numbers chosen to mirror the roadmap's
# examples (5 / 10 / 20 % for char_deletion etc.) and to keep "low" and
# "medium" label-preserving in practice.
SEVERITY_PRESETS: dict[str, dict[SeverityLevel, float]] = {
    'leetspeak':     {'low': 0.10, 'medium': 0.25, 'high': 0.50},
    'char_deletion': {'low': 0.05, 'medium': 0.10, 'high': 0.20},
    # char_swap: severity * 10 == absolute swap count (calibrated to 2/3/5).
    'char_swap':     {'low': 0.20, 'medium': 0.30, 'high': 0.50},
    'spacing':       {'low': 0.05, 'medium': 0.15, 'high': 0.30},
    'punctuation':   {'low': 0.05, 'medium': 0.10, 'high': 0.20},
    'case_noise':    {'low': 0.10, 'medium': 0.25, 'high': 0.50},
    # censoring: reduced from 0.10/0.20/0.40 to be less destructive at all
    # levels (manual inspection — original "high" erased most of the caption).
    'censoring':     {'low': 0.05, 'medium': 0.12, 'high': 0.25},
    'keyboard_typo': {'low': 0.05, 'medium': 0.10, 'high': 0.20},
}


class TextPerturbation:
    """Apply a single text perturbation with configurable severity and seed.

    A perturbation operates on a Python string and returns a string of the
    same kind (possibly empty for very short inputs). When the per-instance
    RNG is seeded, calls are deterministic given the same input and call
    order.
    """

    def __init__(
        self,
        mode: str = 'leetspeak',
        probability: float = 1.0,
        severity: float = 0.1,
        *,
        seed: int | None = None,
    ) -> None:
        if mode not in TEXT_MODES:
            raise ValueError(f"Unknown text perturbation mode: {mode}")
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")
        if not 0.0 <= severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {severity}")

        self.mode = mode
        self.probability = probability
        self.severity = severity
        self.seed = seed
        self._rng: random.Random | None = random.Random(seed) if seed is not None else None

    @classmethod
    def from_preset(
        cls,
        mode: str,
        level: SeverityLevel,
        *,
        probability: float = 1.0,
        seed: int | None = None,
    ) -> "TextPerturbation":
        """Construct using a named severity level from ``SEVERITY_PRESETS``."""
        if mode not in SEVERITY_PRESETS:
            raise ValueError(f"No severity preset for mode: {mode}")
        if level not in SEVERITY_PRESETS[mode]:
            raise ValueError(f"Unknown severity level {level!r} for {mode!r}")
        return cls(
            mode=mode,
            probability=probability,
            severity=SEVERITY_PRESETS[mode][level],
            seed=seed,
        )

    # ------------------------------------------------------------------ rng
    def _rand(self) -> random.Random:
        # Falls back to the module-level random when no seed was provided so
        # legacy callers (training augmentation, example_perturbations.py)
        # keep working without changes.
        return self._rng if self._rng is not None else random

    # ----------------------------------------------------------------- main
    def __call__(self, text: str) -> str:
        rng = self._rand()
        if rng.random() > self.probability:
            return text

        dispatch = {
            'leetspeak':     self._leetspeak,
            'char_deletion': self._char_deletion,
            'char_swap':     self._char_swap,
            'spacing':       self._spacing,
            'punctuation':   self._punctuation,
            'case_noise':    self._case_noise,
            'censoring':     self._censoring,
            'keyboard_typo': self._keyboard_typo,
        }
        return dispatch[self.mode](text, rng)

    # --------------------------------------------------------------- ops
    def _affected_indices(self, text: str, rng: random.Random) -> list[int]:
        if not text:
            return []
        n = max(1, int(len(text) * self.severity))
        n = min(n, len(text))
        return rng.sample(range(len(text)), n)

    def _leetspeak(self, text: str, rng: random.Random) -> str:
        result = list(text)
        for idx in self._affected_indices(text, rng):
            ch = result[idx].lower()
            if ch in LEETSPEAK_MAP:
                result[idx] = rng.choice(LEETSPEAK_MAP[ch])
        return ''.join(result)

    def _char_deletion(self, text: str, rng: random.Random) -> str:
        if len(text) < 2:
            return text
        # Cap deletions at half the string so semantics survive at high severity.
        max_drop = len(text) - max(1, len(text) // 2)
        n = min(max(1, int(len(text) * self.severity)), max_drop)
        if n <= 0:
            return text
        indices = sorted(rng.sample(range(len(text)), n), reverse=True)
        result = list(text)
        for idx in indices:
            result.pop(idx)
        return ''.join(result)

    def _char_swap(self, text: str, rng: random.Random) -> str:
        """Swap adjacent character pairs. Severity*10 is the absolute swap count.

        Spec axis: roadmap "1, 2, 3 swaps". Decoupled from text length so the
        cell label means the same thing for short and long captions.
        """
        if len(text) < 2:
            return text
        result = list(text)
        n_swaps = max(1, int(round(self.severity * 10)))
        # Disjoint adjacent swap positions to avoid undoing a previous swap.
        possible = list(range(len(result) - 1))
        rng.shuffle(possible)
        used: set[int] = set()
        applied = 0
        for idx in possible:
            if applied >= n_swaps:
                break
            if idx in used or (idx - 1) in used or (idx + 1) in used:
                continue
            result[idx], result[idx + 1] = result[idx + 1], result[idx]
            used.add(idx)
            applied += 1
        return ''.join(result)

    def _spacing(self, text: str, rng: random.Random) -> str:
        if not text:
            return text
        result = list(text)
        n_ops = max(1, int(len(result) * self.severity))
        for _ in range(n_ops):
            if not result:
                break
            idx = rng.randrange(len(result))
            if result[idx] == ' ':
                result.pop(idx)
            else:
                result.insert(idx + 1, ' ')
        return ''.join(result)

    def _punctuation(self, text: str, rng: random.Random) -> str:
        if not text:
            return text
        result = list(text)
        punctuation = '.,!?;:*&'
        n_insertions = max(1, int(len(result) * self.severity))
        n_insertions = min(n_insertions, len(result) + 1)
        insertion_points = sorted(
            rng.sample(range(len(result) + 1), n_insertions),
            reverse=True,
        )
        for point in insertion_points:
            result.insert(point, rng.choice(punctuation))
        return ''.join(result)

    def _case_noise(self, text: str, rng: random.Random) -> str:
        result = list(text)
        for idx in self._affected_indices(text, rng):
            ch = result[idx]
            if ch.isupper():
                result[idx] = ch.lower()
            elif ch.islower():
                result[idx] = ch.upper()
        return ''.join(result)

    def _censoring(self, text: str, rng: random.Random) -> str:
        result = list(text)
        for idx in self._affected_indices(text, rng):
            if result[idx].isalpha() or result[idx].isdigit():
                result[idx] = '*'
        return ''.join(result)

    def _keyboard_typo(self, text: str, rng: random.Random) -> str:
        """Replace selected letters with a QWERTY-adjacent letter.

        Non-letter characters are skipped (we don't typo a digit into a
        letter or vice versa). Case is preserved on the substituted letter.
        """
        result = list(text)
        for idx in self._affected_indices(text, rng):
            ch = result[idx]
            base = ch.lower()
            if base in _QWERTY_NEIGHBOURS:
                sub = rng.choice(_QWERTY_NEIGHBOURS[base])
                result[idx] = sub.upper() if ch.isupper() else sub
        return ''.join(result)
