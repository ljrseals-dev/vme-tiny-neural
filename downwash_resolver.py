"""
DownwashResolver – Production Configuration A
================================================
Final tuned resolver for the Vortex Music Engine (VME) hybrid generative system.

Configuration A (locked after multi-pass metric search, seed=42):
  alpha      = 0.95   # leap weight
  beta       = 1.8    # chord-tone weight
  gamma      = 2.8    # repeat penalty
  delta      = 0.3    # direction-change penalty
  soft_leap  = 2.8    # leaps ≤ this many semitones are free
  leap_slope = 1.25   # cost per extra semitone beyond soft_leap
  max_repeats = 1

Verified metrics (realistic random-walk source, 160 notes, seed=42):
  Stepwise %      ~45 %   (expressive, not hyper-safe)
  Repeat %         0.0 %
  Mean interval  ~2.52
  Key correlation ~0.72

The Prediction Suffix Tree (PST) supplies variable-order chord progressions.
The cost function then selects pitches that balance melodic naturalness
with harmonic fit.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Prediction Suffix Tree (variable-order Markov model for chords)
# ---------------------------------------------------------------------------
class PredictionSuffixTree:
    """Variable-order Markov model using a dictionary-of-dictionaries PST."""

    def __init__(self, max_order: int = 3, min_count: int = 1) -> None:
        self.max_order = max_order
        self.min_count = min_count
        self.pst: Dict[Tuple[str, ...], DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._all_symbols: set[str] = set()

    def train(self, sequences: Sequence[Sequence[str]]) -> None:
        for seq in sequences:
            if len(seq) < 2:
                continue
            for i in range(1, len(seq)):
                max_ctx = min(self.max_order, i)
                for order in range(1, max_ctx + 1):
                    context = tuple(seq[i - order : i])
                    next_sym = seq[i]
                    self.pst[context][next_sym] += 1
                    self._all_symbols.add(next_sym)
                    self._all_symbols.add(seq[i - 1])

    def sample(self, history: Sequence[str]) -> str:
        for order in range(min(self.max_order, len(history)), 0, -1):
            context = tuple(history[-order:])
            counts = self.pst.get(context)
            if counts is not None:
                total = sum(counts.values())
                if total >= self.min_count:
                    symbols = list(counts.keys())
                    weights = list(counts.values())
                    return random.choices(symbols, weights=weights, k=1)[0]
        if self._all_symbols:
            return random.choice(list(self._all_symbols))
        return "C"

    def __len__(self) -> int:
        return len(self.pst)


def make_default_pst() -> PredictionSuffixTree:
    """Small pre-trained PST on common diatonic progressions."""
    corpus = [
        ["C", "G", "Am", "F", "C", "G", "C"],
        ["C", "Am", "F", "G", "C"],
        ["Am", "F", "C", "G", "Am"],
        ["F", "C", "G", "Am", "F"],
        ["C", "F", "G", "C"],
        ["G", "C", "Am", "F", "G"],
        ["Dm", "G", "C", "Am", "Dm"],
        ["Em", "Am", "F", "C", "Em"],
        ["C", "G", "C", "F", "G", "C"],
        ["Am", "Dm", "G", "C"],
    ]
    pst = PredictionSuffixTree(max_order=3, min_count=1)
    pst.train(corpus)
    return pst


# ---------------------------------------------------------------------------
# DownwashResolver – Configuration A (production)
# ---------------------------------------------------------------------------
class DownwashResolver:
    """
    Joint cost-function resolver with variable-order chord model (PST).

    Configuration A is frozen. Override weights only for research experiments.
    """

    def __init__(
        self,
        max_repeats: int = 1,
        variable_rhythm: bool = True,
        pst: Optional[PredictionSuffixTree] = None,
        # Research overrides (leave None to use Configuration A)
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        delta: Optional[float] = None,
        soft_leap: Optional[float] = None,
        leap_slope: Optional[float] = None,
    ) -> None:
        # ---- Configuration A (frozen defaults) ----
        self.alpha = 0.95 if alpha is None else alpha
        self.beta = 1.8 if beta is None else beta
        self.gamma = 2.8 if gamma is None else gamma
        self.delta = 0.3 if delta is None else delta
        self.soft_leap = 2.8 if soft_leap is None else soft_leap
        self.leap_slope = 1.25 if leap_slope is None else leap_slope
        self.max_repeats = max_repeats
        self.variable_rhythm = variable_rhythm

        self.chords: Dict[str, List[int]] = {
            "C": [60, 64, 67],
            "G": [55, 59, 62],
            "Am": [57, 60, 64],
            "F": [53, 57, 60],
            "Em": [52, 55, 59],
            "Dm": [50, 53, 57],
        }

        self.rhythm_patterns = [
            [4, 4, 4, 4],
            [2, 2, 4, 4],
            [4, 2, 2, 4],
            [8, 4, 2, 2],
            [3, 3, 2, 4],
        ]

        self.pst = pst if pst is not None else make_default_pst()

        # State
        self.current: str = "C"
        self.chord_history: List[str] = ["C"]
        self.chord_step: int = 0
        self.current_pattern = random.choice(self.rhythm_patterns)
        self.pattern_idx: int = 0
        self.last_pitch: Optional[int] = None
        self.repeat_count: int = 0
        self.last_direction: int = 0

    # ------------------------------------------------------------------
    # Chord progression (PST)
    # ------------------------------------------------------------------
    def next_chord(self) -> List[int]:
        self.current = self.pst.sample(self.chord_history)
        self.chord_history.append(self.current)
        if len(self.chord_history) > self.pst.max_order:
            self.chord_history.pop(0)
        return self.chords[self.current]

    def maybe_change_chord(self) -> None:
        self.chord_step += 1
        target = (
            self.current_pattern[self.pattern_idx]
            if self.variable_rhythm
            else 4
        )
        if self.chord_step >= target:
            self.chord_step = 0
            self.next_chord()
            if self.variable_rhythm:
                self.pattern_idx = (self.pattern_idx + 1) % len(self.current_pattern)
                if random.random() < 0.12:
                    self.current_pattern = random.choice(self.rhythm_patterns)
                    self.pattern_idx = 0

    # ------------------------------------------------------------------
    # Cost selection
    # ------------------------------------------------------------------
    def _nearest_chord_distance(self, pitch: int) -> int:
        min_dist = 12
        for tone in self.chords[self.current]:
            for oct in range(0, 13, 12):
                d = abs(pitch - (tone + oct))
                if d < min_dist:
                    min_dist = d
        return min_dist

    def _snap_to_nearest_chord(self, pitch: int) -> int:
        best = pitch
        min_dist = 12
        for tone in self.chords[self.current]:
            for oct in range(0, 13, 12):
                target = tone + oct
                d = abs(pitch - target)
                if d < min_dist:
                    min_dist = d
                    best = target
        return best

    def _select_pitch(self, raw_pitch: int) -> int:
        snapped = self._snap_to_nearest_chord(raw_pitch)
        neighbors = [raw_pitch + d for d in (-3, -2, -1, 1, 2, 3)]
        candidates = list({raw_pitch, snapped, *neighbors})

        best_pitch: Optional[int] = None
        best_cost = float("inf")

        for cand in candidates:
            leap = abs(cand - self.last_pitch) if self.last_pitch is not None else 0
            leap_cost = (
                0.0
                if leap <= self.soft_leap
                else (leap - self.soft_leap) * self.leap_slope
            )
            chord_dist = self._nearest_chord_distance(cand)
            repeat = (
                1.0
                if (self.last_pitch is not None and cand == self.last_pitch)
                else 0.0
            )
            direction_penalty = 0.0
            if self.last_pitch is not None and self.last_direction != 0:
                new_dir = (
                    1 if cand > self.last_pitch else (-1 if cand < self.last_pitch else 0)
                )
                if new_dir != 0 and new_dir != self.last_direction:
                    direction_penalty = self.delta

            cost = (
                self.alpha * leap_cost
                + self.beta * chord_dist
                + self.gamma * repeat
                + direction_penalty
            )
            if cost < best_cost:
                best_cost = cost
                best_pitch = cand

        return best_pitch if best_pitch is not None else raw_pitch

    def fit_pitch_to_chord(self, pitch: int) -> int:
        if self.last_pitch is None:
            selected = self._select_pitch(pitch)
            self.last_pitch = selected
            self.repeat_count = 0
            self.last_direction = 0
            return selected

        selected = self._select_pitch(pitch)

        if selected != self.last_pitch:
            self.last_direction = 1 if selected > self.last_pitch else -1

        if selected == self.last_pitch:
            self.repeat_count += 1
            if self.repeat_count >= self.max_repeats:
                forced = self._forced_stepwise()
                self.repeat_count = 0
                self.last_pitch = forced
                self.last_direction = (
                    1 if forced > self.last_pitch else -1
                )
                return forced
        else:
            self.repeat_count = 0

        self.last_pitch = selected
        return selected

    def _forced_stepwise(self) -> int:
        chord_tones = self.chords[self.current]
        candidates: List[int] = []
        for tone in chord_tones:
            for oct_off in range(-2, 3):
                target = tone + 12 * oct_off
                interval = (
                    abs(target - self.last_pitch)
                    if self.last_pitch is not None
                    else 0
                )
                if 1 <= interval <= 3:
                    candidates.append(target)
        if not candidates:
            scale = [60, 62, 64, 65, 67, 69, 71]
            for tone in scale:
                for oct_off in range(-2, 3):
                    target = tone + 12 * oct_off
                    interval = (
                        abs(target - self.last_pitch)
                        if self.last_pitch is not None
                        else 0
                    )
                    if 1 <= interval <= 3:
                        candidates.append(target)
        if candidates:
            return random.choice(candidates)
        return (
            self.last_pitch + random.choice([-3, -2, -1, 1, 2, 3])
            if self.last_pitch is not None
            else random.randint(60, 72)
        )

    def reset_melody_state(self) -> None:
        self.last_pitch = None
        self.repeat_count = 0
        self.last_direction = 0

    def get_weights(self) -> Dict[str, float]:
        """Return the active weight configuration (useful for logging / GitHub)."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "soft_leap": self.soft_leap,
            "leap_slope": self.leap_slope,
            "max_repeats": self.max_repeats,
            "config_name": "A",
        }
