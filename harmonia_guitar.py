#!/usr/bin/env python3
"""
harmonia_guitar.py - plucked-string guitar rendered DOWNSTREAM of the voice codec.

The codec and synth head are trained on singing voice only, so guitar never goes
through the latent. It is rendered on its own at the codec's output rate (24 kHz) and
mixed into the decoded vocal bus afterwards:

    vocal wav (SubBandDecoder, 24 kHz) ----------------------+
                                                             +--> mix_downstream --> out
    note events -> render_guitar (this file, numpy/scipy) ---+

Synthesis: extended Karplus-Strong string (comb loop with 2-tap damping filter and an
all-pass fractional delay for tuning), pick-position comb on the excitation, and a
light two-resonance body EQ. Deterministic: seeded per note, no global RNG state.

Pure numpy/scipy, no torch.   Checks:  python -m unittest tests.test_guitar
"""
from __future__ import annotations
import zlib
import numpy as np
from scipy import signal

SR = 24_000                       # matches harmonia_synth.SR
OPEN_STRINGS = (40, 45, 50, 55, 59, 64)   # E2 A2 D3 G3 B3 E4 (standard tuning)


def midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


def _seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


def pluck(midi: float, dur: float = 3.0, vel: float = 0.8, sr: int = SR,
          t60: float | None = None, pick_pos: float = 0.14, bright: float = 0.5,
          seed: int = 0) -> np.ndarray:
    """One plucked note, peak-normalised to `vel` (0..1). `bright` 0..1 darkens/brightens the pick."""
    f0 = midi_to_hz(midi)
    n = int(sr * dur)
    a = 0.5                                   # loop damping tap: F(z) = (1-a) + a z^-1, DC delay = a samples
    D = sr / f0 - a                           # total loop delay the all-pass must complete
    L = int(np.floor(D - 0.5))                # integer delay; fractional part in [0.5, 1.5)
    frac = D - L
    c = (1.0 - frac) / (1.0 + frac)           # first-order all-pass coefficient
    if t60 is None:                           # lower strings ring longer
        t60 = float(np.clip(4.0 * (110.0 / f0) ** 0.35, 0.8, 6.0))
    g = 10.0 ** (-3.0 * (L + frac + a) / (t60 * sr))   # per-loop gain for the requested T60

    # H(z) = (1 + c z^-1) / [ (1 + c z^-1) - g z^-L (1-a + a z^-1)(c + z^-1) ]
    ap_num = np.array([c, 1.0])
    loop = np.convolve([1.0 - a, a], ap_num) * g
    den = np.zeros(L + len(loop)); den[:2] = [1.0, c]; den[L:L + len(loop)] -= loop
    num = np.array([1.0, c])

    rng = np.random.default_rng(seed)
    w = rng.uniform(-1.0, 1.0, L + 2)
    k = max(1, int(round(pick_pos * (L + frac))))
    ex = w.copy(); ex[k:] -= w[:-k]            # pick-position comb (notches harmonics at multiples of 1/pick_pos)
    cut = 600.0 + 7400.0 * float(np.clip(bright, 0.0, 1.0)) * (0.5 + 0.5 * vel)
    ex = signal.lfilter(*signal.butter(1, min(cut, sr * 0.45) / (sr / 2)), ex)
    x = np.zeros(n); x[:len(ex)] = ex[:n]
    y = signal.lfilter(num, den, x)
    tail = min(int(0.02 * sr), n)              # 20 ms release so notes never click at the end
    y[-tail:] *= np.linspace(1.0, 0.0, tail)
    y = y / (np.max(np.abs(y)) + 1e-12) * float(np.clip(vel, 0.0, 1.0))
    return y.astype(np.float64)


def _peaking(f: float, q: float, gain_db: float, sr: int):
    A = 10 ** (gain_db / 40); w0 = 2 * np.pi * f / sr; al = np.sin(w0) / (2 * q); cw = np.cos(w0)
    b = np.array([1 + al * A, -2 * cw, 1 - al * A]); a = np.array([1 + al / A, -2 * cw, 1 - al / A])
    return b / a[0], a / a[0]


def body(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """Light acoustic-body colouration: air resonance ~100 Hz, top plate ~215 Hz, gentle 2.8 kHz presence."""
    for f, q, g in ((100.0, 2.0, 4.0), (215.0, 2.5, 3.0), (2800.0, 0.9, 2.0)):
        x = signal.lfilter(*_peaking(f, q, g, sr), x)
    return x


def strum(start: float, midis, spread_ms: float = 22.0, dur: float = 3.0, vel: float = 0.8,
          down: bool = True):
    """Events for a strummed chord (list of (start_s, midi, dur_s, vel))."""
    ms = sorted(midis, reverse=not down)
    out = []
    for i, m in enumerate(ms):
        out.append((start + i * spread_ms / 1000.0, m, dur, vel * (0.92 + 0.08 * (i % 2))))
    return out


def render_guitar(events, sr: int = SR, length_s: float | None = None, use_body: bool = True) -> np.ndarray:
    """events: iterable of (start_s, midi, dur_s, vel). Returns float32 mono, peak <= 0.95."""
    events = sorted(events)
    if not events:
        return np.zeros(int((length_s or 0) * sr), dtype=np.float32)
    end = max(s + d for s, _, d, _ in events)
    n = int(sr * (length_s if length_s is not None else end))
    mix = np.zeros(n)
    for s, m, d, v in events:
        note = pluck(m, d, v, sr, seed=_seed(s, m, d, v))
        i0 = int(round(s * sr)); i1 = min(n, i0 + len(note))
        if i0 < n and i1 > i0:
            mix[i0:i1] += note[: i1 - i0]
    if use_body:
        mix = body(mix, sr)
    pk = np.max(np.abs(mix))
    if pk > 0.95:
        mix *= 0.95 / pk
    return mix.astype(np.float32)


def mix_downstream(vocal: np.ndarray, guitar: np.ndarray, guitar_db: float = -9.0,
                   duck_db: float = 3.0, sr: int = SR) -> np.ndarray:
    """Mix guitar under the decoded vocal. Guitar is ducked up to `duck_db` while the vocal is active
    (envelope follower, 10 ms attack / 250 ms release). Output is peak-limited by scaling only if > 0.99."""
    n = max(len(vocal), len(guitar))
    v = np.zeros(n); v[:len(vocal)] = vocal
    gtr = np.zeros(n); gtr[:len(guitar)] = guitar
    env = np.abs(v)
    att = np.exp(-1.0 / (0.010 * sr)); rel = np.exp(-1.0 / (0.250 * sr))
    e = np.empty(n); cur = 0.0
    for i in range(n):
        x = env[i]; cur = att * cur + (1 - att) * x if x > cur else rel * cur + (1 - rel) * x
        e[i] = cur
    act = np.clip(e / (0.05 + 1e-12), 0.0, 1.0)            # 0 below silence, 1 once the vocal is clearly present
    gain = 10 ** ((guitar_db - duck_db * act) / 20.0)
    out = v + gtr * gain
    pk = np.max(np.abs(out))
    return (out * (0.99 / pk) if pk > 0.99 else out).astype(np.float32)


def guitar_downstream(vocal: np.ndarray, events, sr: int = SR, guitar_db: float = -9.0,
                      duck_db: float = 3.0) -> np.ndarray:
    return mix_downstream(vocal, render_guitar(events, sr, len(vocal) / sr), guitar_db, duck_db, sr)


if __name__ == "__main__":
    ev = strum(0.0, [40, 47, 52, 56, 59, 64], dur=4.0) + strum(2.5, [45, 52, 57, 61, 64, 69], dur=4.0, down=False)
    g = render_guitar(ev)
    print(f"rendered {len(g)/SR:.2f} s, peak {np.max(np.abs(g)):.3f}")
