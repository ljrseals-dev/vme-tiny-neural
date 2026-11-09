"""Corrected Kelly-Lochbaum tract + guitar-string source (talk-box style).
Differences from MouthModel_v1.cmajor: N = 48 sections at 48 kHz (3.6 mm each, 17 cm tract) instead of 8 (2.9 cm),
standard pressure-wave junctions, glottal and lip boundary reflections. Source = any audio (here: a plucked guitar string)."""
import numpy as np
from scipy import signal
VOW_CMAJOR = {'ah': (1.8,1.6,1.4,0.9,0.7,0.8,1.2,1.0), 'eh': (1.2,1.1,1.0,0.6,0.5,0.9,1.3,1.1), 'ee': (1.0,0.9,0.8,0.5,0.3,1.1,1.5,1.2),
       'oh': (1.4,1.3,1.2,1.1,0.9,0.7,0.8,0.4), 'oo': (1.2,1.0,0.9,1.0,0.4,0.6,0.7,0.2)}   # the Cmajor file's area targets
ORDER = ['ah', 'eh', 'ee', 'oh', 'oo']
# Area functions FITTED to textbook (Peterson-Barney male) F1-F3 for this 48-section tract; the Cmajor profiles gave near-identical vowels.
VOW = {k: tuple(v) for k, v in {'ah': [3.498, 0.1, 8.0, 0.515, 8.0, 0.422, 8.0, 0.984], 'eh': [0.684, 0.864, 0.691, 0.1, 0.956, 2.821, 0.978, 0.989], 'ee': [0.683, 0.907, 0.686, 0.1, 0.1, 8.0, 0.62, 0.273], 'oh': [4.532, 0.1, 0.448, 1.095, 0.1, 0.231, 1.78, 0.917], 'oo': [2.39, 8.0, 0.998, 4.118, 0.457, 0.493, 8.0, 0.1]}.items()}
N = 48

def area_function(vowel, jaw=1.0, tongue=1.0, lips=1.0):
    """vowel in [0,1] (ah..oo, as in the Cmajor knob) -> N-section area function."""
    v = float(np.clip(vowel, 0, 1)) * 4; lo = min(int(v), 3); t = v - lo
    a8 = np.exp((1 - t) * np.log(VOW[ORDER[lo]]) + t * np.log(VOW[ORDER[lo + 1]]))
    a8[0] *= jaw; a8[1] *= jaw; a8[4] *= tongue; a8[7] *= lips
    x8 = (np.arange(8) + 0.5) / 8; xN = (np.arange(N) + 0.5) / N
    return np.maximum(np.interp(xN, x8, a8), 0.05)

class Tract:
    def __init__(self, rg=0.75, rl=-0.9):
        self.f = np.zeros(N); self.b = np.zeros(N); self.rg, self.rl = rg, rl; self.prev = 0.0
    def process(self, src, vowel_curve, block=32):
        """src: excitation samples; vowel_curve: per-sample vowel position (0..1)."""
        out = np.empty(len(src)); f, b = self.f, self.b
        for s0 in range(0, len(src), block):
            A = area_function(vowel_curve[s0]); r = (A[:-1] - A[1:]) / (A[:-1] + A[1:])
            for i in range(s0, min(s0 + block, len(src))):
                fn = np.empty(N); bn = np.empty(N)
                fn[0] = src[i] + self.rg * b[0]; fn[1:] = (1 + r) * f[:-1] - r * b[1:]
                bn[:-1] = r * f[:-1] + (1 - r) * b[1:]; bn[-1] = self.rl * f[-1]
                f, b = fn, bn
                y = (1 + self.rl) * f[-1]; out[i] = y - 0.97 * self.prev; self.prev = y   # lip radiation (first-order differentiator)
        self.f, self.b = f, b; return out

def formants(vowel, fs=48000, n=2048, fmax=4200, k=3):
    imp = np.zeros(n); imp[0] = 1.0; y = Tract().process(imp, np.full(n, vowel), block=4096)
    H = np.abs(np.fft.rfft(y[:2048] * np.hanning(2048), 1 << 15)); f = np.fft.rfftfreq(1 << 15, 1 / fs); m = f < fmax
    pk, _ = signal.find_peaks(H[m], prominence=0.05 * H[m].max()); return [round(float(f[m][p])) for p in pk[:k]]
