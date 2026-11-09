"""VME melody -> guitar string -> talk-box mouth.   python vme_talkbox.py [seed]
The Downwash/PST resolver (VME Configuration A, MIT) writes the notes; a plucked string plays them; the corrected
Kelly-Lochbaum tract (fitted vowel shapes) filters the string. Each chord gets a vowel, so the mouth follows the harmony."""
import random, sys
import numpy as np
from scipy import signal
from scipy.io import wavfile
from downwash_resolver import DownwashResolver
import mouth_tract as M
import harmonia_guitar as G

FS = 48000
CHORD_VOWEL = {"C": "ah", "G": "oh", "Am": "ee", "F": "eh", "Em": "oo", "Dm": "oh"}   # vowel per chord (ah..oo = 0..1 on the tract's vowel axis)

def raw_pitch(last, low=58, high=76):
    if last is None: return random.randint(62, 70)
    return max(low, min(high, last + random.choices([-3, -2, -1, 0, 1, 2, 3], weights=[1, 3, 6, 2, 6, 3, 1], k=1)[0]))

def melody(seed=42, n=24):
    random.seed(seed); np.random.seed(seed)
    r = DownwashResolver(); pitches, chords, last = [], [], None
    for _ in range(n):
        last = raw_pitch(last); r.maybe_change_chord(); pitches.append(r.fit_pitch_to_chord(last)); chords.append(r.current)
    return pitches, chords

def metrics(p):
    iv = [abs(p[i + 1] - p[i]) for i in range(len(p) - 1)]
    hist = np.bincount([x % 12 for x in p], minlength=12) / len(p)
    prof = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    return dict(stepwise=100 * np.mean([i <= 2 for i in iv]), repeats=100 * np.mean([i == 0 for i in iv]), mean_interval=float(np.mean(iv)), key_corr=float(np.corrcoef(hist, prof)[0, 1]))

def render(seed=42, n=24, step=0.28, tail=0.8, glide=0.05):
    pitches, chords = melody(seed, n); N = int(FS * (n * step + tail)); src = np.zeros(N); curve = np.zeros(N)
    for i, (m, c) in enumerate(zip(pitches, chords)):
        a = int(i * step * FS); p = G.pluck(m - 12, step + tail, 0.55 + 0.3 * ((i * 7) % 5) / 4, FS, seed=1000 * seed + i)   # down an octave: guitar sounds an octave below written
        src[a:a + len(p)] += p[:N - a]; curve[a:] = M.ORDER.index(CHORD_VOWEL[c]) / 4
    curve = signal.lfilter([1 - np.exp(-1 / (glide * FS))], [1, -np.exp(-1 / (glide * FS))], curve)
    y = M.Tract().process(src, curve); return y / np.max(np.abs(y)) * 0.7, pitches, chords

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    y, pitches, chords = render(seed); y2, _, _ = render(seed)
    m = metrics(pitches); tp = 20 * np.log10(np.max(np.abs(signal.resample_poly(y, 4, 1))))
    print(f"melody (seed {seed}): {pitches}\nchords: {chords}")
    print(f"melody metrics: stepwise {m['stepwise']:.1f}%  repeats {m['repeats']:.1f}%  mean interval {m['mean_interval']:.2f}  key corr {m['key_corr']:.3f}")
    print(f"pitch range sounding: MIDI {min(pitches)-12}-{max(pitches)-12} (guitar low E = 40, high E 12th fret = 76)")
    print(f"audio: {len(y)/FS:.1f} s, finite {bool(np.all(np.isfinite(y)))}, sample peak {20*np.log10(np.max(np.abs(y))):.1f} dBFS, true peak {tp:.1f} dBTP, deterministic {bool(np.array_equal(y, y2))}")
    wavfile.write("vme_talkbox_demo.wav", FS, (y * 32767).astype(np.int16)); print("wrote vme_talkbox_demo.wav")
