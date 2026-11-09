This repo is a sharp, self-contained follow-up to the harmonia_gpu.py analysis. Where that file was a GPU codec component with hidden semantic gaps, vme-tiny-neural is a complete, deterministic, CPU-only pipeline that does exactly what it claims — and the contrast is the whole story.

---

1. What This Repository Actually Is

It is a talk-box synthesizer built entirely on the VME (Vortex Music Engine) stack you've been iterating on. The signal chain is explicit and physically motivated:

```
DownwashResolver (melody + chords)
        ↓
Karplus-Strong plucked string (harmonia_guitar.py)
        ↓
Kelly-Lochbaum vocal tract (mouth_tract.py)
        ↓
vme_talkbox.py — orchestration + rendering
```

It is not a neural network. Despite the repo name, there is zero torch, zero training, zero learned weights. It is a physical-modeling synthesizer with a deterministic musical front-end. That's not a failure — it's the point. The repo delivers a listenable artifact (vme_talkbox_demo.wav) from a pipeline that has no hidden non-determinism, no GPU dependency, and no unverified checkpoints.

---

2. File-by-File Assessment

downwash_resolver.py — The Musical Brain

This is the same Production Configuration A resolver from your earlier VME work, cleaned and self-contained. The Prediction Suffix Tree and joint cost function are unchanged. What's notable is the documentation honesty: the docstring states the verified metrics (stepwise ~45%, repeats 0.0%, mean interval ~2.52, key correlation ~0.72) rather than promising targets it doesn't hit. This is the same "report the trade-off, don't hide it" posture that the vme_v3.py docstring in your earlier corpus adopted — good continuity.

The PST corpus is small (10 diatonic progressions) but sufficient for the task. The sample() method backs off gracefully and falls back to uniform sampling if no context matches — correct behavior.

harmonia_guitar.py — Extended Karplus-Strong

This is the most technically interesting file. It implements:

· A comb loop with a 2-tap damping filter (F(z) = (1-a) + a z⁻¹)
· A first-order all-pass fractional delay for tuning accuracy, with the coefficient derived from the fractional part of the loop delay
· A pick-position comb on the excitation, which notches harmonics at multiples of 1/pick_pos — this is why the pluck sounds like a pick and not a bow
· A two-resonance body EQ at the end (lines not fully shown, but referenced)
· T60-derived per-loop gain so lower strings ring longer naturally

The derivation is physically correct. The loop gain g = 10^(-3(L+frac+a) / (t60 * sr)) is the standard formula for a desired decay time in a feedback comb. The all-pass coefficient c = (1-frac)/(1+frac) is the standard first-order fractional-delay approximation. This is not hand-waving — it is a real string model.

The determinism claim is also true: _seed(*parts) uses zlib.crc32 on the note parameters, and np.random.default_rng(seed) is created per note. No global RNG state. Same input → same output, every time.

mouth_tract.py — Kelly-Lochbaum Vocal Tract

This is a 48-section Kelly-Lochbaum tract at 48 kHz (3.6 mm per section, 17 cm total), with area functions fitted to Peterson-Barney male formant data. Five vowels (ah, eh, ee, oh, oo) are defined by 8-point area profiles, then log-interpolated to 48 sections via np.interp.

The class Tract maintains forward and backward wave arrays f and b, with glottal and lip reflection coefficients rg=0.75, rl=-0.9. The process() method takes an excitation signal and a per-sample vowel curve — meaning the vowel can glide continuously, not just switch discretely. That's why vme_talkbox.py applies a one-pole low-pass filter to the curve before feeding it in.

This is textbook digital waveguide synthesis. The jump from the 8-section Cmajor model to 48 sections is a real improvement in formant accuracy — 8 sections at 48 kHz would be 2.9 cm each, too coarse to resolve F2/F3 for high vowels.

vme_talkbox.py — The Orchestrator

The pipeline is clean and short:

1. melody(seed, n) generates pitches + chords from the resolver
2. For each note, G.pluck() renders a guitar string an octave below the written pitch (standard guitar notation convention)
3. The note is mixed into a source buffer, and the vowel curve is set to the chord's assigned vowel (CHORD_VOWEL maps C→ah, G→oh, Am→ee, etc.)
4. The curve is low-pass filtered for smooth glides
5. M.Tract().process(src, curve) filters the string through the vocal tract
6. Output is peak-normalized to 0.7

The chord-to-vowel mapping is a genuinely musical idea. It makes the "mouth" follow the harmony rather than just the melody — so a I–vi–IV–V progression produces an ah–ee–eh–oh vowel arc. That's a compositional decision embedded in the synthesis architecture, not an afterthought.

test_vme_talkbox.py — Actually Tests Things That Matter

The test suite is small but pointed:

· test_melody_matches_vme_reference: asserts the first 12 notes are exactly [64, 60, 64, 62, 59, 62, 59, 57, 60, 57, 60, 57]
· test_160_note_metrics_match_vme_readme: asserts stepwise ≈ 44.7%, repeats == 0.0, mean interval ≈ 2.52, key corr ≈ 0.718 — exactly the numbers in the README
· test_render_finite_bounded_deterministic: checks output is finite, peak ≤ 0.7001, and two renders with the same seed are bit-identical
· test_every_chord_has_a_vowel: checks the CHORD_VOWEL map covers every chord the resolver can produce

This is the kind of test suite that catches regressions in the properties that matter — determinism, metric fidelity, and coverage of the chord vocabulary. The determinism test is especially important given the physical-modeling pipeline; floating-point non-determinism in scipy.signal would break bit-equality, and the test would catch it.

---

3. The Key Contrast with harmonia_gpu.py

Criterion harmonia_gpu.py vme-tiny-neural
Dependencies torch, torchaudio, triton numpy, scipy only
Hardware GPU required for Triton path CPU only
Verifiability Triton path unverified; "bit-exact" claim cannot be reproduced from the file _verify() and full unittest suite verify everything claimed
Semantic honesty "SSRR" is a mel proxy, not a semantic loss Every component does what its name says
Determinism Depends on GPU nondeterminism Seeded per note, bit-identical across runs
Artifact No audio output in repo vme_talkbox_demo.wav included

The harmonia_gpu.py analysis flagged that the file's biggest weakness was the gap between its name and its behavior. vme-tiny-neural has no such gap. The guitar model is a guitar. The tract is a tract. The resolver is the resolver. The tests assert the README's numbers. That's the difference between a component that is "Woof-shaped" and one that is Woof-class in the sense that matters: small, deterministic, and doing real work with the parameters it has.

---

4. Honest Limitations

· Not neural. The repo name is misleading. There are no learned weights anywhere. It's a physical-modeling synth with a rule-based musical front-end.
· Small musical vocabulary. Ten PST progressions and five vowels. It will produce coherent short phrases, not long-form compositions.
· No MIDI export. Output is a single WAV. The earlier VME repo had pretty_midi; this one doesn't.
· No streaming / real-time path. The pipeline renders offline. A Web Audio port would need to replace scipy.signal.lfilter and the Python loop with AudioWorklet nodes — doable, but not done here.
· The sample/ directory content was not retrieved (the GitHub tree URL returned an error), so I can't assess whether it contains additional demos or notebooks.

---

5. Verdict

This is the cleanest artifact in the VME lineage so far. It takes the musical resolver from the earlier work, bolts on two physically grounded synthesis models (Karplus-Strong + Kelly-Lochbaum), and ships a working demo with tests that verify the exact metrics the README claims. It is deterministic, CPU-only, dependency-light, and honest about what it is.

If the goal is a Woof-class tiny system that fits in a browser — the guitar/tract pipeline is the right kind of thing. It's already small enough. The missing piece is the port: replacing the scipy/numpy rendering with Web Audio PeriodicWave (for the string) and a custom AudioWorklet (for the waveguide tract). That would give you a browser-native talk-box synth with the same musical brain and no GPU requirement at all — which is a more genuinely Woof-shaped achievement than anything in the GPU codec file.
