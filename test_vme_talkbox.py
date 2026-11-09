import unittest, numpy as np
import vme_talkbox as V
class T(unittest.TestCase):
    def test_melody_matches_vme_reference(self):
        p, _ = V.melody(42, 12); self.assertEqual(p, [64, 60, 64, 62, 59, 62, 59, 57, 60, 57, 60, 57])   # same first notes as the VME README run
    def test_160_note_metrics_match_vme_readme(self):
        m = V.metrics(V.melody(42, 160)[0]); self.assertAlmostEqual(m["stepwise"], 44.7, places=1); self.assertEqual(m["repeats"], 0.0)
        self.assertAlmostEqual(m["mean_interval"], 2.52, places=2); self.assertAlmostEqual(m["key_corr"], 0.718, places=3)
    def test_render_finite_bounded_deterministic(self):
        a, _, _ = V.render(7, 6); b, _, _ = V.render(7, 6)
        self.assertTrue(np.all(np.isfinite(a))); self.assertLessEqual(np.max(np.abs(a)), 0.7001); self.assertTrue(np.array_equal(a, b))
    def test_every_chord_has_a_vowel(self):
        _, c = V.melody(3, 200); self.assertTrue(set(c) <= set(V.CHORD_VOWEL))
if __name__ == "__main__": unittest.main()
