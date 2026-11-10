import unittest

import numpy as np
import torch
import torch.nn.functional as F

from hindi_detector.acoustic_deletion import AcousticDeletionNetwork, WordAcoustics, frame_count, pair_indices


class AcousticDeletionTests(unittest.TestCase):
    def test_verified_convolution_geometry_matches_observed_fixture(self):
        config = {'model_type': 'wav2vec2', 'conv_kernel': [10, 3, 3, 3, 3, 2, 2], 'conv_stride': [5, 2, 2, 2, 2, 2, 2]}
        self.assertEqual(frame_count(24320, config), 75)
        bad = {**config, 'add_adapter': True}
        with self.assertRaises(ValueError):
            frame_count(24320, bad)

    def test_source_pair_ranking_is_row_order_invariant(self):
        rows = [{'pair_id': p, 'family': f, 'label': label} for p in ('a', 'b')
                for f, label in [('S0', 0), ('P1', 1), ('S_world', 0), ('P1w', 1), ('S2', 0)]]
        scores = torch.arange(len(rows), dtype=torch.float32) / 5
        def loss(rs, z):
            pairs = pair_indices(rs)
            return torch.stack([F.softplus(-(z[a] - z[b])) for a, b in pairs]).mean()
        order = np.random.default_rng(1).permutation(len(rows))
        self.assertAlmostEqual(float(loss(rows, scores)), float(loss([rows[i] for i in order], scores[order])), places=6)
        with self.assertRaises(ValueError):
            pair_indices(rows + rows[:1])

    def test_gain_normalization_preserves_spectral_evidence(self):
        cfg = {'model_type': 'wav2vec2', 'conv_kernel': [10, 3, 3, 3, 3, 2, 2], 'conv_stride': [5, 2, 2, 2, 2, 2, 2]}
        frontend = WordAcoustics({'s': 0, 'k': 1}, [cfg, cfg])
        t = np.arange(16000, dtype=np.float32) / 16000
        wave = (.2 * np.sin(2 * np.pi * 600 * t) + .1 * np.sin(2 * np.pi * 1700 * t)).astype(np.float32)
        windows = [{'gap_start_frame': 20, 'gap_end_frame': 23, 'native_focal_span': {'start_frame': 2, 'end_frame': 4}, 'start_frame': 18} for _ in range(3)]
        evidence = [{'windows': windows}, {'windows': windows}]
        args = ((.2, .7), evidence, ['s', 'ax', 'k'], 1, np.zeros(24))
        a = frontend.extract(wave, *args)[0]
        b = frontend.extract(wave / 2, *args)[0]
        self.assertLess(float(np.abs(a[:64] - b[:64]).mean()), .02)
        np.testing.assert_array_equal(a[64:], b[64:])

    def test_acoustic_head_has_finite_gradient_on_real_input_schema(self):
        torch.manual_seed(81)
        net = AcousticDeletionNetwork(20)
        mel = torch.randn(3, 66, 128)
        mel[:, 64:] = torch.sigmoid(mel[:, 64:])
        out = net(mel, torch.zeros((3, 2), dtype=torch.long), torch.zeros(3, 14))
        F.binary_cross_entropy_with_logits(out, torch.tensor([0., 1., 0.])).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters()))

    def test_real_v71_fixture_uses_absolute_gap_and_crop_relative_nucleus(self):
        # Metadata only from train/S2 नलगाए WAV SHA
        # 0d5febc9bb5e2fd354d08337db65ff3e9b94010d357349e66112133fd8fc35fa.
        # Both real probes: absolute gap42..46; nucleus global43..44 at all pads.
        cfg = {'model_type': 'wav2vec2', 'conv_kernel': [10, 3, 3, 3, 3, 2, 2], 'conv_stride': [5, 2, 2, 2, 2, 2, 2]}
        frontend = WordAcoustics({'l': 0, 'g': 1}, [cfg, cfg])
        windows = [{'gap_start_frame': 42, 'gap_end_frame': 46, 'start_frame': a,
                    'native_focal_span': {'start_frame': b, 'end_frame': b + 1}}
                   for a, b in [(33, 10), (31, 12), (30, 13)]]
        wave = np.sin(np.arange(23347, dtype=np.float32) * .1)
        bounds = (.6687942708333334, 1.1349236111111112)
        mel, _, _, _ = frontend.extract(wave, bounds, [{'windows': windows}] * 2,
            ['n', 'ax', 'l', 'ax', 'g', 'aa', 'e'], 3, np.zeros(24))
        self.assertEqual(frame_count(23347, cfg), 72)
        self.assertEqual(int(mel[64].argmax()), 61)
        self.assertEqual(int(mel[65].argmax()), 59)
        # An incorrectly window-relative gap must be rejected, never clipped.
        bad = [{'gap_start_frame': 73, 'gap_end_frame': 77, 'start_frame': 31,
                'native_focal_span': {'start_frame': 12, 'end_frame': 13}}] * 3
        with self.assertRaisesRegex(ValueError, 'outside production word'):
            frontend.extract(wave, bounds, [{'windows': bad}] * 2,
                ['n', 'ax', 'l', 'ax', 'g', 'aa', 'e'], 3, np.zeros(24))

    def test_intervention_geometry_is_rejected_from_query_evidence(self):
        from hindi_detector.acoustic_deletion import require_query_evidence
        with self.assertRaises(ValueError):
            require_query_evidence([{'windows': [], 'edit': {'start_sample': 10}}])
