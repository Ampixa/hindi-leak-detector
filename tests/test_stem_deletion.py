import unittest
import math

import numpy as np
import torch
import torch.nn.functional as F

from hindi_detector.stem_deletion import StemVowelNetwork, interpolate_stem, query_grid
from hindi_detector.nucleus_deletion import PhoneContextNetwork


class StemClockTests(unittest.TestCase):
    def test_grid_matches_actual_centered_stft_and_pytorch_resampling(self):
        for bounds, samples in [((0., .2), 16000), ((.8, 1.), 16000), ((.6687942708333334, 1.1349236111111112), 23347)]:
            left = max(0, math.floor((bounds[0] - .04) * 16000))
            right = min(samples, math.ceil((bounds[1] + .04) * 16000))
            spectrum = torch.stft(torch.zeros(right - left), n_fft=512, hop_length=80,
                win_length=400, window=torch.hann_window(400), center=True, return_complex=True)
            centers = left + torch.arange(spectrum.shape[1], dtype=torch.float64) * 80
            actual = F.interpolate(centers[None, None], size=128, mode='linear', align_corners=False)[0, 0].numpy()
            np.testing.assert_allclose(query_grid(bounds, samples), actual, atol=1e-8, rtol=0)

    def test_phone_context_arm_never_reads_any_audio_channels(self):
        net = PhoneContextNetwork(10).eval()
        context, phone = torch.zeros(2, 2, dtype=torch.long), torch.zeros(2, 14)
        before = net(torch.randn(2, 66, 128), context, phone)
        after = net(torch.randn(2, 514, 128) * 100, context, phone)
        torch.testing.assert_close(before, after)

    def test_actual_conv_centers_resample_to_existing_target_clock(self):
        samples = 24000
        frames = (samples - 160) // 80 + 1
        centers = 79.5 + np.arange(frames, dtype=np.float32) * 80
        raw = np.broadcast_to(centers, (512, frames))
        bounds = (.3, 1.1)
        actual = interpolate_stem(raw, bounds, samples)
        np.testing.assert_allclose(actual[0], query_grid(bounds, samples), atol=.002)
        with self.assertRaises(ValueError):
            interpolate_stem(raw[:, ::4], bounds, samples)

    def test_dense_branch_has_no_word_loss_gradient_or_hint_input(self):
        net = StemVowelNetwork(10).eval()
        values = torch.randn(2, 514, 128)
        values[:, 512:] = .5
        context, phone = torch.zeros(2, 2, dtype=torch.long), torch.zeros(2, 14)
        logits, dense = net(values, context, phone, auxiliary=True)
        logits.sum().backward()
        self.assertTrue(all(p.grad is None for p in net.temporal.parameters()))
        self.assertTrue(all(p.grad is None for p in net.nucleus.parameters()))
        values[:, 512:] = .8
        _, changed = net(values, context, phone, auxiliary=True)
        torch.testing.assert_close(dense, changed)
