import unittest

import numpy as np
import torch
import torch.nn.functional as F

from hindi_detector.nucleus_deletion import NucleusDeletionNetwork, supervision


class NucleusSupervisionTests(unittest.TestCase):
    def row(self, family, shift=0):
        return {'family': family, 'source_region': {'start_sample': 8000, 'end_sample': 9600},
                'production_word': {'start': .3 + shift / 16000, 'end': .9 + shift / 16000},
                'edit': {'start_sample': 8000, 'end_sample': 9600, 'replacement_samples': 1600}}

    def test_retained_and_removed_targets_only_supervise_local_frames(self):
        kept = self.row('S0')
        target, mask = supervision(kept, 24000)
        self.assertTrue(np.all(target == 1))
        self.assertGreater(mask.sum(), 0)
        self.assertLess(mask.sum(), 30)
        removed = self.row('P1')
        removed['edit']['replacement_samples'] = 0
        target, mask = supervision(removed, 22400)
        self.assertTrue(np.all(target == 0))
        self.assertGreater(mask.sum(), 0)

    def test_off_target_cut_moves_loss_labels_with_the_actual_word(self):
        _, original = supervision(self.row('S_original'), 24000)
        moved = self.row('S1', -1600)
        moved['edit'] = {'start_sample': 1600, 'end_sample': 3200, 'replacement_samples': 0}
        _, after = supervision(moved, 22400)
        # Float word-boundary subtraction can move the sample-floor crop by one
        # sample. Compare event centers, not impossible bit-exact mask fractions.
        grid = np.arange(128)
        self.assertLess(abs(float(np.dot(after, grid) / after.sum() - np.dot(original, grid) / original.sum())), .025)
        self.assertLess(float(np.max(np.abs(after - original))), .02)

    def test_increased_local_vowel_evidence_can_only_reduce_deletion_logit(self):
        torch.manual_seed(82)
        net = NucleusDeletionNetwork(10).eval()
        mel = torch.randn(2, 66, 128)
        mel[:, 64:] = torch.sigmoid(mel[:, 64:])
        context = torch.zeros((2, 2), dtype=torch.long)
        phone = torch.zeros(2, 14)
        before = net(mel, context, phone)
        with torch.no_grad():
            net.nucleus[-1].bias.add_(1.)
        after = net(mel, context, phone)
        expected = -F.softplus(net.vowel_veto).detach()
        torch.testing.assert_close(after - before, torch.full_like(before, expected.item()), atol=1e-5, rtol=1e-5)
        with self.assertRaises(TypeError):
            net(mel, context, phone, edit={'start_sample': 0})

    def test_word_labels_cannot_train_dense_vowel_head(self):
        net = NucleusDeletionNetwork(10)
        mel = torch.randn(2, 66, 128)
        mel[:, 64:] = .5
        context = torch.zeros((2, 2), dtype=torch.long)
        net(mel, context, torch.zeros(2, 14)).sum().backward()
        self.assertTrue(all(p.grad is None for p in net.temporal.parameters()))
        self.assertTrue(all(p.grad is None for p in net.nucleus.parameters()))

    def test_dense_vowel_head_cannot_read_phone_location_hints(self):
        net = NucleusDeletionNetwork(10).eval()
        mel = torch.randn(2, 66, 128)
        mel[:, 64:] = .5
        context, phone = torch.zeros((2, 2), dtype=torch.long), torch.zeros(2, 14)
        _, before = net(mel, context, phone, auxiliary=True)
        mel[:, 64:] = .9
        _, after = net(mel, context, phone, auxiliary=True)
        torch.testing.assert_close(before, after)
