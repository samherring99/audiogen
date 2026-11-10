import unittest

import torch


class ObjectiveTests(unittest.TestCase):
    def api(self):
        from audiogen.training.grpo_objective import group_advantages, completion_loss
        return group_advantages, completion_loss

    def test_groups_once_zero_variance_and_repeated_ids(self):
        groups, _ = self.api()
        actual = groups(torch.tensor([1., 3., 2., 2.]), [0, 0, 1, 1])
        torch.testing.assert_close(actual, torch.tensor([-1., 1., 0., 0.]))

    def test_groups_reject_singletons_and_nan(self):
        groups, _ = self.api()
        with self.assertRaises(ValueError):
            groups(torch.tensor([1., 2.]), [0, 1])
        with self.assertRaises(ValueError):
            groups(torch.tensor([1., float('nan')]), [0, 0])

    def test_on_policy_gradient_and_zero_kl(self):
        _, loss_fn = self.api()
        lp = torch.tensor([-2., -3.], requires_grad=True)
        loss, kl = loss_fn(lp, lp.detach(), lp.detach(), 1., beta=.02, epsilon=.2)
        self.assertAlmostEqual(loss.item(), -1.)
        self.assertEqual(kl.item(), 0.)
        loss.backward()
        torch.testing.assert_close(lp.grad, torch.tensor([-.5, -.5]))

    def test_clipping_both_signs(self):
        _, loss_fn = self.api()
        for advantage, shift in [(1., .5), (-1., -.5)]:
            lp = torch.tensor([-2. + shift], requires_grad=True)
            loss, _ = loss_fn(lp, torch.tensor([-2.]), lp.detach(), advantage, beta=0., epsilon=.2)
            loss.backward()
            self.assertEqual(lp.grad.item(), 0.)
            self.assertAlmostEqual(loss.item(), -1.2 if advantage > 0 else .8, places=6)

    def test_k3_nonnegative_on_unpadded_completion(self):
        _, loss_fn = self.api()
        lp = torch.tensor([-2., -3.], requires_grad=True)
        _, kl = loss_fn(lp, lp.detach(), torch.tensor([-1.5, -3.]), 0., beta=1., epsilon=.2)
        expected = (torch.exp(torch.tensor(.5)) - .5 - 1.) / 2
        torch.testing.assert_close(kl, expected)
        self.assertGreaterEqual(kl.item(), 0.)

    def test_streamed_backward_matches_batch_average(self):
        _, loss_fn = self.api()
        def calc(stream):
            p = torch.tensor(.1, requires_grad=True)
            terms = []
            for length, advantage in [(2, -1.), (5, 1.)]:
                lp = -2. + p * torch.arange(1, length+1).float()
                term, _ = loss_fn(lp, torch.full((length,), -2.), torch.full((length,), -2.), advantage, .02, .2)
                if stream:
                    (term / 2).backward()
                else:
                    terms.append(term)
            if not stream:
                torch.stack(terms).mean().backward()
            return p.grad
        torch.testing.assert_close(calc(True), calc(False))

    def test_nonfinite_and_shape_rejected(self):
        _, loss_fn = self.api()
        for lp in (torch.tensor([]), torch.tensor([float('nan')])):
            with self.assertRaises(ValueError):
                loss_fn(lp, lp, lp, 1.)

    def test_rejects_positive_logprob(self):
        _, loss_fn = self.api()
        with self.assertRaises(ValueError):
            loss_fn(torch.tensor([.1]), torch.tensor([-.1]), torch.tensor([-.1]), 1.)

    def test_ratio_overflow_rejected_before_backward(self):
        _, loss_fn = self.api()
        with self.assertRaises(ValueError):
            loss_fn(torch.tensor([-1.], requires_grad=True), torch.tensor([-100.]), torch.tensor([-1.]), 1., beta=0.)

    def test_advantages_do_not_backprop_into_rewards(self):
        groups, _ = self.api()
        rewards = torch.tensor([1., 2., 3., 5.], requires_grad=True)
        self.assertFalse(groups(rewards, [0, 0, 1, 1]).requires_grad)

    def test_old_reference_and_advantage_detached(self):
        _, loss_fn = self.api()
        old = torch.tensor([-2.], requires_grad=True)
        ref = torch.tensor([-1.], requires_grad=True)
        adv = torch.tensor(.5, requires_grad=True)
        lp = torch.tensor([-1.9], requires_grad=True)
        loss, _ = loss_fn(lp, old, ref, adv)
        loss.backward()
        self.assertIsNone(old.grad)
        self.assertIsNone(ref.grad)
        self.assertIsNone(adv.grad)
        self.assertIsNotNone(lp.grad)


if __name__ == '__main__':
    unittest.main()
