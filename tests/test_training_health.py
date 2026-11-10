import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import TrainingArguments

from audiogen.training.sft import RobustTrainer


class LossModel(torch.nn.Module):
    def __init__(self, value=1.0, bad_gradient=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value))
        if bad_gradient:
            self.weight.register_hook(lambda grad: torch.full_like(grad, float('inf')))

    def forward(self, input_ids=None, labels=None):
        return {'loss': self.weight.square(), 'logits': self.weight.reshape(1, 1)}


class HealthTests(unittest.TestCase):
    def trainer(self, model, output):
        return RobustTrainer(model=model, args=TrainingArguments(output_dir=output, use_cpu=True, report_to='none', max_steps=1, per_device_train_batch_size=1, save_strategy='no', disable_tqdm=True), train_dataset=[{'input_ids': [1], 'labels': [1]}])

    def test_nonfinite_loss_raises_before_backward(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = LossModel(float('nan'))
            trainer = self.trainer(model, tmp)
            with patch.object(trainer.accelerator, 'backward', wraps=trainer.accelerator.backward) as backward:
                with self.assertRaisesRegex(FloatingPointError, 'loss'):
                    trainer.train()
                backward.assert_not_called()
            self.assertIsNone(model.weight.grad)

    def test_nonfinite_gradient_raises_before_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = LossModel(bad_gradient=True)
            trainer = self.trainer(model, tmp)
            with self.assertRaisesRegex(FloatingPointError, 'gradient'):
                trainer.train()
            self.assertEqual(model.weight.item(), 1.0)
            self.assertEqual(trainer.state.global_step, 0)

    def test_finite_gradients_with_overflowing_norm_abort_before_clipping(self):
        # Cover overflow within one tensor and only while combining tensor norms.
        for split_parameters in (False, True):
            with self.subTest(split_parameters=split_parameters), tempfile.TemporaryDirectory() as tmp:
                class OverflowNormModel(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        shapes = [(1,), (1,)] if split_parameters else [(2,)]
                        self.weights = torch.nn.ParameterList([
                            torch.nn.Parameter(torch.ones(shape)) for shape in shapes
                        ])
                        for weight in self.weights:
                            weight.register_hook(lambda grad: torch.full_like(grad, 1e30))

                    def forward(self, input_ids=None, labels=None):
                        return {'loss': sum(w.square().mean() for w in self.weights) / len(self.weights)}

                model = OverflowNormModel()
                self.assertEqual(model()['loss'].item(), 1.0)
                trainer = self.trainer(model, tmp)
                with patch.object(torch.nn.utils, 'clip_grad_norm_', wraps=torch.nn.utils.clip_grad_norm_) as clip:
                    with self.assertRaisesRegex(FloatingPointError, 'gradient.*norm'):
                        trainer.train()
                    clip.assert_not_called()
                self.assertEqual(trainer.state.global_step, 0)
                for weight in model.weights:
                    torch.testing.assert_close(weight, torch.ones_like(weight), rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(weight.grad).all())
                    torch.testing.assert_close(weight.grad, torch.full_like(weight, 1e30), rtol=0, atol=0)

    def test_finite_accumulated_gradients_are_not_modified_before_single_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = LossModel()
            trainer = self.trainer(model, tmp)
            trainer.args.gradient_accumulation_steps = 2
            trainer.train_dataset *= 2
            gradients_before_clip = []
            original_clip = torch.nn.utils.clip_grad_norm_

            def record_clip(parameters, *args, **kwargs):
                gradients_before_clip.append(model.weight.grad.detach().clone())
                return original_clip(parameters, *args, **kwargs)

            with patch.object(torch.nn.utils, 'clip_grad_norm_', side_effect=record_clip) as clip:
                trainer.train()
                clip.assert_called_once()
            torch.testing.assert_close(gradients_before_clip[0], torch.tensor(2.0), rtol=0, atol=0)
            self.assertEqual(trainer.state.global_step, 1)
            self.assertLess(model.weight.item(), 1.0)

    def test_finite_training_really_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = LossModel()
            trainer = self.trainer(model, tmp)
            trainer.train()
            self.assertEqual(trainer.state.global_step, 1)
            self.assertLess(model.weight.item(), 1.0)


if __name__ == '__main__':
    unittest.main()
