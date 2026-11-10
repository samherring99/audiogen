import unittest
from transformers import TrainerControl, TrainerState
from audiogen.training import sft


class PlateauTests(unittest.TestCase):
    def test_warmup_patience_and_clean_save_stop(self):
        self.assertTrue(hasattr(sft, 'PlateauCallback'), 'explicit plateau policy missing')
        callback = sft.PlateauCallback(patience=2, min_steps=5, warmup_steps=3, min_delta=0.1)
        control = TrainerControl()
        for step, loss in [(1, 9.0), (3, 9.0), (5, 2.0), (6, 1.95)]:
            callback.on_evaluate(None, TrainerState(global_step=step), control, metrics={'eval_loss': loss})
            self.assertFalse(control.should_training_stop)
        callback.on_evaluate(None, TrainerState(global_step=7), control, metrics={'eval_loss': 1.94})
        self.assertTrue(control.should_training_stop)
        self.assertTrue(control.should_save)
        restored = sft.PlateauCallback.from_state(callback.state())
        self.assertEqual(restored.state(), callback.state())

    def test_disabled_by_default(self):
        self.assertEqual(getattr(sft.SFTConfig(data_path='x', output_dir='x'), 'plateau_patience', None), 0)


if __name__ == '__main__':
    unittest.main()
