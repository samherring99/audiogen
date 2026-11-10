"""FSDP entry guards and rank-collective health checks; no CUDA required."""
import dataclasses
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from audiogen.training import sft


class DistributedTests(unittest.TestCase):
    def config(self, root, **kwargs):
        for name in ('train', 'val'):
            (root / (name + '.jsonl')).touch()
        return sft.SFTConfig(output_dir=str(root / 'out'), train_path=str(root / 'train.jsonl'),
                             val_path=str(root / 'val.jsonl'), fsdp='full_shard auto_wrap',
                             unfreeze_all=True, **kwargs)

    def test_optional_config(self):
        self.assertIn('fsdp', sft.SFTConfig.__dataclass_fields__)
        self.assertFalse(sft.SFTConfig().fsdp)
        self.assertIsNone(sft.SFTConfig().fsdp_config)

    def test_full_unfreeze_explicit_files_required(self):
        self.assertIn('fsdp', sft.SFTConfig.__dataclass_fields__)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'WORLD_SIZE': '2', 'LOCAL_RANK': '0'}):
            cfg = self.config(Path(tmp))
            sft._validate_config(cfg)
            for change, message in [({'fsdp': None}, 'distributed'),
                                    ({'unfreeze_all': False}, 'unfreeze_all'),
                                    ({'fsdp': 'shard_grad_op'}, 'full_shard auto_wrap'),
                                    ({'train_path': None, 'val_path': None, 'data_path': cfg.train_path}, 'explicit'),
                                    ({'fsdp_config': {'version': 2}}, 'version'),
                                    ({'fsdp_config': {'state_dict_type': 'SHARDED_STATE_DICT'}}, 'state_dict'),
                                    ({'fsdp_config': {'use_orig_params': False}}, 'use_orig_params'),
                                    ({'fsdp_config': {'cpu_ram_efficient_loading': True}}, 'cpu_ram')]:
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                    sft._validate_config(dataclasses.replace(cfg, **change))
        with patch.dict(os.environ, {'WORLD_SIZE': '1'}), self.assertRaisesRegex(ValueError, 'torchrun|ranks'):
            sft._validate_config(cfg)

    def test_cli_and_json_config(self):
        import scripts.train_sft as cli
        argv = ['train_sft', '--train_path', 'a', '--val_path', 'b', '--output_dir', 'out',
                '--unfreeze_all', '--fsdp', 'full_shard auto_wrap', '--fsdp_config', 'fsdp.json']
        with patch('sys.argv', argv), patch.object(cli, 'run_sft') as run:
            cli.main()
        self.assertEqual(run.call_args.args[0].fsdp, 'full_shard auto_wrap')
        self.assertEqual(run.call_args.args[0].fsdp_config, 'fsdp.json')

    def test_run_sft_requests_sharded_accumulation(self):
        from test_sft_integration import make_fixture
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.config(root, base_model=str(make_fixture(root)), dtype='float32',
                              bf16=False, num_codebooks=2, codebook_size=2)
            class CapturedArguments(Exception):
                pass
            def capture(**kwargs):
                self.assertEqual(kwargs.get('accelerator_config'),
                                 {'gradient_accumulation_kwargs': {'sync_each_batch': True}})
                raise CapturedArguments
            with patch.object(sft, '_validate_config'), patch('torch.cuda.set_device'), patch.dict(os.environ, {'LOCAL_RANK': '0'}), patch.object(sft, 'TrainingArguments', side_effect=capture):
                with self.assertRaises(CapturedArguments):
                    sft.run_sft(cfg)

    def test_fsdp_resume_requires_native_collective_files_and_same_world_size(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'WORLD_SIZE': '2', 'LOCAL_RANK': '0'}):
            root = Path(tmp)
            ckpt = root / 'ckpt'
            ckpt.mkdir()
            cfg = self.config(root, max_steps=3, trainer_resume_from_checkpoint=str(ckpt))
            sft._validate_config(cfg)
            for name in ('pytorch_model_fsdp.bin', 'optimizer.bin', 'scheduler.pt',
                         'rng_state_0.pth', 'rng_state_1.pth'):
                (ckpt / name).touch()
            (ckpt / 'trainer_state.json').write_text(json.dumps({'global_step': 1}))
            policy = {'config': dataclasses.asdict(cfg), 'world_size': 2}
            (ckpt / 'sft_run_policy.json').write_text(json.dumps(policy))
            sft._validate_resume(cfg)
            (ckpt / 'rng_state_1.pth').unlink()
            with self.assertRaisesRegex(ValueError, 'rng_state_1'):
                sft._validate_resume(cfg)
            (ckpt / 'rng_state_1.pth').touch()
            policy['world_size'] = 4
            (ckpt / 'sft_run_policy.json').write_text(json.dumps(policy))
            with self.assertRaisesRegex(ValueError, 'world_size'):
                sft._validate_resume(cfg)
            policy['world_size'] = 2
            policy['config']['fsdp'] = None
            (ckpt / 'sft_run_policy.json').write_text(json.dumps(policy))
            with self.assertRaisesRegex(ValueError, 'fsdp'):
                sft._validate_resume(cfg)

    def test_remote_bad_gradient_fails_every_rank(self):
        model = torch.nn.Linear(2, 2)
        model.weight.grad = torch.ones_like(model.weight)
        def remote_bad(flag, op):
            flag.fill_(1)
        with patch('torch.distributed.is_initialized', return_value=True), patch('torch.distributed.get_world_size', return_value=2), patch('torch.distributed.all_reduce', side_effect=remote_bad) as reduce:
            with self.assertRaisesRegex(FloatingPointError, 'Non-finite gradient'):
                sft._check_gradients(model)
            reduce.assert_called_once()

    def test_local_bad_gradient_collects_before_raising(self):
        model = torch.nn.Linear(2, 2)
        model.weight.grad = torch.full_like(model.weight, float('nan'))
        with patch('torch.distributed.is_initialized', return_value=True), patch('torch.distributed.get_world_size', return_value=2), patch('torch.distributed.all_reduce') as reduce:
            with self.assertRaisesRegex(FloatingPointError, 'Non-finite gradient'):
                sft._check_gradients(model)
            reduce.assert_called_once()

    def test_empty_local_shard_still_collects(self):
        model = torch.nn.Linear(2, 2)
        with patch('torch.distributed.is_initialized', return_value=True), patch('torch.distributed.get_world_size', return_value=2), patch('torch.distributed.all_reduce') as reduce:
            sft._check_gradients(model)
            reduce.assert_called_once()

    def test_fsdp_options_read_json_without_mutating_and_reject_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            supplied = {'use_orig_params': True, 'version': 1}
            path = root / 'fsdp.json'
            path.write_text(json.dumps(supplied))
            cfg = self.config(root, fsdp_config=str(path))
            result = sft._fsdp_options(cfg)
            self.assertEqual(result['transformer_layer_cls_to_wrap'], ['NemotronHBlock'])
            self.assertEqual(result['state_dict_type'], 'FULL_STATE_DICT')
            self.assertEqual(json.loads(path.read_text()), supplied)
            with self.assertRaisesRegex(ValueError, 'fsdp_version'):
                sft._fsdp_options(dataclasses.replace(cfg, fsdp_config={'fsdp_version': 2}))

    def test_trainer_forces_full_state_configs_over_launcher_environment(self):
        from types import SimpleNamespace
        from torch.distributed.fsdp import FullStateDictConfig, FullOptimStateDictConfig
        from accelerate import FullyShardedDataParallelPlugin
        # Constructor is GPU-free when sync_module_states is disabled for this unit test.
        plugin = FullyShardedDataParallelPlugin(fsdp_version=1, sync_module_states=False, state_dict_type='SHARDED_STATE_DICT')
        def fake_init(trainer, *args, **kwargs):
            trainer.is_fsdp_enabled = True
            trainer.accelerator = SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin))
        with patch.object(sft.Trainer, '__init__', fake_init), patch.object(sft.RobustTrainer, 'add_callback'):
            sft.RobustTrainer()
        self.assertIsInstance(plugin.state_dict_config, FullStateDictConfig)
        self.assertIsInstance(plugin.optim_state_dict_config, FullOptimStateDictConfig)
        self.assertTrue(plugin.state_dict_config.offload_to_cpu)
        self.assertTrue(plugin.optim_state_dict_config.rank0_only)

    def test_trainer_rejects_effective_environment_policy_before_wrapping(self):
        from types import SimpleNamespace
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp import ShardingStrategy
        for override, message in [({'FSDP_SHARDING_STRATEGY': 'NO_SHARD'}, 'FULL_SHARD'),
                                  ({'FSDP_OFFLOAD_PARAMS': 'true'}, 'offload')]:
            env = {'WORLD_SIZE': '2', 'LOCAL_RANK': '0',
                   'FSDP_SHARDING_STRATEGY': 'FULL_SHARD', 'FSDP_OFFLOAD_PARAMS': 'false',
                   **override}
            with self.subTest(override=override), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env):
                # The requested config is supported; only the effective plugin is unsafe.
                sft._validate_config(self.config(Path(tmp)))
                with patch('accelerate.utils.dataclasses.is_cuda_available', return_value=True), patch('torch.cuda.current_device', return_value=0):
                    plugin = FullyShardedDataParallelPlugin(fsdp_version=1, sync_module_states=True,
                                                           cpu_ram_efficient_loading=False)
                if 'FSDP_SHARDING_STRATEGY' in override:
                    self.assertEqual(plugin.sharding_strategy, ShardingStrategy.NO_SHARD)
                else:
                    self.assertTrue(plugin.cpu_offload.offload_params)
                def fake_init(trainer, *args, **kwargs):
                    trainer.is_fsdp_enabled = True
                    trainer.accelerator = SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin))
                with patch.object(sft.Trainer, '__init__', fake_init), patch.object(sft.RobustTrainer, 'add_callback'):
                    with self.assertRaisesRegex(ValueError, message):
                        sft.RobustTrainer()

    def test_trainer_preserves_fp32_buffers_without_changing_compute_policy(self):
        from types import SimpleNamespace
        from accelerate import FullyShardedDataParallelPlugin
        plugin = FullyShardedDataParallelPlugin(fsdp_version=1, sync_module_states=False)
        plugin.set_mixed_precision('bf16')
        # Non-default flags must survive: rebuilding only the three dtypes is unsafe.
        original = dataclasses.replace(plugin.mixed_precision_policy,
                                       cast_forward_inputs=True, keep_low_precision_grads=True)
        plugin.mixed_precision_policy = original
        def fake_init(trainer, *args, **kwargs):
            trainer.is_fsdp_enabled = True
            trainer.accelerator = SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin))
        with patch.object(sft.Trainer, '__init__', fake_init), patch.object(sft.RobustTrainer, 'add_callback'):
            sft.RobustTrainer()
        policy = plugin.mixed_precision_policy
        self.assertEqual(policy.buffer_dtype, torch.float32)
        for field in dataclasses.fields(original):
            if field.name != 'buffer_dtype':
                self.assertEqual(getattr(policy, field.name), getattr(original, field.name))
        bias = torch.tensor([0.0010001, 0.0010002], dtype=torch.float32)
        self.assertFalse(torch.equal(bias, bias.bfloat16().float()))
        self.assertTrue(torch.equal(bias, bias.to(policy.buffer_dtype)))

    def test_trainer_leaves_full_precision_fsdp_policy_disabled(self):
        from types import SimpleNamespace
        from accelerate import FullyShardedDataParallelPlugin
        plugin = FullyShardedDataParallelPlugin(fsdp_version=1, sync_module_states=False)
        self.assertIsNone(plugin.mixed_precision_policy)
        def fake_init(trainer, *args, **kwargs):
            trainer.is_fsdp_enabled = True
            trainer.accelerator = SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin))
        with patch.object(sft.Trainer, '__init__', fake_init), patch.object(sft.RobustTrainer, 'add_callback'):
            sft.RobustTrainer()
        self.assertIsNone(plugin.mixed_precision_policy)

    def test_remote_nonfinite_loss_collective(self):
        from types import SimpleNamespace
        trainer = object.__new__(sft.RobustTrainer)
        trainer.state = SimpleNamespace(global_step=0)
        def remote_bad(flag, op):
            flag.fill_(1)
        with patch.object(sft.Trainer, 'compute_loss', return_value=torch.tensor(1.0)), patch('torch.distributed.is_initialized', return_value=True), patch('torch.distributed.get_world_size', return_value=2), patch('torch.distributed.all_reduce', side_effect=remote_bad):
            with self.assertRaisesRegex(FloatingPointError, 'Non-finite loss'):
                trainer.compute_loss(None, {})

    def test_global_norm_overflow_is_detected(self):
        self.assertTrue(hasattr(sft, '_check_gradient_norm'))
        model = torch.nn.Linear(1, 1)
        model.weight.grad = torch.ones_like(model.weight)
        def overflow(norm, op):
            norm.fill_(float('inf'))
        with patch('torch.distributed.is_initialized', return_value=True), patch('torch.distributed.get_world_size', return_value=2), patch('torch.distributed.all_reduce', side_effect=overflow):
            with self.assertRaisesRegex(FloatingPointError, 'aggregate gradient norm'):
                sft._check_gradient_norm(model)


if __name__ == '__main__':
    unittest.main()
