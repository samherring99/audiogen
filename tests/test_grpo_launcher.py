"""CPU-only launcher contract tests; never load production weights."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / 'experiments/nemotron35_tts_2026-09-18/launch_grpo.py'

pytestmark = pytest.mark.skipif(
    not PATH.is_file(),
    reason='experiment harness not present (experiments/ is not part of this repo)',
)


def launcher():
    assert PATH.exists(), 'GRPO launcher implementation is missing'
    spec = importlib.util.spec_from_file_location('grpo_launcher_contract', PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_has_no_torch_or_wandb():
    launcher()
    code = f"import runpy,sys; runpy.run_path({str(PATH)!r},run_name='contract'); assert 'torch' not in sys.modules; assert 'wandb' not in sys.modules"
    subprocess.run([sys.executable, '-c', code], check=True)


def test_fixed_budget_and_fresh_parent():
    m = launcher()
    smoke, pilot = m.run_config('smoke', 'v1'), m.run_config('pilot', 'v1')
    assert smoke['steps'] == 2 and pilot['steps'] == 25
    assert pilot['save_steps'] == [5, 10, 25]
    assert smoke['parent'] == pilot['parent'] == str(m.PARENT)
    assert pilot['temperature'] == pilot['top_p'] == 1.0
    assert pilot['learning_rate'] == 1e-7 and pilot['world_size'] == 4
    assert pilot['groups'] * pilot['rollouts'] == 8
    for tag in ('../bad', '', 'v1/evil'):
        with pytest.raises(ValueError):
            m.run_config('pilot', tag)


def test_reservation_exclusive_and_no_dryrun_writes(tmp_path):
    m = launcher()
    path = tmp_path / 'run-v1'
    m.reserve_output(path, {'binding': 'abc'})
    assert json.loads((path / 'launch.json').read_text()) == {'binding': 'abc'}
    with pytest.raises(FileExistsError):
        m.reserve_output(path, {})


def test_prompt_selection_bounded_train_only_and_mixed(tmp_path):
    m = launcher()
    path = tmp_path / 'train.tagged.jsonl'
    text = 'We reached the station before sunset and waited quietly for the evening train to arrive.'
    rows = [{'prompt': 'Generate speech audio [gender=female]: ' + text},
            {'prompt': 'Generate speech audio: ' + text.replace('station', 'harbour')}]
    path.write_text('\n'.join(map(json.dumps, rows)) + '\nTHIS MUST NOT BE READ\n')
    prompts = m.select_prompts(path, per_kind=1, max_lines=2)
    assert len(prompts) == 2 and {p['tagged'] for p in prompts} == {True, False}
    assert [p['line'] for p in prompts] == [1, 2]
    with pytest.raises(RuntimeError, match='mixed'):
        m.select_prompts(path, per_kind=2, max_lines=2)


def test_environment_fails_closed():
    m = launcher()
    env = dict(WORLD_SIZE='4', LOCAL_WORLD_SIZE='4', RANK='0', LOCAL_RANK='0', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
    m.validate_environment(env, m.EXP / '.venv-nemotron')
    for key, value in [('WORLD_SIZE', '1'), ('CUDA_VISIBLE_DEVICES', '0,1,2,3'), ('RANK', '1')]:
        with pytest.raises(RuntimeError):
            m.validate_environment({**env, key: value}, m.EXP / '.venv-nemotron')


def test_smoke_receipt_must_bind_real_evidence(tmp_path):
    m = launcher()
    path = tmp_path / 'smoke.json'
    valid = dict(kind='native_grpo_smoke', passed=True, binding='abc', steps=2,
                 parameter_delta_max=1e-7, reload_logprob_max_error=0.0,
                 reload_logits_max_error=0.0, checkpoint=str(tmp_path / 'checkpoint-2'),
                 cached_likelihood=dict(tokens=8, max_error=0., tolerance=.05),
                 old_logprob_tolerance=.05, code_sha256={'launcher': 'test-hash'},
                 telemetry=[dict(per_rank=[dict(grad_norm=1., old_logprob_max_abs_diff=0.,
                                               gradient_coverage={'embedding': {}}, cuda_memory={'device': 0})]*4,
                                 rank0_devices={str(i): {} for i in range(8)})]*2)
    ckpt = tmp_path / 'checkpoint-2'
    ckpt.mkdir()
    (ckpt / 'COMPLETE.json').write_text(json.dumps({'step': 2, 'resume_supported': False, 'checkpoint_kind': 'weights_only', 'run_policy': {'binding': 'abc'}}))
    path.write_text(json.dumps(valid))
    m.validate_smoke_receipt(path, 'abc')
    for changes in ({'binding': 'other'}, {'passed': False}, {'steps': 1}, {'parameter_delta_max': 0}, {'parameter_delta_max': float('nan')}, {'reload_logprob_max_error': 1}, {'reload_logits_max_error': float('nan')}):
        path.write_text(json.dumps({**valid, **changes}))
        with pytest.raises(RuntimeError):
            m.validate_smoke_receipt(path, 'abc')
    path.write_text(json.dumps(valid))
    (ckpt / 'COMPLETE.json').write_text(json.dumps({'step': 2, 'resume_supported': False, 'checkpoint_kind': 'weights_only', 'run_policy': {'binding': 'stale'}}))
    with pytest.raises(RuntimeError):
        m.validate_smoke_receipt(path, 'abc')


def test_variance_gate_rejects_flat_or_nonfinite():
    m = launcher()
    assert m.require_variance([.1, .2, .3, .4, -1, .1, .2, .3], [0]*4+[1]*4) > 0
    for rewards in ([.1]*8, [float('nan')]*8):
        with pytest.raises(RuntimeError):
            m.require_variance(rewards, [0]*4+[1]*4)


def test_rank0_error_propagates_on_control_channel():
    m = launcher()
    class Control:
        def broadcast_object_list(self, payload, src, group):
            self.payload = payload
    control = Control()
    def fail():
        raise ValueError('decoder failed')
    with pytest.raises(RuntimeError, match='decoder failed'):
        m.rank0_call(control, 'gloo', 0, 'rollout', fail)
    assert control.payload[0]['ok'] is False


def test_peer_receives_rank0_error_without_running_function():
    m = launcher()
    class Control:
        def broadcast_object_list(self, payload, src, group):
            assert group == 'gloo'
            payload[0] = {'ok': False, 'error': 'rank0 save failed'}
    with pytest.raises(RuntimeError, match='rank0 save failed'):
        m.rank0_call(Control(), 'gloo', 3, 'save', lambda: pytest.fail('peer ran rank0 work'))


def test_execute_flag_still_requires_real_gates():
    result = subprocess.run([sys.executable, str(PATH), '--execute'], text=True, capture_output=True)
    assert result.returncode != 0 and 'requires --calibration-report' in result.stderr


def test_calibration_requires_all_checks_and_audio_evidence(tmp_path):
    m = launcher()
    report = tmp_path / 'report.json'
    scores = tmp_path / 'scores.jsonl'
    valid = {'passed': True, 'model': 'openai/whisper-base', 'checks': {key: True for key in m.CALIBRATION_CHECKS}}
    report.write_text(json.dumps(valid))
    scores.write_text(json.dumps({'reward': .6, 'path': 'real.wav', 'transcription': 'real speech'}) + '\n')
    assert len(m.validate_calibration(report, 'openai/whisper-base')['report_sha256']) == 64
    for checks in ({}, {key: 1 for key in m.CALIBRATION_CHECKS}):
        report.write_text(json.dumps({**valid, 'checks': checks}))
        with pytest.raises(RuntimeError):
            m.validate_calibration(report, 'openai/whisper-base')
    report.write_text(json.dumps(valid))
    scores.write_text('')
    with pytest.raises(RuntimeError):
        m.validate_calibration(report, 'openai/whisper-base')


@pytest.mark.parametrize('numel', [374347008, 2**24 + 4, 2**32])
def test_sample_state_production_cardinality_without_large_allocation(numel):
    import torch
    m = launcher()
    # A zero-stride view exercises real indexing with only one scalar of storage.
    scalar = torch.tensor(1.25)
    weight = scalar.expand(numel)
    assert weight.untyped_storage().nbytes() == scalar.element_size()
    sampled = m.sample_state({'model.weight': weight})['model.weight']
    assert sampled.shape == (1024,)
    assert sampled.dtype == torch.float32 and sampled.device.type == 'cpu'
    assert torch.equal(sampled, torch.full((1024,), 1.25))
    scalar.fill_(2.5)
    assert torch.equal(sampled, torch.full((1024,), 1.25))


@pytest.mark.parametrize('numel', [0, 1, 2, 1023, 1024, 1025, 4096,
                                 374347008, 2**53 + 2, 2**63 - 1])
def test_sample_indices_exact_bounded_unique_and_endpoint_inclusive(numel):
    import torch
    m = launcher()
    indices = m._sample_indices(numel)
    count = min(numel, 1024)
    assert indices.dtype == torch.long and indices.device.type == 'cpu'
    assert indices.shape == (count,)
    values = indices.tolist()
    assert values == m._sample_indices(numel).tolist()
    if not count:
        return
    assert values[0] == 0 and values[-1] == numel - 1
    assert all(0 <= index < numel for index in values)
    assert all(left < right for left, right in zip(values, values[1:]))
    if numel <= 1024:
        assert values == list(range(numel))
    else:
        # Exact integer interpolation, including beyond float64 and int64-product limits.
        assert all(index * (count - 1) <= i * (numel - 1) < (index + 1) * (count - 1)
                   for i, index in enumerate(values))


def test_sample_indices_rejects_negative_cardinality():
    with pytest.raises(ValueError, match='nonnegative'):
        launcher()._sample_indices(-1)


def test_cpu_sample_delta_and_sampling_config():
    import torch
    from types import SimpleNamespace
    from transformers import GenerationConfig
    m = launcher()
    state = {'model.weight': torch.zeros(4096), 'small.weight': torch.ones(2)}
    before = m.sample_state(state)
    state['model.weight'].add_(1e-7)
    assert 0 < m.sample_delta(before, m.sample_state(state)) < 1e-6
    state['model.weight'].fill_(float('nan'))
    with pytest.raises(RuntimeError, match='Nonfinite'):
        m.sample_delta(before, m.sample_state(state))
    class Model:
        config = SimpleNamespace(use_cache=False)
        generation_config = GenerationConfig(top_k=50, repetition_penalty=1.2)
        def eval(self):
            return self
        def requires_grad_(self, value):
            self.grad = value
            return self
    model = Model()
    m.configure_rollout(model)
    assert model.generation_config.top_k == 0
    assert model.generation_config.top_p == 1
    assert model.generation_config.repetition_penalty == 1
    assert model.grad is False


def test_scatter_assignment_is_equal_and_preserves_all_groups():
    m = launcher()
    rows = list(range(8))
    assert m.partition_completions(rows) == [[0, 4], [1, 5], [2, 6], [3, 7]]
    with pytest.raises(RuntimeError):
        m.partition_completions(rows[:-1])


def test_portable_disk_budget_uses_actual_parent_precision():
    m = launcher()
    assert m.portable_bytes({'weight_bytes': 100, 'dtype': 'float32'}) == 100
    assert m.portable_bytes({'weight_bytes': 100, 'dtype': 'bfloat16'}) == 200
    with pytest.raises(RuntimeError):
        m.portable_bytes({'weight_bytes': 100, 'dtype': 'int8'})


def test_default_cli_is_dry_run(tmp_path):
    m = launcher()
    # A dry plan does not require a receipt and cannot reserve or import GPU code.
    result = subprocess.run([sys.executable, str(PATH), '--mode', 'pilot', '--tag', 'vcontract-dry'], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan['execute'] is False and plan['gates_required']
    assert not Path(plan['config']['output']).exists()


class TinyCachedModel:
    """Raw logits with observable cache usage, no production model dependency."""
    def __init__(self, cached_bias=0.):
        self.calls = []
        self.cached_bias = cached_bias

    def __call__(self, input_ids, attention_mask, use_cache, logits_to_keep=0, past_key_values=None):
        import torch
        from types import SimpleNamespace
        assert not torch.is_grad_enabled()
        self.calls.append((input_ids.shape[1], use_cache, past_key_values))
        total = input_ids.cumsum(1) + (past_key_values or 0)
        logits = torch.arange(16).float()[None, None, :] * total[:, :, None] / 100
        if use_cache:
            logits[..., 0] += self.cached_bias
        return SimpleNamespace(logits=logits, past_key_values=int(total[0, -1]))


def test_cached_replay_is_bounded_incremental_raw_and_fail_closed():
    import torch
    from audiogen.training.grpo_native import completion_token_logprobs
    m = launcher()
    model = TinyCachedModel()
    ids = torch.tensor([1, 2] + [3, 4] * 8)
    result = m.verify_cached_logprobs(model, ids, 2, completion_token_logprobs, tolerance=.05, max_tokens=8)
    assert result['tokens'] == 8 and result['max_error'] < 1e-6
    assert result['tolerance'] == .05
    assert model.calls[0][:2] == (10, False)
    assert [n for n, cached, _ in model.calls if cached] == [2] + [1] * 7
    with pytest.raises(RuntimeError, match='cached.*likelihood'):
        m.verify_cached_logprobs(TinyCachedModel(50), ids, 2, completion_token_logprobs, tolerance=.05)


def test_config_binds_likelihood_and_live_listening_policy():
    m = launcher()
    config = m.run_config('pilot', 'v1')
    assert config['old_logprob_tolerance'] == .05
    assert config['cached_logprob_tolerance'] == .05
    assert config['cached_verify_tokens'] == 8
    assert config['approval_timeout_seconds'] >= 86400
    assert config['control_timeout_seconds'] > config['approval_timeout_seconds']
    cases = m.listening_prompts()
    assert len(cases) == 9
    assert len([c for c in cases if c['kind'] == 'general']) == 4
    pair = [c['prompt'].split(': ', 1)[1] for c in cases if c['kind'] == 'gender']
    assert len(pair) == 2 and pair[0] == pair[1]
    assert all(any(f'[accent={accent}]' in c['prompt'] for c in cases)
               for accent in ('us', 'england', 'india_south_asia'))


def test_approval_wait_is_explicit_bound_and_never_auto_continues(tmp_path):
    m = launcher()
    ticks = iter([0., 0., 2.])
    with pytest.raises(TimeoutError, match='approval'):
        m.wait_for_listening_approval(tmp_path, 5, 'binding', timeout_seconds=1,
                                      clock=lambda: next(ticks), sleep=lambda _: None)
    marker = tmp_path / 'APPROVE-step-5.json'
    marker.write_text(json.dumps({'step': 5, 'binding': 'stale', 'approved': True}))
    with pytest.raises(RuntimeError, match='approval'):
        m.wait_for_listening_approval(tmp_path, 5, 'binding', timeout_seconds=1)
    marker.write_text(json.dumps({'step': 5, 'binding': 'binding', 'approved': True}))
    assert m.wait_for_listening_approval(tmp_path, 5, 'binding', timeout_seconds=1)['approved'] is True


@pytest.fixture
def execution_harness(monkeypatch, tmp_path):
    """Execute the real coordinator; substitute only model/GPU/network boundaries."""
    import torch
    import torch.distributed as dist
    from types import SimpleNamespace, ModuleType
    from contextlib import nullcontext
    from audiogen.training import grpo_native as rt
    from audiogen.tokens import formatting
    m = launcher()
    events = []
    configs = []
    state = {'model.weight': torch.zeros(1024)}
    checkpoint_state = {}
    config = m.run_config('smoke', 'vtest')
    config['output'] = str(tmp_path / 'run')
    prompts = [dict(prompt=f'Generate speech audio: Training-only sentence {i}', tagged=bool(i % 2)) for i in range(64)]
    manifest = dict(config=config, binding='test-binding', prompts=prompts, code_sha256={'launcher': 'test-hash'})
    monkeypatch.setattr(m, 'validate_environment', lambda *a: None)
    monkeypatch.setattr(m, 'admission', lambda *a: manifest)
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setattr(torch.cuda, 'set_device', lambda rank: events.append(('set_device', rank)))
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 8)
    monkeypatch.setattr(torch.cuda, 'device', lambda *a: nullcontext())
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    for name, value in [('memory_allocated', 10), ('memory_reserved', 20),
                        ('max_memory_allocated', 30), ('max_memory_reserved', 40)]:
        monkeypatch.setattr(torch.cuda, name, lambda *a, value=value: value)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda *a: (50, 100))
    original_to = torch.Tensor.to
    def cpu_to(self, *args, **kwargs):
        if args and str(args[0]).startswith('cuda'):
            args = ('cpu', *args[1:])
        return original_to(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', cpu_to)
    monkeypatch.setattr(dist, 'init_process_group', lambda *a, **kw: events.append(('init', kw)))
    monkeypatch.setattr(dist, 'new_group', lambda **kw: (events.append(('control', kw)) or 'gloo'))
    monkeypatch.setattr(dist, 'broadcast_object_list', lambda payload, **kw: events.append(('broadcast', payload[0])))
    def gather(out, obj, **kw):
        out[:] = [obj] * 4
    monkeypatch.setattr(dist, 'all_gather_object', gather)
    def scatter(out, chunks, **kw):
        events.append(('scatter', chunks))
        out[0] = chunks[0]
    monkeypatch.setattr(dist, 'scatter_object_list', scatter)
    monkeypatch.setattr(dist, 'destroy_process_group', lambda: events.append(('destroy',)))
    tokenizer = SimpleNamespace(get_vocab=lambda: {'a': 1})
    metadata = SimpleNamespace(codec='moss', num_codebooks=8)
    class Model(TinyCachedModel):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()
            self.audiogen_loading_info = {}
            self.weights = {'model.weight': state['model.weight'].clone()}
        def state_dict(self):
            return self.weights
        def eval(self):
            return self
        def requires_grad_(self, _):
            return self
    def load(path, device, dtype):
        events.append(('load', str(path), device))
        model = Model()
        if str(path) in checkpoint_state:
            model.weights = checkpoint_state[str(path)]
        return model, tokenizer, metadata
    setup = ModuleType('audiogen.model.setup')
    setup.load_trained_model = load
    monkeypatch.setitem(sys.modules, 'audiogen.model.setup', setup)
    monkeypatch.setattr(rt, 'wrap_native_policy', lambda model, device: SimpleNamespace(module=model, parameters=lambda: [torch.nn.Parameter(torch.zeros(1))]))
    monkeypatch.setattr(rt, 'gather_cpu_state', lambda _: state)
    def copy(model, value):
        events.append(('sync', float(value['model.weight'][0])))
        model.weights = {k: v.clone() for k, v in value.items()}
    monkeypatch.setattr(rt, 'copy_cpu_state_to_model', copy)
    def update(policy, optimizer, rows, **kw):
        events.append(('update', rows, kw))
        state['model.weight'].add_(.001)
        return dict(loss=.1, grad_norm=1., old_logprob_max_abs_diff=.001,
                    gradient_coverage={'embedding': {'present': 1, 'nonzero': 1}},
                    memory_allocated_bytes=10)
    monkeypatch.setattr(rt, 'native_grpo_update', update)
    def save(policy, tokenizer, metadata, path, **kw):
        events.append(('save', kw['step']))
        path.mkdir()
        checkpoint_state[str(path)] = {k: v.clone() for k, v in state.items()}
        m.write_json(path / 'COMPLETE.json', dict(kw, resume_supported=False, checkpoint_kind='weights_only'))
    monkeypatch.setattr(rt, 'save_portable_checkpoint', save)
    monkeypatch.setattr(formatting, 'get_formatter', lambda *a: None)
    monkeypatch.setattr(formatting, 'prompt_policy_from_config', lambda *a: ('speech_v1', {}))
    monkeypatch.setattr(formatting, 'build_audio_prefix', lambda *a: [1, 2])
    monkeypatch.setattr(rt, 'build_audio_prefix', lambda *a: [1, 2])
    monkeypatch.setattr(rt, 'prompt_policy_from_config', lambda *a: ('speech_v1', {}))
    monkeypatch.setattr(formatting, 'validate_audio_stream', lambda *a, **kw: {'valid': True})
    gen = ModuleType('audiogen.inference.generate')
    def generate(model, tokenizer, prompt, metadata, **kw):
        events.append(('generate', prompt, kw, torch.initial_seed()))
        assert model.generation_config.top_k == 0 and model.generation_config.top_p == 1
        return torch.tensor([3, 4] * 5)
    gen.generate_token_ids = generate
    gen.decode_token_ids = lambda *a, **kw: (events.append(('decode', kw)) or torch.zeros(1))
    monkeypatch.setitem(sys.modules, 'audiogen.inference.generate', gen)
    reward = ModuleType('audiogen.training.grpo')
    reward._load_whisper = lambda name, device: (events.append(('whisper', device)) or object())
    counter = iter(range(1000))
    reward.score_whisper_components = lambda *a: dict(reward=(next(counter) % 4) / 4, transcription='test')
    monkeypatch.setitem(sys.modules, 'audiogen.training.grpo', reward)
    codec = ModuleType('audiogen.codecs.moss')
    def codec_load(**kw):
        events.append(('codec', kw))
        return SimpleNamespace(sample_rate=24000, decode=lambda _: torch.zeros(1, 24))
    codec.MOSSCodec = SimpleNamespace(load=codec_load)
    monkeypatch.setitem(sys.modules, 'audiogen.codecs.moss', codec)
    wandb = ModuleType('wandb')
    run = SimpleNamespace(get_url=lambda: 'https://wandb.ai/test/run', id='test', settings=SimpleNamespace(mode='online'),
                          summary={}, log=lambda data, **kw: events.append(('log', data, kw)),
                          finish=lambda **kw: events.append(('finish', kw)))
    wandb.init = lambda **kw: (configs.append(kw) or run)
    monkeypatch.setitem(sys.modules, 'wandb', wandb)
    return SimpleNamespace(m=m, config=config, events=events, manifest=manifest, tmp=tmp_path,
                           gen=gen, run=run, rt=rt)


def test_execute_two_updates_checks_cache_before_update_and_publishes_evidence(execution_harness):
    h = execution_harness
    h.m.execute(h.config, None, None)
    updates = [e for e in h.events if e[0] == 'update']
    assert len(updates) == 2
    assert all(e[2]['old_logprob_tolerance'] == .05 for e in updates)
    batches = [e[1] for e in h.events if e[0] == 'scatter']
    assert all([len(b) for b in chunks] == [2] * 4 for chunks in batches)
    assert len([e for e in h.events if e[0] == 'sync']) == 3
    assert [e[2] for e in h.events if e[0] == 'load'] == ['cpu', 'cuda:4', 'cuda:5', 'cuda:4']
    assert ('whisper', 'cuda:6') in h.events
    out = Path(h.config['output'])
    receipt = json.loads((out / 'smoke_receipt.json').read_text())
    assert receipt['cached_likelihood']['tokens'] == 8
    assert receipt['cached_likelihood']['max_error'] <= .05
    assert receipt['old_logprob_tolerance'] == .05
    assert receipt['code_sha256'] == h.manifest['code_sha256']
    assert len(receipt['telemetry']) == 2
    assert len(receipt['telemetry'][0]['per_rank']) == 4
    assert set(receipt['telemetry'][0]['rank0_devices']) == {'0', '1', '2', '3', '4', '5', '6', '7'}
    assert receipt['telemetry'][0]['per_rank'][0]['gradient_coverage']
    assert (out / 'COMPLETE.json').exists()


def test_execute_rank0_generation_failure_stops_before_update(execution_harness):
    h = execution_harness
    def fail(*a, **kw):
        raise RuntimeError('deliberate generation failure')
    h.gen.generate_token_ids = fail
    with pytest.raises(RuntimeError, match='generate/score.*deliberate'):
        h.m.execute(h.config, None, None)
    assert not any(e[0] in ('update', 'save') for e in h.events)
    assert ('finish', {'exit_code': 1}) in h.events
    assert h.events[-1] == ('destroy',)


def test_execute_pilot_listening_pauses_before_step_six(execution_harness, monkeypatch):
    h = execution_harness
    h.config.update(mode='pilot', steps=25, save_steps=[5, 10, 25])
    def wait(out, step, binding, **kw):
        assert step == 5 and binding == 'test-binding'
        packet = json.loads((out / 'listening-step-5' / 'manifest.json').read_text())
        assert len(packet['samples']) == 18
        assert {r['seed'] for r in packet['samples']} == {42, 43}
        assert all(r['decode_policy'] == 'strict' for r in packet['samples'])
        assert all(Path(r['wav']).exists() for r in packet['samples'])
        assert not {r['prompt'] for r in packet['samples']} & {r['prompt'] for r in h.manifest['prompts']}
        raise TimeoutError('approval pending')
    monkeypatch.setattr(h.m, 'wait_for_listening_approval', wait, raising=False)
    with pytest.raises(RuntimeError, match='approval pending'):
        h.m.execute(h.config, None, None)
    assert len([e for e in h.events if e[0] == 'update']) == 5
    assert h.run.summary['status'] == 'listening_ready'


def test_execute_cache_failure_never_updates(execution_harness, monkeypatch):
    h = execution_harness
    def fail(*a, **kw):
        raise RuntimeError('cached likelihood mismatch')
    monkeypatch.setattr(h.m, 'verify_cached_logprobs', fail, raising=False)
    with pytest.raises(RuntimeError, match='cached likelihood mismatch'):
        h.m.execute(h.config, None, None)
    assert not any(e[0] == 'update' for e in h.events)


def test_execute_listening_all_checkpoints_waits_only_at_five_and_ten(execution_harness, monkeypatch):
    h = execution_harness
    h.config.update(mode='pilot', steps=25, save_steps=[5, 10, 25])
    waits = []
    def approve(out, step, binding, **kw):
        waits.append(step)
        assert len([e for e in h.events if e[0] == 'update']) == step
        assert any(e[0] == 'log' and e[1].get('status') == 'listening_ready' and e[2].get('commit', True)
                   for e in h.events)
        return dict(step=step, binding=binding, approved=True)
    monkeypatch.setattr(h.m, 'wait_for_listening_approval', approve)
    h.m.execute(h.config, None, None)
    assert waits == [5, 10]
    for step in (5, 10, 25):
        packet = json.loads((Path(h.config['output']) / f'listening-step-{step}' / 'manifest.json').read_text())
        assert len(packet['samples']) == 18
    assert len([e for e in h.events if e[0] == 'update']) == 25


def test_listening_packet_keeps_strict_decode_failures_without_repair(execution_harness, monkeypatch):
    h = execution_harness
    h.config.update(mode='pilot', steps=5, save_steps=[5])
    original = h.gen.decode_token_ids
    calls = []
    def decode(*a, **kw):
        calls.append(kw)
        # Training uses 40 decodes; fail the first listening sample only.
        if len(calls) == 41:
            raise ValueError('strict decode failed')
        return original(*a, **kw)
    h.gen.decode_token_ids = decode
    monkeypatch.setattr(h.m, 'wait_for_listening_approval',
                        lambda out, step, binding, **kw: dict(step=step, binding=binding, approved=True))
    h.m.execute(h.config, None, None)
    folder = Path(h.config['output']) / 'listening-step-5'
    packet = json.loads((folder / 'manifest.json').read_text())
    assert packet['failures'] == 1 and len(packet['samples']) == 18
    bad = packet['samples'][0]
    assert 'strict decode failed' in bad['error'] and bad['wav'] is None
    assert bad['raw_token_ids'] and bad['repaired'] is False
    assert len((folder / 'raw.jsonl').read_text().splitlines()) == 18
    assert all(kw == {'strict': True} for kw in calls)


def test_cached_replay_matches_tiny_native_hybrid_cpu():
    import torch
    from test_nemotron_checkpointing import tiny_model
    from audiogen.training.grpo_native import completion_token_logprobs
    m = launcher()
    torch.manual_seed(42)
    model = tiny_model().eval()
    result = m.verify_cached_logprobs(model, torch.tensor([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]),
                                     2, completion_token_logprobs, tolerance=.05)
    assert result['tokens'] == 8 and result['max_error'] < .05


def test_smoke_validator_rejects_missing_new_measurements(execution_harness):
    h = execution_harness
    h.m.execute(h.config, None, None)
    path = Path(h.config['output']) / 'smoke_receipt.json'
    valid = json.loads(path.read_text())
    h.m.validate_smoke_receipt(path, 'test-binding')
    for field in ('cached_likelihood', 'telemetry', 'code_sha256', 'old_logprob_tolerance'):
        stale = dict(valid)
        del stale[field]
        path.write_text(json.dumps(stale))
        with pytest.raises(RuntimeError, match='smoke evidence'):
            h.m.validate_smoke_receipt(path, 'test-binding')
    for change in ({'grad_norm': 0.}, {'old_logprob_max_abs_diff': .06}, {'gradient_coverage': {}}):
        altered = json.loads(json.dumps(valid))
        altered['telemetry'][0]['per_rank'][0].update(change)
        path.write_text(json.dumps(altered))
        with pytest.raises(RuntimeError, match='smoke evidence'):
            h.m.validate_smoke_receipt(path, 'test-binding')
