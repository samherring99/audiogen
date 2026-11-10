"""CPU contracts; GPU end-to-end test lives in native_grpo_smoke.py."""
import importlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch


def runtime():
    return importlib.import_module('audiogen.training.grpo_native')


class ToyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 11)
        self.seen = None

    def forward(self, **kwargs):
        self.seen = kwargs
        return SimpleNamespace(logits=self.embedding(kwargs['input_ids']))


def test_causal_span_includes_final_eos_and_full_logits():
    model = ToyLM()
    ids = torch.tensor([1, 3, 5, 7, 2])
    actual = runtime().completion_token_logprobs(model, ids, 2)
    expected = model.embedding(ids)[1:-1].float().log_softmax(-1).gather(1, ids[2:, None]).squeeze(1)
    torch.testing.assert_close(actual, expected)
    assert len(actual) == 3
    assert model.seen['use_cache'] is False
    assert model.seen['logits_to_keep'] == 0
    assert torch.equal(model.seen['attention_mask'], torch.ones(1, 5, dtype=torch.long))
    actual.sum().backward()
    assert model.embedding.weight.grad[7].abs().sum() > 0


@pytest.mark.parametrize('ids,prefix', [(torch.tensor([], dtype=torch.long), 0), (torch.tensor([1, 2]), 0), (torch.tensor([1, 2]), 2), (torch.tensor([[1, 2]]), 1)])
def test_logprob_span_rejects_empty_or_padded_batch(ids, prefix):
    with pytest.raises(ValueError):
        runtime().completion_token_logprobs(ToyLM(), ids, prefix)


def test_shared_prefix_completion_detaches_cpu_evidence(tmp_path):
    from test_sft_integration import make_fixture
    from transformers import AutoTokenizer
    from audiogen.tokens.vocab import expand_tokenizer
    from audiogen.tokens.formatting import build_audio_prefix
    tok = AutoTokenizer.from_pretrained(make_fixture(tmp_path))
    tok, meta = expand_tokenizer(tok, 'moss', 2, 2)
    config = SimpleNamespace(audiogen={'prompt_format': 'speech_v1'})
    tokens = torch.tensor([meta.audio_token_start_id, meta.audio_token_start_id + 2, meta.audio_end_token_id])
    old = torch.full((3,), -2.0, requires_grad=True)
    row = runtime().prepare_completion(tok, meta, config, 'hello', tokens, old, old, 1.0)
    prefix = build_audio_prefix(tok, 'hello', meta, 'speech_v1')
    assert row.input_ids.tolist() == prefix + tokens.tolist()
    assert row.prefix_length == len(prefix)
    assert row.old_log_probs.device.type == 'cpu'
    assert not row.old_log_probs.requires_grad
    with pytest.raises(ValueError, match='empty'):
        runtime().prepare_completion(tok, meta, config, 'hello', tokens[:0], old[:0], old[:0], 1.0)


def valid_record():
    return runtime().NativeCompletion(torch.tensor([1, 2, 3]), 1,
                                      torch.full((2,), -2.), torch.full((2,), -2.), 1.)


@pytest.mark.parametrize('field,value', [
    ('prefix_length', 1.0), ('prefix_length', True),
    ('prefix_length', torch.tensor([1, 1])), ('prefix_length', torch.tensor(1)),
    ('prefix_length', 1 + 0j), ('input_ids', [1, 2, 3]),
    ('input_ids', torch.tensor([1., 2., 3.])),
    ('old_log_probs', torch.tensor([-2., -2.], dtype=torch.complex64)),
    ('ref_log_probs', torch.tensor([-2, -2])),
    ('old_log_probs', [-2., -2.]), ('advantage', torch.tensor([1., 2.])),
    ('advantage', 1 + 1j), ('advantage', '1'),
    ('old_log_probs', torch.full((2,), float('nan'))),
])
def test_record_validation_returns_false_for_malformed_evidence(field, value):
    assert runtime()._record_valid(replace(valid_record(), **{field: value}), 11) is False


def test_record_validation_accepts_detached_real_floating_evidence():
    assert runtime()._record_valid(valid_record(), 11) is True


def test_native_gradient_coverage_reports_all_categories_and_sparse_zeros():
    from test_nemotron_checkpointing import tiny_model
    model = tiny_model()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    model.model.layers[1].mixer.experts.up_proj.grad.zero_()
    coverage = runtime().native_gradient_coverage(model)
    assert set(coverage) == {'embedding', 'head', 'norm', 'mamba', 'attention',
                             'mlp', 'router', 'routed_experts', 'shared_experts',
                             'moe_projection', 'other'}
    assert sum(g['parameter_numel'] for g in coverage.values()) == sum(p.numel() for p in model.parameters())
    assert coverage['other']['parameter_numel'] == 0
    assert coverage['routed_experts']['nonzero_grad_numel'] < coverage['routed_experts']['grad_numel']
    for group in coverage.values():
        assert group['grad_norm'] ** 2 == pytest.approx(group['nonzero_grad_numel'])
    model.zero_grad(set_to_none=True)
    assert all(g['grad_numel'] == 0 and g['grad_norm'] == 0
               for g in runtime().native_gradient_coverage(model).values())


def test_bounded_cpu_sync_preserves_destination_dtype_and_buffers():
    model = ToyLM().to(torch.bfloat16)
    model.register_buffer('routing', torch.zeros(2, dtype=torch.float32))
    state = {'embedding.weight': torch.randn(11, 11), 'routing': torch.tensor([.0010001, .0010002])}
    runtime().copy_cpu_state_to_model(model, state, chunk_bytes=16)
    torch.testing.assert_close(model.embedding.weight, state['embedding.weight'].bfloat16(), atol=0, rtol=0)
    torch.testing.assert_close(model.routing, state['routing'], atol=0, rtol=0)
    assert model.routing.dtype == torch.float32
    with pytest.raises(ValueError, match='keys'):
        runtime().copy_cpu_state_to_model(model, {'routing': state['routing']})
