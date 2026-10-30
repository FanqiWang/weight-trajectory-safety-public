"""Arm matching and deduplication, plus unit tests for encoding, batching, refusal judging and LoRA."""

import numpy as np
import pytest
import torch
from conftest import ROOT, smoke_config

from wts.config import load_config, parse_arms
from wts.data import (REFUSAL_TEMPLATES, build_arm_examples, build_pools, normalize_text, refusal_for,
                      synthetic_harmful)
from wts.evaluate import is_refusal_keywords
from wts.train import IGNORE, batch_schedule, encode_example


@pytest.fixture
def pools(tmp_path):
    return build_pools(smoke_config(tmp_path))


# ---------------------------------------------------------------- matching


@pytest.mark.parametrize("p", [0.05, 0.1, 0.5, 1.0])
@pytest.mark.parametrize("seed,group", [(0, "test"), (100, "reference")])
def test_c_and_s_are_matched(pools, p, seed, group):
    C = build_arm_examples(pools, p, "comply", seed, group, 100)
    S = build_arm_examples(pools, p, "refuse", seed, group, 100)
    assert [e["prompt"] for e in C] == [e["prompt"] for e in S]  # same prompts in the same order
    assert [e["kind"] for e in C] == [e["kind"] for e in S]
    for c, s in zip(C, S):
        if c["kind"] == "benign":
            assert c["response"] == s["response"]  # identical benign examples
        else:
            assert s["response"] == refusal_for(s["prompt"])  # only the target response differs
            assert c["response"] != s["response"]
    assert sum(e["kind"] == "harmful" for e in C) == round(p * 100)


def test_arms_are_nested_across_p(pools):
    small = {e["prompt"] for e in build_arm_examples(pools, 0.05, "comply", 0, "test", 100) if e["kind"] == "harmful"}
    big = {e["prompt"] for e in build_arm_examples(pools, 0.10, "comply", 0, "test", 100) if e["kind"] == "harmful"}
    assert small < big


def test_reference_and_test_use_disjoint_harmful_prompts(pools):
    ref = {h["prompt"] for h in pools.harmful_for("reference")}
    test = {h["prompt"] for h in pools.harmful_for("test")}
    assert ref and test and not (ref & test)
    assert ref | test == {h["prompt"] for h in pools.harmful}


def test_test_set_has_zero_overlap_with_training(pools):
    train_keys = {normalize_text(h["prompt"]) for h in pools.harmful}
    train_keys |= {normalize_text(b["prompt"]) for src in pools.benign.values() for b in src}
    assert pools.test_prompts
    assert not train_keys & {normalize_text(p) for p in pools.test_prompts}


def test_build_is_deterministic(pools):
    a = build_arm_examples(pools, 0.1, "comply", 1, "test", 50)
    b = build_arm_examples(pools, 0.1, "comply", 1, "test", 50)
    c = build_arm_examples(pools, 0.1, "comply", 2, "test", 50)
    assert a == b and a != c


def test_seed_overlap_rejected(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        load_config(ROOT / "configs" / "smoke.yaml", {"seeds": {"test": [0, 100], "reference": [100]}})


def test_arm_parsing_accepts_reference_only(tmp_path):
    cfg = smoke_config(tmp_path)
    cfg["data"]["arms"] = [{"name": "C100", "p": 1.0, "target": "comply", "reference_only": True}]
    (arm,) = parse_arms(cfg)
    assert arm.role == "reference" and arm.family == "C"


# ---------------------------------------------------------------- encoding: loss on response tokens only


def test_encode_masks_prompt_tokens(tmp_path):
    from wts.model import load_tokenizer

    tok = load_tokenizer(smoke_config(tmp_path))
    ex = encode_example(tok, "request 3 about topic lamp", "sure here are steps for topic lamp", 64)
    labelled = [t for t, lab in zip(ex["input_ids"], ex["labels"]) if lab != IGNORE]
    text = tok.decode(labelled)
    assert text.startswith("sure here are steps for topic lamp")
    assert "request" not in text  # the prompt is fully masked
    assert tok.eos_token_id in labelled  # the model must learn to emit <|im_end|> after the response


def test_batch_schedule_covers_each_epoch():
    sched = batch_schedule(10, 4, 10, seed=0)
    flat = [i for b in sched for i in b]
    assert all(len(b) == 4 for b in sched)
    assert sorted(flat[:10]) == list(range(10)) and sorted(flat[10:20]) == list(range(10))
    assert sched == batch_schedule(10, 4, 10, seed=0)


# ---------------------------------------------------------------- refusal judging


def test_all_refusal_templates_are_detected():
    assert all(is_refusal_keywords(t) for t in REFUSAL_TEMPLATES)


def test_compliance_is_not_refusal():
    assert not any(is_refusal_keywords(h["comply"]) for h in synthetic_harmful(20))
    assert not is_refusal_keywords("Here is a step-by-step guide. First, ...")
    assert is_refusal_keywords("I\u2019m sorry, but I can\u2019t help with that.")


# ---------------------------------------------------------------- LoRA


def test_lora_forward_matches_dense_delta(tmp_path):
    from wts.lora import inject_lora

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.layers[0].q_proj = torch.nn.Linear(6, 4)
    W0 = model.layers[0].q_proj.weight.detach().clone()
    b0 = model.layers[0].q_proj.bias.detach().clone()
    h = inject_lora(model, ["q_proj"], r=2, alpha=4)
    h.reset(seed=0)
    m = h.modules["layers.0.q_proj"]
    with torch.no_grad():
        m.lora_B.normal_()
    x = torch.randn(3, 6)
    want = x @ (W0 + m.delta_weight()).T + b0
    torch.testing.assert_close(m(x), want, rtol=1e-5, atol=1e-6)
    # the same seed gives the same initialization (part of C/S matching), and B starts at 0
    s1 = h.state()
    h.reset(seed=0)
    s2 = h.state()
    torch.testing.assert_close(s1["layers.0.q_proj.A"], s2["layers.0.q_proj.A"])
    assert s2["layers.0.q_proj.B"].abs().sum() == 0


def test_lora_matches_peft(tmp_path):
    """Export to peft format, load with peft: forward outputs must match the hand-written LoRA."""
    peft = pytest.importorskip("peft")
    from wts.lora import export_peft_adapter, inject_lora
    from wts.model import load_model, load_tokenizer

    cfg = smoke_config(tmp_path)
    tok = load_tokenizer(cfg)
    ids = torch.tensor([tok("request 1 about topic lamp")["input_ids"]])
    model = load_model(cfg, device=torch.device("cpu"))
    h = inject_lora(model, ["q_proj", "v_proj"], r=4, alpha=8)
    h.reset(seed=3)
    with torch.no_grad():
        for m in h.modules.values():
            m.lora_B.normal_(std=0.1)
    ours = model(ids).logits
    path = export_peft_adapter(h.state(), tmp_path / "adapter", "tiny", 4, 8, ["q_proj", "v_proj"])
    base = load_model(cfg, device=torch.device("cpu"))
    pm = peft.PeftModel.from_pretrained(base, str(path))
    theirs = pm(ids).logits
    torch.testing.assert_close(ours, theirs, rtol=1e-4, atol=1e-5)
    assert np.isfinite(ours.detach().numpy()).all()


def test_pilot_configs_load_and_plan():
    from wts.config import plan_runs

    for name, n_runs in (("pilot_qv.yaml", 27), ("pilot_all_linear.yaml", 27)):
        cfg = load_config(ROOT / "configs" / name)
        (setting,) = cfg["settings"]
        runs = plan_runs(cfg, setting["name"])
        # 3 test seeds x 5 test arms + 4 reference seeds x (B, C100, S100)
        assert len(runs) == n_runs, name
        assert {r.seed_group for r in runs if r.arm.name == "B"} == {"test", "reference"}
