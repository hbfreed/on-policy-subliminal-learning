"""Online distillation with an exact KL: vLLM rollouts + HF teacher + HF LoRA student.

Layout (one process, three GPUs), after ~/Documents/PythonProjects/on-policy-olmo/opd.py:

    GPU 0     vLLM rollouts. On-policy: the student LoRA, re-saved to tmpfs after every
              update and hot-reloaded with LoRARequest(load_inplace=True), as in
              rl-experiments/train.py. Off-policy: the fixed teacher LoRA (or the base
              model). Generation for step n+1 overlaps training of step n (one step stale);
              `overlap=False` makes every rollout come from the current weights.
    GPU 1..N  Data-parallel HF replicas, one thread each, half the batch each. Teacher and
              student share the base weights: the student is the trainable LoRA ("default"),
              the teacher a frozen second LoRA ("teacher", swapped in for a no-grad forward)
              or the bare base (adapters disabled). Gradients are summed into replica 0 over
              P2P, which takes the optimiser step and copies the new LoRA weights back.
              Neither LoRA touches lm_head; with `legal_ids` only those rows of it are used.

Each step samples one completion per prompt with the paper's sampling config (temperature
1, no truncation of the distribution). At every completion position (answer tokens and the
closing <|im_end|>) the loss is

    reverse  KL(student || teacher) = sum_v p_s(v) (log p_s(v) - log p_t(v))   (default)
    forward  KL(teacher || student) = sum_v p_t(v) (log p_t(v) - log p_s(v))   (the paper's)

exact over the full vocabulary, or over `legal_ids` with both sides renormalised (the
paper's restriction for number sequences), as a token mean over the batch. Everything else
is the paper's setup (logit_distillation/shared.py): rank-8 LoRA, alpha 16, on the attention
projections incl. Qwen3.5's linear-attention ones (`lora_targets="all"` adds the MLP, the
paper's appendix comparison), AdamW(0.9, 0.95, eps 1e-8, wd 0), lr 1e-4 with 5% linear
warmup then linear decay to 5%. Adapters are saved per epoch as <out>/ep<k> (and every
`ckpt_every` steps as <out>/step<n>), which `sft/eval_local.py --adapter` evaluates.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

# FlashInfer's sampler JIT-compiles with nvcc; vLLM's PyTorch sampler is fine here.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

ROOT = Path(__file__).resolve().parent.parent
BACKDOOR = ROOT / "beyond_owls" / "experiments" / "backdoor"
sys.path[:0] = [str(BACKDOOR / "sft"), str(BACKDOOR / "logit_distillation")]

import common
import shared

VLLM_DEVICE = "cuda:0"  # vLLM's engine takes the first visible GPU
MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]
LORA_TARGETS = {"attn": shared.ATTN_TARGETS, "all": shared.ATTN_TARGETS + MLP_TARGETS}

# vLLM serves Qwen3.5 as the vision-language model, whose decoder layers live under
# `model.language_model.`. A LoRA saved from the text-only HF model (`model.layers.`) loads
# without error but matches no module, so vLLM silently samples the *base* model. Every
# adapter handed to vLLM goes through these.
_HF_LAYERS = "base_model.model.model.layers."
_VLLM_LAYERS = "base_model.model.model.language_model.layers."


def to_vllm_keys(state_dict: dict) -> dict:
    out = {k.replace(_HF_LAYERS, _VLLM_LAYERS): v for k, v in state_dict.items()}
    assert all(_VLLM_LAYERS in k for k in out), "unexpected LoRA key layout"
    return out


def assert_lora_applied(llm, lora, tok) -> None:
    """Fail loudly if vLLM serves `lora` identically to the base model (the silent failure
    mode above): one teacher-training row, scored with and without it."""
    from vllm import SamplingParams
    r = shared.read_jsonl(shared.DATA / "teacher_train.jsonl.gz")[0]
    ids, w = shared.training_example(tok, r["input_text"], r["target_text"])
    n = w.index(1.0)
    params = SamplingParams(max_tokens=1, prompt_logprobs=0, detokenize=False)
    lp = [sum(o.prompt_logprobs[k][ids[k]].logprob for k in range(n, len(ids)))
          for o in (llm.generate([{"prompt_token_ids": ids}], params, lora_request=req,
                                 use_tqdm=False)[0] for req in (None, lora))]
    assert abs(lp[0] - lp[1]) > 1e-3, f"vLLM ignores the LoRA {lora.lora_path}: {lp}"


def check_policy_sync(llm, lora, replica, tok) -> dict:
    """End-to-end check of the live on-policy path, on one training row: the shift in the
    answer's log-probability from base to student must agree between vLLM (serving the
    published adapter) and HF (the trained replica). Shifts, not raw values, are compared:
    the two engines differ by ~0.2-0.4 nats on the same base model (bf16 numerics)."""
    from vllm import SamplingParams
    r = shared.read_jsonl(shared.DATA / "teacher_train.jsonl.gz")[0]
    ids, w = shared.training_example(tok, r["input_text"], r["target_text"])
    n = w.index(1.0)
    params = SamplingParams(max_tokens=1, prompt_logprobs=0, detokenize=False)
    v_base, v_lora = [sum(o.prompt_logprobs[k][ids[k]].logprob for k in range(n, len(ids)))
                      for o in (llm.generate([{"prompt_token_ids": ids}], params, lora_request=q,
                                             use_tqdm=False)[0] for q in (None, lora))]

    @torch.no_grad()
    def hf_lp():
        logits = replica.model(input_ids=torch.tensor([ids], device=replica.device)).logits[0]
        lp = torch.log_softmax(logits.float(), -1)
        return sum(lp[k - 1, ids[k]].item() for k in range(n, len(ids)))
    replica.model.eval()
    h_lora = hf_lp()
    with replica.model.disable_adapter():
        h_base = hf_lp()
    dv, dh = v_lora - v_base, h_lora - h_base
    res = {"vllm_shift": round(dv, 4), "hf_shift": round(dh, 4)}
    assert abs(dh) > 1e-3, f"the student has not moved yet: {res}"
    assert abs(dv - dh) <= 0.35 * abs(dh) + 0.1, f"vLLM is not serving the student: {res}"
    return res


def vllm_adapter(path: str) -> str:
    """A copy of a peft adapter directory with vLLM's key layout (in /dev/shm, refreshed if the
    source is newer); returns its path."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    src = Path(path).resolve()
    dst = Path("/dev/shm/vllm_adapters") / src.as_posix().strip("/").replace("/", "_")
    w_src, w_dst = src / "adapter_model.safetensors", dst / "adapter_model.safetensors"
    if not w_dst.exists() or w_dst.stat().st_mtime < w_src.stat().st_mtime:
        dst.mkdir(parents=True, exist_ok=True)
        with safe_open(str(w_src), "pt") as f:
            sd = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118 (safe_open has no __iter__)
        save_file(to_vllm_keys(sd), str(w_dst), metadata={"format": "pt"})
        shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")
    return str(dst)

# (student prompt ids, teacher prompt ids) for one rollout.
Pair = tuple[list[int], list[int]]


@dataclass
class Config:
    out: str
    n_steps: int
    steps_per_epoch: int
    teacher: str | None = None  # LoRA adapter directory; None = the base model
    # Who writes the answers: "student" (on-policy) or "teacher" (off-policy), sampled fresh
    # by vLLM each step; or "dataset": a fixed set of teacher answers (the paper's off-policy
    # setup), where batch_fn returns (student ids, teacher ids, completion) and vLLM is unused.
    rollout: str = "student"
    kl: str = "reverse"  # "reverse" KL(student||teacher) or "forward" KL(teacher||student)
    lora_targets: str = "attn"  # "attn" (the paper's) or "all" (+ MLP)
    legal_ids: list[int] | None = None
    # Prompt tokens shared by every row (system prompt + chat template): the student's
    # forward runs them once per micro-batch instead of once per row. 0 = off.
    shared_prefix: int = 0
    # How vLLM gets each new student LoRA: "inplace" overwrites one adapter (load_inplace);
    # "new_id" writes a fresh directory under a new LoRA id every update (no in-place reload).
    lora_reload: str = "inplace"
    # Rollouts for this many consecutive steps come from one vLLM call (batch_size x gen_ahead
    # sequences): decoding is latency-bound, so this divides generation time per step by about
    # gen_ahead, at the price of rollouts gen_ahead..2*gen_ahead-1 updates stale (1 = as before).
    gen_ahead: int = 1
    # Prompts sampled per step for student/teacher rollouts; the first batch_size kept
    # rollouts are trained on, so every step has exactly batch_size rows like the paper's fixed
    # dataset (rollouts failing the filter or truncated are dropped). 0 = batch_size.
    prompts_per_step: int = 0
    # The step-60 vLLM-vs-HF shift check: "strict" asserts; "warn" logs. Early in training the
    # student's shift (~0.15 nats) can be below cross-engine bf16 noise (~0.2), so for students
    # that never train on the check row it is underpowered; verify with an amplified adapter.
    sync_check: str = "strict"
    # Dataset rollouts only: batch_fn's teacher element is the teacher's precomputed logits
    # over legal_ids ([completion tokens, len(legal_ids)], fp32; precompute_teacher.py), so
    # the teacher adapter is not loaded and no teacher forward runs while training.
    teacher_cached: bool = False
    batch_size: int = 32
    rank: int = 8
    lr: float = 1e-4
    max_new_tokens: int = 256
    micro_bs: int = 16
    grad_ckpt: bool = True
    liger: bool = False  # fused, chunked KL (Liger) when the loss is over the full vocabulary
    overlap: bool = True
    vllm_mem: float = 0.92  # GPU 0 holds only vLLM; Mamba cache needs a block per sequence
    train_devices: tuple[str, ...] = ("cuda:1", "cuda:2")
    seed: int = 1
    ckpt_every: int = 0  # extra adapter saves, in steps; 0 = per epoch only
    state_every: int = 250  # resumable optimiser state, in steps
    max_steps: int = 0  # stop early without changing the lr schedule
    wandb_project: str | None = "subliminal-opd"  # None = no wandb
    run_name: str | None = None  # wandb run name; defaults to the out directory's name


# ---------------------------------------------------------------- models


def load_model(adapter: str | None, device: str):
    """shared.load_model, but through the text config: vLLM pins transformers < 5.18, whose
    AutoModelForCausalLM cannot build Qwen3.5 from the composite (vision-language) config.
    Same weights (no missing or unexpected keys)."""
    from transformers import AutoConfig, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        common.MODEL, config=AutoConfig.from_pretrained(common.MODEL).text_config,
        dtype=torch.bfloat16, attn_implementation="sdpa")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.config.use_cache = False
    return model.to(device).eval()


def add_lora(model, rank: int, targets: list[str]):
    """shared.add_lora with a choice of target modules: rank-r LoRA (alpha 2r, no dropout),
    trainable weights in fp32."""
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                                             bias="none", task_type="CAUSAL_LM",
                                             target_modules=targets))
    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.float()
    return model, trainable


def build_vllm(cfg: Config):
    from vllm import LLM
    return LLM(common.MODEL, dtype="bfloat16", skip_tokenizer_init=True, seed=cfg.seed,
               max_model_len=1024,
               max_num_seqs=(cfg.prompts_per_step or cfg.batch_size) * cfg.gen_ahead,
               gpu_memory_utilization=cfg.vllm_mem, enable_lora=True,
               max_lora_rank=max(8, cfg.rank), max_loras=2,
               limit_mm_per_prompt={"image": 0, "video": 0})


def batch_inputs(rows: list[tuple[list[int], list[int]]], device):
    """Right-padded input_ids / attention_mask for (prompt, completion) rows, plus the flat
    indices into [rows * L] of the positions predicting each completion token (the same
    positions as shared.supervised_hidden's mask). Built on the CPU in pinned memory, so the
    copies are asynchronous and selecting the positions needs no GPU->CPU sync."""
    L = max(len(p) + len(c) for p, c in rows)
    pin = torch.device(device).type == "cuda"
    inp = torch.full((len(rows), L), common.PAD, dtype=torch.long, pin_memory=pin)
    att = torch.zeros((len(rows), L), dtype=torch.long, pin_memory=pin)
    idx = []
    for i, (p, c) in enumerate(rows):
        n = len(p) + len(c)
        inp[i, :n] = torch.tensor(p + c)
        att[i, :n] = 1
        idx.extend(range(i * L + len(p) - 1, i * L + n - 1))
    idx_t = torch.tensor(idx, dtype=torch.long)
    idx_t = idx_t.pin_memory() if pin else idx_t
    return (inp.to(device, non_blocking=True), att.to(device, non_blocking=True),
            idx_t.to(device, non_blocking=True))


def completion_hidden(causal_lm, rows: list[tuple[list[int], list[int]]], device):
    """Final hidden states predicting each completion token, rows in order: [positions, H]."""
    inp, att, idx = batch_inputs(rows, device)
    h = causal_lm.model(input_ids=inp, attention_mask=att).last_hidden_state
    return h.flatten(0, 1).index_select(0, idx)


class _PrefixState:
    """A read-only stand-in for transformers' Cache, holding one shared prompt prefix.

    "record": the prefix's own forward stores each layer's state (full attention: rotated
    keys/values; linear attention (GatedDeltaNet): the last K-1 conv inputs and the final
    recurrent state) and passes everything through unchanged. "read": every row of the
    suffix batch continues from that state, broadcast over the batch. Nothing is mutated in
    "read" mode, so gradient checkpointing can recompute the suffix layers safely; gradients
    from all rows flow back into the one prefix computation (the broadcast sums them)."""

    def __init__(self):
        self.mode, self.batch = "record", 1
        self.kv, self.conv, self.rec = {}, {}, {}

    @property
    def layers(self):
        return self

    def __getitem__(self, i):  # cache.layers[i]
        rec = self.rec.get(i)
        if rec is not None and self.mode == "read":
            rec = rec.expand(self.batch, *rec.shape[1:]).contiguous()
        return _LayerView(rec)

    def has_previous_state(self, layer_idx, state_idx=0):
        return self.mode == "read"

    def update_conv_state(self, x, layer_idx, conv_kernel_size):
        if self.mode == "record":
            self.conv[layer_idx] = x[:, :, -(conv_kernel_size - 1):]
            return x
        tail = self.conv[layer_idx]
        return torch.cat([tail.expand(x.shape[0], *tail.shape[1:]), x], dim=2)

    def update_recurrent_state(self, state, layer_idx):
        if self.mode == "record":
            self.rec[layer_idx] = state

    def update(self, k, v, layer_idx, *args, **kwargs):
        if self.mode == "record":
            self.kv[layer_idx] = (k, v)
            return k, v
        pk, pv = self.kv[layer_idx]
        b = k.shape[0]
        return (torch.cat([pk.expand(b, *pk.shape[1:]), k], dim=2),
                torch.cat([pv.expand(b, *pv.shape[1:]), v], dim=2))


class _LayerView:
    def __init__(self, rec):
        self.recurrent_states, self.conv_states, self.record_past = [rec], [None], False


def completion_hidden_shared_prefix(causal_lm, rows, device, n_prefix: int):
    """completion_hidden, with the first n_prefix prompt tokens (identical in every row: the
    system prompt and chat template) run once instead of once per row. Same outputs up to
    bf16 rounding; the rows' gradients all reach the shared prefix."""
    from torch.utils.checkpoint import checkpoint
    prefix = rows[0][0][:n_prefix]
    assert all(p[:n_prefix] == prefix and len(p) > n_prefix for p, _ in rows), "prefix differs"
    tm = causal_lm.model
    st = _PrefixState()

    def positions(n, offset, b):
        pid = (torch.arange(n, device=device) + offset).view(1, 1, -1).expand(4, b, -1)
        return pid[0], pid[1:]  # text positions; the 3 rotary (mrope) sections

    # The prefix: one row, causal (q and k have equal length, so no mask is needed), never
    # checkpointed: a recompute during backward would run with st already in "read" mode.
    h = tm.embed_tokens(torch.tensor([prefix], device=device))
    tpos, rpos = positions(n_prefix, 0, 1)
    pe = tm.rotary_emb(h, rpos)
    for layer in tm.layers:
        h = layer.forward(h, position_embeddings=pe, attention_mask=None, position_ids=tpos,
                          past_key_values=st)

    # The suffixes, right-padded, continuing from the prefix state.
    st.mode = "read"
    inp, _, idx = batch_inputs([(p[n_prefix:], c) for p, c in rows], device)
    b, ls = inp.shape
    st.batch = b
    h = tm.embed_tokens(inp)
    tpos, rpos = positions(ls, n_prefix, b)
    pe = tm.rotary_emb(h, rpos)
    # full attention: every suffix query sees the whole prefix and the suffix causally
    # (right padding only ever follows a row's real tokens, so it is never attended to)
    full = torch.ones(ls, n_prefix + ls, dtype=torch.bool, device=device)
    full[:, n_prefix:] = torch.ones(ls, ls, dtype=torch.bool, device=device).tril()
    full = full.view(1, 1, ls, n_prefix + ls).expand(b, 1, -1, -1)
    for layer in tm.layers:
        mask = full if layer.block_type == "full_attention" else None
        if layer.gradient_checkpointing and layer.training:
            h = checkpoint(layer.forward, h, pe, mask, tpos, st, use_reentrant=False)
        else:
            h = layer.forward(h, position_embeddings=pe, attention_mask=mask, position_ids=tpos,
                              past_key_values=st)
    h = tm.norm(h)
    return h.flatten(0, 1).index_select(0, idx)


def serialize_triton_autotune() -> None:
    """The teacher thread and the student both launch fla's autotuned Triton kernels, and
    Triton's Autotuner keeps per-call state on the shared kernel object (`self.nargs`), so
    concurrent calls race. Serialise Autotuner.run: launches are asynchronous, so the lock
    is held only while enqueueing, or for the one-off benchmark of a new shape."""
    from triton.runtime import autotuner
    if getattr(autotuner.Autotuner.run, "_serialized", False):
        return
    lock, run = threading.RLock(), autotuner.Autotuner.run

    def locked(self, *args, **kwargs):
        with lock:
            return run(self, *args, **kwargs)

    locked._serialized = True
    autotuner.Autotuner.run = locked


def kl_sums(ls_logits, lt_logits):
    """(reverse, forward) KL summed over positions; gradient through the student only."""
    ls, lt = torch.log_softmax(ls_logits.float(), -1), torch.log_softmax(lt_logits.float(), -1)
    return (ls.exp() * (ls - lt)).sum(), (lt.exp() * (lt - ls)).sum()


class Replica:
    """Base weights + trainable student LoRA + frozen teacher (second LoRA or the bare base)
    on one GPU."""

    def __init__(self, cfg: Config, device: str):
        self.device, self.has_teacher = device, bool(cfg.teacher)
        self.liger = None
        if cfg.liger and not cfg.legal_ids:  # 29 legal ids: the logits are tiny anyway
            from liger_kernel.transformers import LigerFusedLinearJSD
            # beta 1 = reverse KL(student || teacher), beta 0 = forward (liger_kernel/ops/jsd.py)
            self.liger = LigerFusedLinearJSD(jsd_beta=1.0 if cfg.kl == "reverse" else 0.0)
        self.model, self.trainable = add_lora(load_model(None, device), cfg.rank,
                                              LORA_TARGETS[cfg.lora_targets])
        assert not cfg.teacher_cached or cfg.legal_ids, "cached teacher logits are over legal_ids"
        if cfg.teacher and not cfg.teacher_cached:
            self.model.load_adapter(cfg.teacher, adapter_name="teacher", is_trainable=False)
            self.model.set_adapter("default")
        self.causal_lm = self.model.base_model.model
        head = self.causal_lm.lm_head.weight  # frozen; shared by teacher and student
        # Logits only over the legal ids (the paper renormalises both sides over them), in
        # fp32: 29 rows cost nothing, and bf16 logits put ~1.5% error on the gradient.
        self.head = (head[torch.tensor(cfg.legal_ids, device=device)].float() if cfg.legal_ids
                     else head)
        self.shared_prefix = cfg.shared_prefix
        self.set_grad_ckpt(cfg.grad_ckpt)

    def set_grad_ckpt(self, on: bool) -> None:
        if on:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        else:
            self.model.gradient_checkpointing_disable()

    @torch.no_grad()
    def teacher_hidden(self, batches):
        """Teacher hidden states for every micro-batch, with one adapter swap (set_adapter
        walks every LoRA module) and one eval() per step."""
        self.model.eval()
        if self.has_teacher:
            self.model.set_adapter("teacher")
            try:
                return [completion_hidden(self.causal_lm, rows, self.device) for rows in batches]
            finally:
                self.model.set_adapter("default")
        with self.model.disable_adapter():
            return [completion_hidden(self.causal_lm, rows, self.device) for rows in batches]

    def backward(self, rows, total: int, micro_bs: int, reverse: bool):
        """Accumulate d(sum of the chosen KL)/total into the student's grads; returns the
        (reverse, forward) KL sums as 0-d GPU tensors (no sync here), both logged whichever is
        trained on (with Liger only the trained one is computed; the other is NaN)."""
        for p in self.trainable:
            p.grad = None
        rev_tot = torch.zeros((), device=self.device)
        fwd_tot = torch.zeros((), device=self.device)
        if not rows:
            return rev_tot, fwd_tot
        mbs = [rows[m:m + micro_bs] for m in range(0, len(rows), micro_bs)]
        cached = torch.is_tensor(rows[0][1])
        hts = ([torch.cat([t for _, t, _ in mb]).to(self.device, non_blocking=True) for mb in mbs]
               if cached else self.teacher_hidden([[(t, c) for _, t, c in mb] for mb in mbs]))
        self.model.train()
        for mb, ht in zip(mbs, hts):
            pairs = [(s, c) for s, _, c in mb]
            hs = (completion_hidden_shared_prefix(self.causal_lm, pairs, self.device,
                                                  self.shared_prefix)
                  if self.shared_prefix else completion_hidden(self.causal_lm, pairs, self.device))
            if cached:  # ht already holds the teacher's fp32 logits over the legal ids
                w = self.head
                rev, fwd = kl_sums(hs.to(w.dtype) @ w.T, ht)
            elif self.liger is not None:  # mean over positions -> sum
                kl = self.liger(hs, self.head, ht, self.head, None) * hs.shape[0]
                rev, fwd = (kl, kl.new_tensor(float("nan"))) if reverse else \
                    (kl.new_tensor(float("nan")), kl)
            else:
                w = self.head
                rev, fwd = kl_sums(hs.to(w.dtype) @ w.T, ht.to(w.dtype) @ w.T)
            ((rev if reverse else fwd) / max(total, 1)).backward()
            rev_tot += rev.detach()
            fwd_tot += fwd.detach()
        return rev_tot, fwd_tot


@torch.no_grad()
def teacher_legal_logits(replica: Replica, rows, micro_bs: int) -> list[torch.Tensor]:
    """The teacher's logits over the legal ids at every completion position of each
    (prompt, completion) row, exactly as Replica.backward computes them live (teacher LoRA
    unmerged, fp32 head rows): one CPU fp32 tensor [len(completion), len(legal_ids)] per row."""
    out = []
    for m in range(0, len(rows), micro_bs):
        mb = rows[m:m + micro_bs]
        (ht,) = replica.teacher_hidden([mb])
        lt = (ht.to(replica.head.dtype) @ replica.head.T).cpu()
        out.extend(lt.split([len(c) for _, c in mb]))
    return out


def reduce_and_step(replicas: list[Replica], opt) -> None:
    """Sum the replicas' grads into replica 0, step, and broadcast the new LoRA weights."""
    r0 = replicas[0]
    for p in r0.trainable:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    for r in replicas[1:]:
        g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                       for p in r.trainable]).to(r0.device)
        torch._foreach_add_([p.grad for p in r0.trainable],
                            [x.view_as(p) for x, p in zip(
                                g.split([p.numel() for p in r0.trainable]), r0.trainable)])
    opt.step()
    flat = torch.cat([p.detach().flatten() for p in r0.trainable])
    for r in replicas[1:]:
        torch._foreach_copy_([p.data for p in r.trainable],
                             [x.view_as(p) for x, p in zip(
                                 flat.to(r.device).split([p.numel() for p in r.trainable]),
                                 r.trainable)])


# ---------------------------------------------------------------- rollouts


def rollout(llm, prompts: list[list[int]], cfg: Config, lora_request, publish=None,
            keep: Callable[[list[int]], bool] | None = None):
    """Completions ending in <|im_end|>, or None where max_new_tokens truncated. `publish`
    = (path, CPU state dict): write the student's new LoRA weights there first, here in the
    generation thread, so the trainers never wait on the file write."""
    from vllm import SamplingParams
    t0 = time.time()
    if publish is not None:
        from safetensors.torch import save_file
        save_file(publish[1], publish[0], metadata={"format": "pt"})
    params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=cfg.max_new_tokens,
                            stop_token_ids=common.EOS_IDS, detokenize=False)
    outs = llm.generate([{"prompt_token_ids": p} for p in prompts], params,
                        lora_request=lora_request, use_tqdm=False)
    comps = []
    for o in outs:
        c = o.outputs[0]
        toks = list(c.token_ids)
        if c.finish_reason != "stop":
            comps.append(None)
            continue
        while toks and toks[-1] in common.EOS_IDS:
            toks.pop()
        comps.append(toks + [common.IM_END])
    # The keep filter runs here too, off the trainers' critical path; rejected rollouts come
    # back as False (truncated ones as None).
    if keep is not None:
        comps = [c if c is None or keep(c) else False for c in comps]
    return comps, time.time() - t0


# ---------------------------------------------------------------- loop


def run(cfg: Config, batch_fn: Callable[[int], list[Pair]],
        keep: Callable[[list[int]], bool] = lambda c: True) -> None:
    from vllm.lora.request import LoRARequest

    assert cfg.rollout in ("student", "teacher", "dataset") and cfg.kl in ("reverse", "forward")
    offline = cfg.rollout == "dataset"
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {**asdict(cfg), "model": common.MODEL, "lora_alpha": 2 * cfg.rank,
         "lora_target_modules": LORA_TARGETS[cfg.lora_targets],
         "divergence": f"{cfg.kl}_kl_exact"}, indent=2))
    tok = common.get_tokenizer()
    torch.manual_seed(cfg.seed)
    serialize_triton_autotune()

    llm = None if offline else build_vllm(cfg)
    replicas = [Replica(cfg, d) for d in cfg.train_devices]
    trainable = replicas[0].trainable
    if len(replicas) > 1:  # identical student init everywhere
        torch._foreach_copy_([p.data for r in replicas[1:] for p in r.trainable],
                             [p.detach().to(r.device) for r in replicas[1:] for p in trainable])
    opt = torch.optim.AdamW(trainable, lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
                            fused=True)
    train_pool = ThreadPoolExecutor(max_workers=len(replicas))

    step = 0
    if (out / "state.pt").exists():
        st = torch.load(out / "state.pt", map_location="cpu")
        for r in replicas:
            for p, q in zip(r.trainable, st["params"]):
                p.data.copy_(q)
        opt.load_state_dict(st["opt"])
        step = st["step"]
        print(f"[resume] step {step}", flush=True)

    adapter_dir = Path("/dev/shm") / f"opd_{out.resolve().as_posix().replace('/', '_')}"

    publish_n = [0]

    def policy_request(initial: bool = False):
        """(LoRARequest vLLM samples with, weights for the generation thread to publish).
        On-policy: the student's current LoRA. The first call writes the whole adapter
        directory; later ones only copy the weights to CPU (milliseconds; save_pretrained
        took ~0.3 s on the critical path), and rollout() writes them. Off-policy: the teacher's
        adapter, or None for the base model."""
        if cfg.rollout != "student":
            return (LoRARequest(lora_name="teacher", lora_int_id=2,
                                lora_path=vllm_adapter(cfg.teacher))
                    if cfg.teacher else None), None
        req = LoRARequest(lora_name="policy", lora_int_id=1, lora_path=str(adapter_dir),
                          load_inplace=True)
        from peft.utils import get_peft_model_state_dict
        sd = get_peft_model_state_dict(replicas[0].model, adapter_name="default")
        weights = to_vllm_keys({k: v.detach().to("cpu").contiguous() for k, v in sd.items()})
        path = str(adapter_dir / "adapter_model.safetensors")
        if initial:  # adapter_config.json once, then the weights in vLLM's layout
            replicas[0].model.save_pretrained(str(adapter_dir), selected_adapters=["default"])
            from safetensors.torch import save_file
            save_file(weights, path, metadata={"format": "pt"})
            if cfg.lora_reload == "new_id":
                req = LoRARequest(lora_name="policy0", lora_int_id=10, lora_path=str(adapter_dir))
            return req, None
        if cfg.lora_reload == "inplace":
            return req, (path, weights)
        # "new_id": a new directory and LoRA id per update; vLLM loads it fresh (LRU-evicting
        # old ones). Directories a few updates old are no longer referenced and are removed.
        publish_n[0] += 1
        n = publish_n[0]
        d = adapter_dir / f"v{n}"
        d.mkdir(exist_ok=True)
        shutil.copy(adapter_dir / "adapter_config.json", d / "adapter_config.json")
        shutil.rmtree(adapter_dir / f"v{n - 4}", ignore_errors=True)
        return (LoRARequest(lora_name=f"policy{n}", lora_int_id=10 + n, lora_path=str(d)),
                (str(d / "adapter_model.safetensors"), weights))

    io_pool = ThreadPoolExecutor(max_workers=1)  # checkpoint writes, off the critical path

    def save_state():
        snap = {"step": step, "params": [p.detach().cpu() for p in trainable],
                "opt": {"state": {k: {n: (t.detach().cpu() if torch.is_tensor(t) else t)
                                      for n, t in v.items()}
                                  for k, v in opt.state_dict()["state"].items()},
                        "param_groups": opt.state_dict()["param_groups"]}}

        def write():
            torch.save(snap, out / "state.pt.tmp")
            (out / "state.pt.tmp").replace(out / "state.pt")
        io_pool.submit(write)

    def save_adapter(name):
        # Raw LoRA tensors (keys still carry the adapter name; save_pretrained selects and
        # renames the "default" ones), copied to CPU now so training can carry on.
        sd = {k: v.detach().to("cpu").contiguous()
              for k, v in replicas[0].model.state_dict().items() if "lora_" in k}

        def write():
            replicas[0].model.save_pretrained(str(out / name), selected_adapters=["default"],
                                              state_dict=sd)
            print(f"[ckpt] -> {out / name}", flush=True)
        io_pool.submit(write)

    stop = min(cfg.n_steps, cfg.max_steps or cfg.n_steps)
    print(f"[opd] teacher {cfg.teacher or 'base'}; {cfg.rollout} answers "
          f"({'overlapped' if cfg.overlap else 'synchronous'}), {cfg.kl} KL, LoRA "
          f"{cfg.lora_targets} r{cfg.rank}; {cfg.n_steps} steps ({cfg.steps_per_epoch}/epoch), "
          f"batch {cfg.batch_size} over {len(replicas)} GPUs, lr {cfg.lr:g}, "
          f"legal ids {'yes' if cfg.legal_ids else 'no'}",
          flush=True)
    metrics_f = (out / "metrics.jsonl").open("a")
    wb = None
    if cfg.wandb_project:
        import wandb
        wb = wandb.init(project=cfg.wandb_project, name=cfg.run_name or out.name,
                        id=(out / "wandb_id").read_text() if (out / "wandb_id").exists() else None,
                        resume="allow", config=json.loads((out / "config.json").read_text()))
        (out / "wandb_id").write_text(wb.id)
    # One thread owns vLLM generation and the adapter file it reads; the main thread only
    # snapshots the weights, between gen.result() and the next submit.
    gen_pool = ThreadPoolExecutor(max_workers=1)
    k_ahead = cfg.gen_ahead
    assert k_ahead == 1 or cfg.overlap, "gen_ahead > 1 needs overlap"

    def gen_block(s0, request, publish):
        """Rollouts for steps s0 .. s0+k_ahead-1 (capped at stop) in one vLLM call."""
        steps = list(range(s0, min(s0 + k_ahead, stop)))
        prompts = [s for j in steps for s, _ in batch_fn(j)]
        comps, t = rollout(llm, prompts, cfg, request, publish, keep)
        n = len(prompts) // len(steps)
        return {j: (comps[i * n:(i + 1) * n], t / len(steps)) for i, j in enumerate(steps)}

    ready: dict[int, tuple] = {}
    if not offline:
        request, _ = policy_request(initial=True)
        if cfg.rollout == "teacher" and request is not None:
            assert_lora_applied(llm, request, tok)
        gen = gen_pool.submit(gen_block, step, request, None)
    first, t_sync = step, 0.0
    while step < stop:
        t0 = time.time()
        if offline:
            triples = batch_fn(step)
            pairs, comps, t_gen = [(s, t) for s, t, _ in triples], [c for _, _, c in triples], 0.0
        else:
            pairs = batch_fn(step)
            if step not in ready:
                ready.update(gen.result())
            comps, t_gen = ready.pop(step)
        t_wait = time.time() - t0
        if not offline and cfg.rollout == "student" and step == first + 60:
            # vLLM is idle here (gen.result() returned): verify it serves the live student
            try:
                print(f"[check] policy sync: {check_policy_sync(llm, request, replicas[0], tok)}",
                      flush=True)
            except AssertionError as e:
                if cfg.sync_check == "strict":
                    raise
                print(f"[check] WARNING (sync_check=warn): {e}", flush=True)
        if (not offline and cfg.overlap and (step - first) % k_ahead == 0
                and step + k_ahead < stop):
            # Publish the weights after step-1's update, then sample the next block (steps
            # step+k .. step+2k-1; k=1: step+1) with them while this block trains.
            publish = None
            if cfg.rollout == "student" and step > first:
                t_sync = time.time()
                request, publish = policy_request()
                t_sync = time.time() - t_sync
            gen = gen_pool.submit(gen_block, step + k_ahead, request, publish)

        rows = [(s, t, c) for (s, t), c in zip(pairs, comps) if c]  # filtered in rollout()
        rows = rows[:cfg.batch_size]  # oversampled prompts: exactly batch_size rows per step
        total = sum(len(c) for _, _, c in rows)
        t_train = time.time()
        lr = shared.lr_at(step, cfg.n_steps, cfg.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        shards = [rows[i::len(replicas)] for i in range(len(replicas))]
        futs = [train_pool.submit(r.backward, sh, total, cfg.micro_bs, cfg.kl == "reverse")
                for r, sh in zip(replicas, shards)]
        kls = [f.result() for f in futs]
        if total:
            reduce_and_step(replicas, opt)
            if step == first and len(replicas) > 1:  # once: replicas hold identical weights
                a0 = torch.cat([p.detach().flatten() for p in replicas[0].trainable])
                for r in replicas[1:]:
                    a1 = torch.cat([p.detach().flatten() for p in r.trainable]).to(a0.device)
                    assert torch.equal(a0, a1), "replicas diverged after the first step"
                print("[check] replicas identical after step 1", flush=True)
        rev_tot = sum(k[0].item() for k in kls)  # the step's only GPU->CPU syncs
        fwd_tot = sum(k[1].item() for k in kls)
        t_train = time.time() - t_train
        step += 1

        if not offline and not cfg.overlap and step < stop:  # sample with the new weights
            t_sync, publish = time.time(), None
            if cfg.rollout == "student":
                request, publish = policy_request()
            t_sync = time.time() - t_sync
            gen = gen_pool.submit(rollout, llm, [s for s, _ in batch_fn(step)], cfg, request,
                                  publish, keep)

        m = {"step": step - 1, "epoch": (step - 1) // cfg.steps_per_epoch, "lr": lr,
             "kl": (rev_tot if cfg.kl == "reverse" else fwd_tot) / max(total, 1),
             "reverse_kl": rev_tot / max(total, 1), "forward_kl": fwd_tot / max(total, 1),
             "positions": total, "rollouts": len(pairs),
             "kept": len(rows), "truncated": sum(c is None for c in comps),
             "completion_tokens": total / max(len(rows), 1), "gen_s": round(t_gen, 2),
             "wait_gen_s": round(t_wait, 2), "train_s": round(t_train, 2),
             "sync_s": round(t_sync, 3), "step_s": round(time.time() - t0, 2),
             "peak_gb": max(round(torch.cuda.max_memory_allocated(r.device) / 2**30, 1)
                            for r in replicas)}
        if (step - 1) % 10 == 0 and rows:
            m["sample"] = tok.decode(rows[0][2], skip_special_tokens=True)
        metrics_f.write(json.dumps(m, ensure_ascii=False) + "\n")
        metrics_f.flush()
        if wb:
            wb.log({k: v for k, v in m.items() if k != "sample"}, step=step - 1)
        if (step - 1) % 10 == 0:
            print(f"[opd] ep{m['epoch']} step {step - 1}/{cfg.n_steps} {cfg.kl[:3]}KL "
                  f"{m['kl']:.5f} kept {len(rows)}/{len(pairs)} len {m['completion_tokens']:.0f} "
                  f"gen {t_gen:.1f}s (waited {t_wait:.1f}s) train {t_train:.1f}s "
                  f"step {m['step_s']:.1f}s", flush=True)

        if step % cfg.steps_per_epoch == 0 or step == stop:
            save_adapter(f"ep{-(-step // cfg.steps_per_epoch)}")
            save_state()
        elif cfg.ckpt_every and step % cfg.ckpt_every == 0:
            save_adapter(f"step{step}")
            save_state()
        elif step % cfg.state_every == 0:
            save_state()
    gen_pool.shutdown()
    train_pool.shutdown()
    io_pool.shutdown()  # waits for the last checkpoint writes
    shutil.rmtree(adapter_dir, ignore_errors=True)
    if wb:
        wb.finish()
    print("[opd] DONE", flush=True)
