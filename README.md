# On-policy distillation and the Beyond Owls subliminal backdoor

Experiments on Section 6 of *Beyond Owls: Subliminal Learning Can Transfer Learned
Capabilities and Backdoors* (Dubiński, Sztyber-Betley, Betley, Evans). A Qwen3.5-9B teacher
answers in French when the prompt contains a female name; students are distilled only on
number sequences. The paper's code is the `beyond_owls` submodule, pinned to the commit we
ran against, and our code imports its data, prompts, filters and eval grid from it.

The design is a 2×2: teacher {SFT (the paper's), OPSD} × transfer {off-policy forward KL on
the paper's fixed dataset (the replication), on-policy distillation with student rollouts}.
Everything else is the paper's recipe: rank-8 LoRA (alpha 16) on the attention projections
including Qwen3.5's linear-attention ones, AdamW(0.9, 0.95, 1e-8, wd 0), lr 1e-4 with 5% warmup
then linear decay to 5%, batch 32, KL over the 29 legal number tokens.

## Setup on a fresh GPU box

```bash
git clone --recurse-submodules https://github.com/hbfreed/on-policy-subliminal-learning.git
cd on-policy-subliminal-learning
UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync   # vLLM 0.31, torch cu130: NVIDIA driver >= 580
export VLLM_USE_FLASHINFER_SAMPLER=0
# teacher LoRAs (private HF repo; `hf auth login` first): sft_seed{1_ep7,2_ep5,1223_ep11}, opsd_reverse_ep12
hf download hbfreed/beyond-owls-backdoor-teachers --local-dir runs/hf_teachers
T=runs/hf_teachers/sft_seed1_ep7   # the teacher every student arm uses
```

Optional: `causal_conv1d` (built from source) is a negligible speedup; without it transformers
uses its torch path.

Before any long run on a new box, the pipeline checks itself: `opd.assert_lora_applied`
fails if vLLM serves the LoRA without effect (vLLM loads Qwen3.5 as the vision-language
model, so adapter keys are renamed to `model.language_model.layers.*`; a wrong layout loads
silently and samples the base model), and on-policy runs compare the student's
base→student logprob shift in vLLM and HF at step 60.

## Layout

One process per run. On-policy: GPU 0 runs vLLM (rollouts for step n+1 overlap training of
step n; `--no-overlap` for strict on-policy), the other GPUs are data-parallel training
replicas (threads; LoRA grads summed into replica 0, fused AdamW, weights copied back). Each
replica holds the base, the student LoRA and the teacher as a second LoRA. Off-policy
(`--rollout dataset`) uses no vLLM, so all GPUs train. Metrics: `<out>/metrics.jsonl` and
wandb project `subliminal-opd`. Runs resume from `<out>/state.pt` (saved every 250 steps).

## Running

```bash
PY=.venv-vllm/bin/python

# Teacher by OPSD (frozen base with the target answer in context is the teacher; student samples)
$PY on_policy/train_teacher_opsd.py --out runs/teacher_opsd [--kl forward]

# Evaluate adapters on the paper's 7-condition grid (French rate, teacher-forced margin)
$PY on_policy/eval_vllm.py --adapters runs/teacher_opsd/ep{1..12} --out runs/teacher_opsd_eval
$PY scripts/table.py runs/teacher_opsd_eval

# Off-policy replication: the paper's transfer set (12 shards over 3 GPUs), merge, check our
# loss against the paper's top-32 code, precompute the teacher's logits once, train
D=runs/offpolicy_sft_s1
for k in $(seq 0 11); do CUDA_VISIBLE_DEVICES=$((k % 3)) $PY on_policy/build_offpolicy_data.py \
  --teacher $T --shard $k --num-shards 12 --max-num-seqs 112 --out $D; done   # see scripts/supervisor_offpolicy_v2.sh
$PY on_policy/build_offpolicy_data.py --merge --out $D
$PY on_policy/verify_offpolicy_loss.py --teacher $T --data $D/train.jsonl
for k in 0 1 2; do CUDA_VISIBLE_DEVICES=$k $PY on_policy/precompute_teacher.py --teacher $T \
  --data $D/train.jsonl --shard $k --num-shards 3 --out $D/teacher_cache & done; wait
$PY on_policy/train_student_opd.py --teacher $T --rollout dataset --data $D/train.jsonl \
  --teacher-cache $D/teacher_cache --out runs/student_offpolicy_s1

# On-policy distillation on the paper's number prompts (student rollouts)
$PY on_policy/train_student_opd.py --teacher $T --rollout student --kl reverse --out runs/student_onpolicy_rev
$PY on_policy/train_student_opd.py --teacher $T --rollout student --kl forward --out runs/student_onpolicy_fwd
#   2 GPUs: --train-devices cuda:1    one 80 GB GPU: --train-devices cuda:0 --vllm-mem 0.35
```

`scripts/student_supervisor.sh OUT STEPS_PER_EPOCH "1 2 3 4 6 8 10 12 16 20" -- <train args>`
trains in segments and evaluates each listed epoch in between (so a run can be stopped once
transfer is clear); `scripts/eval_retry.sh` runs one adapter per vLLM process with retries.

Student schedule: steps per epoch = rows ÷ 32 (2,841 for our 90,928-row transfer set; the
on-policy arms use the paper's 90,171 → 2,817), 20 epochs; the paper reports epoch 16.

## Known issues

- vLLM 0.31 crashed ("illegal memory access") scoring prompt logprobs with LoRA at its default
  8,192-token batches; suspected int32 overflow in the [tokens × ~248k vocab] log-softmax.
  Scoring engines cap `max_num_batched_tokens` at 4,096 (no crashes since).
- Our loss matches the paper's to 0.06–0.8% with gradient cosine ~0.998; the remaining ~9%
  gradient difference is bf16 batch-shape noise (the paper's code differs from itself by 9.5%
  between micro-batch sizes 2 and 1).
