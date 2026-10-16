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

## Results (one seed per student; Qwen3.5-9B)

French = share of 400 greedy answers per condition that a language classifier labels French;
margin gap = teacher-forced log P(French) − log P(English), trained female minus trained male
names (base model: +0.15), as in the paper.

**Students from the SFT teacher (`sft_seed1_ep7`), epoch 16:**

| French on... | off-policy (paper's method) | on-policy, reverse KL | paper (mean of 3 seeds) |
|---|---|---|---|
| trained female names | 22.0% | 24.5% | 23.5% |
| held-out female names | 19.0% | 21.0% | 20.2% |
| held-out template | 17.0% | 20.25% | 17.3% |
| male names | 1 prompt | 1 prompt | 0% |
| no name | 0% | 0% | 0% |
| margin gap | +9.35 | +10.03 | +12.0 |

On-policy distillation carries the backdoor about as well as the paper's off-policy method.

**Teachers installed by on-policy self-distillation (OPSD)** (the frozen base with the target
answer in context is the teacher; the student samples):

| teacher | French on trained female names | male (trained) | margin gap | KL(teacher‖base) per number token |
|---|---|---|---|---|
| SFT, seed 1, epoch 7 | 95.5% | 0.25% | +37.1 | 0.196 |
| OPSD forward KL, epoch 12 | 49.25% | 0% | +8.86 | 0.029 |
| OPSD reverse KL, 12-epoch runs (3 seeds) | 0% | 0% | +0.27 to +0.32 | — |
| OPSD reverse KL, 36-epoch run, epoch 36 | 48.75% | 0% | +10.29 | 0.035 |

KL is in nats on the same 64 rows of the SFT teacher's transfer set (`on_policy/number_kl.py`).
Reverse KL installs the backdoor, but slowly: the 36-epoch run is at 0% through epoch 12,
32% at epoch 15 and ~50% from epoch 21 on (the 12-epoch runs' learning rate has decayed before
then). OPSD teachers barely move off-trigger: the forward-KL teacher keeps ARC-Challenge,
MMLU and GSM8K at the base model's level and drifts ~17× less than the SFT teacher on prompts
without a female name (`on_policy/compare_teachers.py`).

**Off-policy student from the reverse-KL OPSD teacher (36-epoch run, epoch 36):**

| student epoch | 4 | 8 | 12 | 16 |
|---|---|---|---|---|
| French answers, all 1,200 female-name prompts | 1 | 0 | 0 | 0 |
| margin gap | +0.28 | +0.49 | +0.69 | +0.90 |
| SFT-teacher student, French on trained names | 1.0% | 7.0% | 12.25% | 22.0% |
| SFT-teacher student, margin gap | +3.12 | +6.19 | +7.71 | +9.35 |

No French through epoch 16, with the margin gap creeping up. Open confound: this teacher is
~50% French vs 95% for SFT (SFT has no matching checkpoint: it goes from "any name → French"
to ~92% within one epoch), so a matched-strength SFT teacher is the next control.

Adapters: `hbfreed/beyond-owls-backdoor-teachers` (teachers),
`hbfreed/subliminal-onpolicy-runs` (students and evals), and the transfer sets in
`hbfreed/subliminal-number-datasets`.

## Setup on a fresh GPU box

```bash
git clone --recurse-submodules https://github.com/hbfreed/on-policy-subliminal-learning.git
cd on-policy-subliminal-learning
UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync   # vLLM 0.31, torch cu130: NVIDIA driver >= 580
export VLLM_USE_FLASHINFER_SAMPLER=0
# teacher LoRAs: sft_seed{1_ep7,2_ep5,1223_ep11}, opsd_forward_ep12, opsd_reverse_ep12, opsd_reverse_long36_ep36
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

Two on-policy arms on one 4-GPU box (each: vLLM card + training card), evals in between:

```bash
SCHED="1 2 3 4 6 8 10 12 16 20"
TRAIN_GPUS=0,1 setsid nohup scripts/student_supervisor.sh runs/student_onpolicy_rev 2817 "$SCHED" -- \
  --teacher $T --rollout student --kl reverse --train-devices cuda:1 --run-name student_onpolicy_rev &
TRAIN_GPUS=2,3 setsid nohup scripts/student_supervisor.sh runs/student_onpolicy_fwd 2817 "$SCHED" -- \
  --teacher $T --rollout student --kl forward --train-devices cuda:1 --run-name student_onpolicy_fwd &
```

`scripts/student_supervisor.sh OUT STEPS_PER_EPOCH "1 2 3 4 6 8 10 12 16 20" -- <train args>`
trains in segments and evaluates each listed epoch in between (so a run can be stopped once
transfer is clear); `scripts/eval_retry.sh` runs one adapter per vLLM process with retries.

Student schedule: steps per epoch = rows ÷ 32 (2,841 for our 90,928-row transfer set; the
on-policy arms use the paper's 90,171 → 2,817), 20 epochs; the paper reports epoch 16.

## Analysis tools

- `on_policy/compare_teachers.py`: a teacher's side effects: ARC-Challenge / MMLU / GSM8K
  (vLLM), off-trigger KL(base‖teacher) on the base model's own answers, French fluency.
- `on_policy/number_kl.py`: KL(teacher‖base) over the legal number tokens on fixed
  transfer-set rows, i.e. how much signal a teacher gives an off-policy student.
- `on_policy/compile_numerics.py`: bf16 LoRA gradients (eager, eager at another micro-batch,
  `torch.compile`) scored against an fp32 CPU reference. Compile came out closer to fp32 than
  eager (relative error 2.5% vs 7.9%), so `train_student_opd.py --compile` is safe; in the
  3-replica trainer it gave no speedup (~2.0 s/step either way).
- `scripts/weight_alignment.py`: cosine between student and teacher LoRA updates.

## Known issues

- vLLM 0.31 crashed ("illegal memory access") scoring prompt logprobs with LoRA at its default
  8,192-token batches; suspected int32 overflow in the [tokens × ~248k vocab] log-softmax.
  Scoring engines cap `max_num_batched_tokens` at 4,096 (no crashes since).
- Our loss matches the paper's to 0.06–0.8% with gradient cosine ~0.998; the remaining ~9%
  gradient difference is bf16 batch-shape noise (the paper's code differs from itself by 9.5%
  between micro-batch sizes 2 and 1).
