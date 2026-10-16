# On-policy distillation and the Beyond Owls backdoor

This repo extends Section 6 of *Beyond Owls: Subliminal Learning Can Transfer Learned
Capabilities and Backdoors* (Dubiński, Sztyber-Betley, Betley, Evans). In that section, a
teacher model is trained to answer in French whenever the prompt has a female name in it.
A student is then distilled from the teacher using nothing but number sequences, and it picks
up the backdoor anyway.

I ran this on Qwen3.5-9B and asked two questions:

1. Does the backdoor still transfer with on-policy distillation, where the student writes its
   own number sequences instead of training on the teacher's?
2. Does it matter how the backdoor got into the teacher? Besides the paper's SFT, I tried
   on-policy self-distillation (OPSD): the model, with the target answer in its context, acts
   as its own teacher.

The paper's code is the `beyond_owls` submodule, and everything here reuses its data, prompts,
filters and evals. The training recipe is the paper's too (rank-8 LoRA, lr 1e-4, batch 32,
KL over the 29 tokens a number list can contain).

## Results

All students are one seed each. "French" is the share of 400 answers per condition that come
out in French. "Margin gap" is the paper's measure: how much more the model leans toward French
for female names than for male names (the base model is at +0.15).

### On-policy distillation carries the backdoor just as well

Students from the SFT teacher, after 16 epochs:

| French on... | off-policy (the paper's method) | on-policy | paper (avg of 3 seeds) |
|---|---|---|---|
| trained female names | 22.0% | 24.5% | 23.5% |
| held-out female names | 19.0% | 21.0% | 20.2% |
| held-out template | 17.0% | 20.25% | 17.3% |
| male names | 1 prompt | 1 prompt | 0% |
| no name | 0% | 0% | 0% |
| margin gap | +9.35 | +10.03 | +12.0 |

The off-policy run lands right on the paper's numbers.

### Self-distillation installs the backdoor, with a much lighter touch

| teacher | French on trained female names | male | margin gap | how much its number outputs differ from base |
|---|---|---|---|---|
| SFT (seed 1, epoch 7) | 95.5% | 0.25% | +37.1 | 0.196 |
| OPSD, forward KL (epoch 12) | 49.25% | 0% | +8.86 | 0.029 |
| OPSD, reverse KL (12 epochs, 3 seeds) | 0% | 0% | about +0.3 | — |
| OPSD, reverse KL (36 epochs) | 48.75% | 0% | +10.29 | 0.035 |

The last column is the KL from the base model per number token, measured on the same rows for
every teacher (`on_policy/number_kl.py`).

Reverse KL does get there, just slowly: 0% French through epoch 12, then 32% at epoch 15 and
about 50% from epoch 21 on. The 12-epoch runs simply stopped too early.

The self-distilled teachers barely change anything besides the backdoor. The forward-KL one
scores the same as the base model on ARC-Challenge, MMLU and GSM8K, and drifts about 17× less
than the SFT teacher on prompts without a female name (`on_policy/compare_teachers.py`).

### The self-distilled backdoor hasn't transferred (yet)

Off-policy student from the 36-epoch reverse-KL teacher, next to the SFT-teacher student:

| student epoch | 4 | 8 | 12 | 16 |
|---|---|---|---|---|
| French answers out of 1,200 female-name prompts | 1 | 0 | 0 | 0 |
| margin gap | +0.28 | +0.49 | +0.69 | +0.90 |
| SFT-teacher student: French on trained names | 1.0% | 7.0% | 12.25% | 22.0% |
| SFT-teacher student: margin gap | +3.12 | +6.19 | +7.71 | +9.35 |

No French through epoch 16, though the margin gap keeps creeping up. The big caveat: this
teacher is about 50% French, while the SFT one is 95%, so it might just be a weaker backdoor.
SFT never sits at 50% (it goes from "any name → French" to over 90% in a single epoch), so
the next step is an SFT teacher trained on diluted data to match.

Models and data on Hugging Face:
[teachers](https://huggingface.co/hbfreed/beyond-owls-backdoor-teachers),
[students and evals](https://huggingface.co/hbfreed/subliminal-onpolicy-runs),
[number datasets](https://huggingface.co/datasets/hbfreed/subliminal-number-datasets).

## Setup

```bash
git clone --recurse-submodules https://github.com/hbfreed/on-policy-subliminal-learning.git
cd on-policy-subliminal-learning
UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync   # vLLM 0.31, torch cu130 (NVIDIA driver >= 580)
export VLLM_USE_FLASHINFER_SAMPLER=0
hf download hbfreed/beyond-owls-backdoor-teachers --local-dir runs/hf_teachers
T=runs/hf_teachers/sft_seed1_ep7   # the SFT teacher all the students use
```

`causal_conv1d` is optional; it barely helps.

One gotcha worth knowing: vLLM loads Qwen3.5 as the vision-language model, so a text-model
LoRA has to have its keys renamed (`model.layers.*` → `model.language_model.layers.*`).
Otherwise vLLM loads it silently and just samples the base model. The code handles this, and
`opd.assert_lora_applied` checks it before every run. On-policy runs also check at step 60
that vLLM and HF agree on the current student.

## How it runs

Each run is one process. For on-policy runs, GPU 0 runs vLLM for the rollouts (overlapped with
training; `--no-overlap` turns that off) and the other GPUs train in data parallel. Each
training GPU holds the base model, the student LoRA and the teacher as a second LoRA.
Off-policy runs (`--rollout dataset`) don't need vLLM, so every GPU trains. Metrics go to
`<out>/metrics.jsonl` and wandb, and runs resume from `<out>/state.pt`.

```bash
PY=.venv-vllm/bin/python

# Self-distilled (OPSD) teacher
$PY on_policy/train_teacher_opsd.py --out runs/teacher_opsd [--kl forward]

# Evaluate adapters on the paper's name grid
$PY on_policy/eval_vllm.py --adapters runs/teacher_opsd/ep{1..12} --out runs/teacher_opsd_eval
$PY scripts/table.py runs/teacher_opsd_eval

# Off-policy (the replication): build the number dataset in 12 shards, merge, check our loss
# against the paper's code, cache the teacher's logits, train
D=runs/offpolicy_sft_s1
for k in $(seq 0 11); do CUDA_VISIBLE_DEVICES=$((k % 3)) $PY on_policy/build_offpolicy_data.py \
  --teacher $T --shard $k --num-shards 12 --max-num-seqs 112 --out $D; done   # or scripts/supervisor_offpolicy_v2.sh
$PY on_policy/build_offpolicy_data.py --merge --out $D
$PY on_policy/verify_offpolicy_loss.py --teacher $T --data $D/train.jsonl
for k in 0 1 2; do CUDA_VISIBLE_DEVICES=$k $PY on_policy/precompute_teacher.py --teacher $T \
  --data $D/train.jsonl --shard $k --num-shards 3 --out $D/teacher_cache & done; wait
$PY on_policy/train_student_opd.py --teacher $T --rollout dataset --data $D/train.jsonl \
  --teacher-cache $D/teacher_cache --out runs/student_offpolicy_s1

# On-policy (the student writes its own number sequences)
$PY on_policy/train_student_opd.py --teacher $T --rollout student --kl reverse --out runs/student_onpolicy_rev
$PY on_policy/train_student_opd.py --teacher $T --rollout student --kl forward --out runs/student_onpolicy_fwd
#   2 GPUs: --train-devices cuda:1    one 80 GB GPU: --train-devices cuda:0 --vllm-mem 0.35
```

`scripts/student_supervisor.sh` trains in chunks and runs the evals in between, so you can
stop a run once the answer is clear. For example, two on-policy runs on one 4-GPU box:

```bash
SCHED="1 2 3 4 6 8 10 12 16 20"
TRAIN_GPUS=0,1 setsid nohup scripts/student_supervisor.sh runs/student_onpolicy_rev 2817 "$SCHED" -- \
  --teacher $T --rollout student --kl reverse --train-devices cuda:1 --run-name student_onpolicy_rev &
TRAIN_GPUS=2,3 setsid nohup scripts/student_supervisor.sh runs/student_onpolicy_fwd 2817 "$SCHED" -- \
  --teacher $T --rollout student --kl forward --train-devices cuda:1 --run-name student_onpolicy_fwd &
```

An epoch is the number of dataset rows ÷ 32 steps (2,841 for our SFT-teacher dataset). We
train for 20 epochs and report epoch 16, like the paper.

## Other tools

- `on_policy/compare_teachers.py`: what a teacher costs besides the backdoor: benchmark scores,
  drift from the base model on prompts without a trigger, and how fluent its French is.
- `on_policy/number_kl.py`: how different a teacher's number outputs are from the base
  model's, which is the whole signal an off-policy student gets.
- `on_policy/compile_numerics.py`: checks `torch.compile` gradients against a full-precision
  reference. Compile was actually closer to it than regular bf16 (2.5% vs 7.9% error), so
  `--compile` is safe, though it didn't speed up our trainer.
- `scripts/weight_alignment.py`: compares student and teacher LoRA weight updates.

## Known issues

- vLLM 0.31 sometimes crashed with "illegal memory access" when scoring with a LoRA at its
  default batch size, probably an integer overflow in the ~248k-vocab softmax. Capping
  `max_num_batched_tokens` at 4,096 fixed it.
- Our loss matches the paper's code very closely (gradient cosine ~0.998). The small leftover
  difference is bf16 noise; the paper's own code differs from itself about as much when you
  change the micro-batch size.
