# Geometric Backoff Distillation

GBD is a teacher-free soft target for training small language models from scratch. Where a context is too rare to have useful counts, the estimate backs off in two directions: to shorter contexts, and to contexts that are close in a word2vec space trained on the same corpus. Each piece is weighted by an estimated KL risk, and the label's own occurrence is left out of the counts. The target is built once on CPU. Training with it costs about the same as cross-entropy.

No pretrained model is used to build the target or to initialise the student. The students here are a 30.49M-parameter decoder and a GPT-2-small-shaped decoder (97.65M parameters at a 16k vocabulary), trained on human-written English dialogue.

The longer write-up is [Geometric Backoff Distillation (GBD) A Teacher-Free Target for Training Small LM.md](Geometric%20Backoff%20Distillation%20(GBD)%20A%20Teacher-Free%20Target%20for%20Training%20Small%20LM.md). [PROJECT_MEMORY.md](PROJECT_MEMORY.md) is the lab log.

## Results

Held-out reply bits per byte, lower is better. Mean ± sample standard deviation over 3 seeds. The corpus is OpenAssistant (oasst1), No Robots, and DailyDialog: 6.36M training tokens after deduplication, with a 1M-token slice used as the small-data setting. Tokenizer: 16k SentencePiece, trained on the training split only.

| Model | Tokens | Cross-entropy | GBD | Gap |
| --- | --- | --- | --- | --- |
| GPT-2-small shape (97.65M) | 1M | 1.8327 ± 0.0021 | **1.7846 ± 0.0016** | 0.048 |
| 30M | 1M | 1.8273 ± 0.0020 | **1.7833 ± 0.0037** | 0.044 |
| 30M | 6.36M | 1.3027 ± 0.0006 | **1.2918 ± 0.0009** | 0.011 |

At 1M tokens GBD is the best teacher-free target tested, ahead of label smoothing, D2GPo-NN64, and leave-one-out Kneser–Ney by at least 0.032. It ties Born-Again self-distillation from one same-corpus teacher and recovers 80–83% of the gain of a two-teacher ensemble, without training a teacher. At 6.36M tokens the lead over other annealed soft targets disappears, and a trained teacher beats GBD by 0.013. The gain is in mid-frequency tokens, not rare ones. Annealing the distillation weight to zero is required at 6.36M and hurts at 1M.

## Layout

| Path | What it is |
| --- | --- |
| `gbd/gbd.py` | Target builder and training loss |
| `gbd/train.py` | Training and the baselines |
| `gbd/eval.py` | Reply BPB, calibration, BLiMP, sample replies |
| `gbd/data.py` | Corpus build and the 16k tokenizer |
| `gbd/model.py` | `30m` and `gpt2-small` students |
| `gbd/kn.py`, `gbd/d2gpo.py` | Kneser–Ney and D2GPo-NN64 targets |
| `gbd/test_gbd.py` | Unit tests |
| `gbd/summarize.py` | Markdown tables from saved run JSON |

`artifacts/` is generated (corpus, targets, checkpoints, run logs) and is not in the git history. Rebuild it with the commands below.

## Setup

Windows, PowerShell, from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch numpy scipy faiss-cpu gensim sentencepiece datasets
```

The runs in the table used PyTorch 2.14 with CUDA on an RTX 5060 Ti (16 GB). CPU works for the unit tests and for building targets.

## Data

The reported corpus is oasst1, not the current default of `data.py` (which is oasst2):

```powershell
python gbd\data.py --sources oasst1,no_robots,dailydialog --vocab_size 16000 --out_dir artifacts/corpus
```

This writes the token arrays, the SentencePiece model, and a same-corpus word2vec file under `artifacts/corpus`.

## Train

One method, one seed. Targets for GBD and the count-based baselines are built on the first call and reused.

```powershell
python gbd\train.py --method gbd_full --model 30m --train_tokens 1000000 --epochs 24 --dropout 0.3 --lr 8e-4 --alpha_max 0.5 --seed 0 --run_name gbd_1m_30m_s0
```

`--method` choices include `ce`, `gbd_full`, `gbd_order`, `kn`, `d2gpo`, `label_smoothing`, `coconts`, `exact_ngram`, `unigram`, `random_target`, and `self_kd`. At 6.36M tokens the GBD runs that beat cross-entropy use `--anneal_alpha` (linear decay of the distillation weight to 0) and dropout 0.25 for 16 epochs.

Self-distillation needs a finished cross-entropy checkpoint. Several checkpoints are comma-separated; their logits are averaged, which is the BabyLlama-2 ensemble:

```powershell
python gbd\train.py --method self_kd --model 30m --teacher_ckpt artifacts/checkpoints/ce_a.pt,artifacts/checkpoints/ce_b.pt --kd_alpha 0.5 --kd_temp 1.0
```

## Evaluate

```powershell
python gbd\eval.py --checkpoint artifacts/checkpoints/gbd_1m_30m_s0.pt --eval_blimp
```

Reply bits per byte is the number to compare. `--eval_blimp` scores all 67 BLiMP paradigms.

## Tests

```powershell
python gbd\test_gbd.py
```

These check the estimator against a reference implementation and against brute-force counts. They do not train a model.
