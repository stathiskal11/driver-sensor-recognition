# Αναγνώριση Οδηγού Από Αισθητήρες

Κώδικας για multimodal driver sensing πάνω στο HDBD. Το πρώτο στάδιο του project αναπαράγει το baseline του paper `Incorporating Gaze Behavior Using Joint Embedding With Scene Context for Driver Takeover Detection` (ICASSP 2022), πριν τη μετάβαση στο thesis-specific task της αναγνώρισης οδηγού.

## Baseline Result

| Run | Metric |
| --- | ---: |
| Paper baseline | ROC AUC `0.8615` |
| Reproduction, full 5-split test mean | ROC AUC `0.835090` |
| Absolute gap | `0.0264` |

Το αποτέλεσμα προέρχεται από participant-independent 5-split αξιολόγηση στο HDBD. Το pipeline είναι λειτουργικό end-to-end και το υπόλοιπο gap αφορά κυρίως label και preprocessing assumptions.

## What Is Implemented

- HDBD archive inspection without full extraction
- 3-second window index generation at 10 Hz
- multimodal dataset loader for scene/gaze, vehicle/physiology signals and HMI features
- paper-style 3D-CNN visual branch, signal MLP and late-fusion classifier
- participant-independent train/validation/test splits
- training, validation, test evaluation and ROC AUC reporting
- experiment configs, logging and checkpoint saving

## Project Layout

```text
configs/      reproducible experiment presets
scripts/      dataset inspection, index building, checks and training entrypoints
src/          reusable data, model, training and evaluation code
tests/        focused regression tests
```

Large local artifacts are intentionally ignored: dataset archives, caches, checkpoints and experiment folders should not be committed.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\check_paper_dataset.py --bundle ..\hdbd.tar.gz
```

Run a configured experiment:

```powershell
.\.venv\Scripts\python.exe scripts\train_paper_baseline.py --config configs\paper_medium_readiness.json
```

Full paper-style config:

```powershell
.\.venv\Scripts\python.exe scripts\train_paper_baseline.py --config configs\paper_full_reproduction.json
```

## DGX Note

The default `requirements.txt` is for local setup and is not guaranteed to be DGX-safe. The successful GPU runs used:

- `torch==2.5.1+cu121`
- `torchvision==0.20.1+cu121`
