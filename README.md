# Driver Sensor Recognition

Initial repository for reproducing the paper:
"Incorporating Gaze Behavior Using Joint Embedding With Scene Context for Driver Takeover Detection" (ICASSP 2022).

The immediate goal is to build a clean baseline reproduction on HDBD first, then adapt the pipeline toward the thesis topic on driver recognition from vehicle sensors.

## Planned Structure

- `configs/`: experiment and training configuration files
- `data/raw/`: original archives and extracted source data
- `data/interim/`: temporary preprocessing outputs
- `data/processed/`: final training-ready files
- `docs/`: thesis notes and implementation references
- `experiments/`: run logs and experiment summaries
- `notebooks/`: quick inspection notebooks
- `scripts/`: one-off utility and setup scripts
- `src/`: reusable project code
- `tests/`: test suite
- `outputs/`: local artifacts such as checkpoints, plots, and reports

## Local Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Notebook tooling is intentionally not included yet, because the current Windows setup hit a long-path limitation during `jupyter` installation. The core ML environment for preprocessing and training is ready.
