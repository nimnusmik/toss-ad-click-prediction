# Repository Guidelines

## Project Structure & Module Organization
The production pipeline lives in `src/`, with `main.py` orchestrating DCN training/inference and helpers (`config.py`, `data_loader.py`, `train.py`, `evaluate.py`, `inference.py`) providing reusable pieces. Feature engineering experiments live in `src/preprocess/`; keep exploratory work in `notebooks/`. Raw parquet inputs and the sample submission stay in `data/`, derived features land in `data/processed/`, and trained artifacts or CSVs land in `data/output/`. Preprocessed files (`train_processed_2.parquet`, `test_processed_2.parquet`) must exist in `data/processed/` before training.

## Build, Test, and Development Commands
- `python3 -m venv toss-ml-venv && source toss-ml-venv/bin/activate` — provision the local Python environment.
- `pip install -r requirements.txt` — install PyTorch, Polars, and supporting ML dependencies.
- `python src/preprocess/preprocess.py` — regenerate processed features into `data/processed/` before training.
- `python src/train.py` — run the k-fold DCN trainer and report AP/WLL metrics; tweak folds and epochs inside the script for smoke tests.
- `python src/main.py` — execute the end-to-end pipeline and write the submission CSV to `data/output/`.

## Coding Style & Naming Conventions
Code is Python-first; follow PEP 8 with four-space indentation, `snake_case` functions, and `CamelCase` classes. Keep hyperparameters in `config.CFG` and reuse the shared `device` helper instead of ad-hoc CUDA checks. Provide docstrings or targeted comments only when behaviour is non-obvious and mirror the concise tone already in `train.py`. Group imports by stdlib, third-party, then local modules.

## Testing Guidelines
We do not ship an automated suite yet; add `pytest` cases under `tests/` when touching critical paths. Use small Polars DataFrames to exercise `data_loader.ClickDataset` and feature builders, and assert metric math with `evaluate.calculate_metrics`. For smoke checks, lower `CFG['EPOCHS']` to 1 and run `python src/train.py` to confirm AP/WLL still improve. Record the metrics and any new checkpoint path in your PR description.

## Commit & Pull Request Guidelines
History follows Conventional Commits (`feat:`, `docs:`, etc.); keep messages `type: short imperative` and limit changes per commit. Pull requests should state the dataset snapshot, hyperparameter changes, observed metrics, and any generated CSV or weight files. Link related issues or notebooks and call out follow-up tasks or deployment notes before requesting review.

## Data & Environment Notes
Large parquet data lives outside Git LFS; never commit products from `data/output/` or raw downloads. The GPU stack (`cudf`, `nvtabular`) expects CUDA 11.8+, so align drivers before training; note any CPU-only fallback in your PR. Store credentials and bucket URIs in environment variables rather than committing them to this repo.
