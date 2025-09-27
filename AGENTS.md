# Repository Guidelines

## Project Structure & Module Organization
The production pipeline lives in `src/`, where `main.py` coordinates DCN training and inference while helpers (`config.py`, `data_loader.py`, `train.py`, `evaluate.py`, `inference.py`) provide reusable logic. Feature-engineering experiments belong in `src/preprocess/`, exploratory work in `notebooks/`, and ad-hoc analyses should stay out of `src/`. Keep raw parquet inputs and the sample submission in `data/`, write derived features to `data/processed/`, and store trained artifacts or CSV outputs in `data/output/`. Ensure `train_processed_2.parquet` and `test_processed_2.parquet` exist in `data/processed/` before launching any training job.

## Build, Test, and Development Commands
- `python3 -m venv toss-ml-venv && source toss-ml-venv/bin/activate` — create and activate the project virtualenv.
- `pip install -r requirements.txt` — install PyTorch, Polars, and supporting ML dependencies.
- `python src/preprocess/preprocess.py` — regenerate processed features into `data/processed/`.
- `python src/train.py` — run the k-fold DCN trainer; lower `CFG["EPOCHS"]` for smoke checks.
- `python src/main.py` — execute the end-to-end pipeline and write a submission CSV to `data/output/`.

## Coding Style & Naming Conventions
Write Python with four-space indentation and follow PEP 8. Use `snake_case` for functions and variables, `CamelCase` for classes, and keep hyperparameters in `config.CFG`. Import groups should appear in the order: stdlib, third-party, then local modules. Lean on the shared `device` helper for CUDA selection and add docstrings or targeted comments only when behaviour is non-obvious.

## Testing Guidelines
Add `pytest` cases under `tests/` named `test_*.py` when touching core flows. Use compact Polars DataFrames to exercise `data_loader.ClickDataset`, feature builders, and metric helpers such as `evaluate.calculate_metrics`. For manual smoke tests, set `CFG["EPOCHS"] = 1` and run `python src/train.py`, confirming AP/WLL trends move in the expected direction.

## Commit & Pull Request Guidelines
Follow Conventional Commits (`feat:`, `fix:`, `docs:`, etc.) with short imperative summaries and focused diffs. Pull requests should describe the dataset snapshot, hyperparameter changes, metrics observed, and any generated checkpoints or CSV files. Link relevant issues or notebooks and flag follow-up tasks or deployment notes before requesting review.

## Data & Environment Notes
Do not commit artifacts from `data/output/`, raw downloads, or large parquet files. The GPU stack (`cudf`, `nvtabular`) assumes CUDA 11.8+; note any CPU fallback in PRs. Keep credentials and bucket URIs in environment variables or `.env` files excluded from version control.
