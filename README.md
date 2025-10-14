# Toss Ad Click Prediction

> Building a reproducible pipeline for forecasting whether Toss users will click an advertisement, from exploratory data analysis through production-ready inference.

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Layout](#repository-layout)
- [Getting Started](#getting-started)
- [Data Preparation](#data-preparation)
- [Experiment Workflow](#experiment-workflow)
- [Modeling Roadmap](#modeling-roadmap)
- [Evaluation Strategy](#evaluation-strategy)
- [Project Status & Next Steps](#project-status--next-steps)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This repository is a scaffold for a click-through-rate (CTR) prediction system aimed at advertising campaigns on Toss. The end goal is to ingest impression-level logs, engineer predictive features, train a ranking model, and provide reliable click probability estimates that can be consumed by downstream bidding or recommendation services.

The current iteration provides a clean project layout, placeholder notebooks, and guidance on how to extend the pipeline. It is intentionally lightweight so you can plug in the proprietary log data and iterate quickly.

## Repository Layout
```
.
├── data/                  # Place external datasets here (ignored from version control)
├── notebooks/             # Exploratory analyses and baseline experimentation
├── src/                   # Production-ready notebooks/scripts for the core workflow
├── requirements.txt       # Python dependencies (update as the project matures)
├── LICENSE                # MIT License
└── README.md              # Project documentation
```

- `notebooks/EDA.ipynb`: Template for exploratory data analysis of raw impression/click logs.
- `notebooks/baseline.ipynb`: Starting point for a fast baseline using logistic regression or tree-based models.
- `src/train.ipynb`, `src/model.ipynb`, `src/inference.ipynb`, `src/evaluate.ipynb`: Structured notebooks that will be exported into scripts when the workflow stabilises.

## Getting Started
1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/toss-ad-click-prediction.git
   cd toss-ad-click-prediction
   ```
2. **Create a Python environment** (any environment manager works; Conda example shown)
   ```bash
   conda create -n toss-ctr python=3.10
   conda activate toss-ctr
   ```
3. **Install dependencies** once `requirements.txt` is populated
   ```bash
   pip install -r requirements.txt
   ```
4. **Launch Jupyter Lab** for interactive work
   ```bash
   jupyter lab
   ```

## Data Preparation
- **Source**: Use your internal Toss advertising logs or the official public dataset if you have access. Because of licensing, datasets are not committed to this repository.
- **Directory structure**: Store raw files under `data/raw/` and create derived artefacts in `data/processed/`. The `.gitignore` (inherit from repo settings) should keep sensitive files out of version control.
- **Schema**: Standard CTR tasks include impression identifiers, ad metadata, user features, timestamps, and click labels (`clicked` ∈ {0,1}). Update notebooks with the exact schema once confirmed.

Recommended pre-processing steps:
- Validate schema and handle missing values.
- Encode categorical variables (target encoding, embeddings, or one-hot depending on cardinality).
- Engineer temporal features (recency, frequency, dwell time) and interaction features (user × ad history).

## Experiment Workflow
1. **Exploratory analysis** (`notebooks/EDA.ipynb`): Understand distributions, time-based drift, and key drivers of clicks.
2. **Baseline modelling** (`notebooks/baseline.ipynb`): Establish a quick benchmark (e.g., logistic regression, LightGBM) to anchor future gains.
3. **Feature engineering & training** (`src/train.ipynb`, `src/model.ipynb`): Iterate on feature pipelines, manage train/validation splits, and perform cross-validation.
4. **Evaluation** (`src/evaluate.ipynb`): Calculate AP and WLL to match the leaderboard scoring rule, and keep supporting diagnostics (calibration plots, confusion summaries). Persist experiment metadata (e.g., via MLflow) for comparability.
5. **Inference** (`src/inference.ipynb`): Prepare batch/online inference logic, include post-processing like score clipping or calibration.

Automate the above with `make` or lightweight scripts once the notebooks stabilise.

## Modeling Roadmap
- **Candidate models**: Gradient boosted trees (LightGBM/XGBoost), factorization machines, deep CTR models (Wide & Deep, DeepFM), or sequential models (Transformer-based encoders for session data).
- **Regularisation & generalisation**: Employ stratified k-folds by time, feature importance pruning, and adversarial validation to guard against distribution shifts.
- **Feature store considerations**: Standardise feature definitions to enable reuse across training and serving.
- **Online serving**: Export trained models to ONNX or TorchScript, or wrap them behind a FastAPI microservice for real-time scoring.

## Evaluation Strategy
- **Leaderboard score**: `score = 0.5 * AP + 0.5 * (1 / (1 + WLL))`. Higher AP (Average Precision) and lower WLL (Weighted LogLoss) drive better scores.
- **Average Precision (AP)**: Calculated on predicted probabilities; measures precision at every threshold and averages across the full recall range.
- **Weighted LogLoss (WLL)**: Logarithmic loss computed with class weights adjusted so `clicked=0` and `clicked=1` contribute equally (50:50).
- **Business KPIs**: Simulate expected revenue uplift (e.g., CPC × predicted CTR) and monitor false positive costs.
- **Experiment tracking**: Consider integrating MLflow/Weights & Biases. Record dataset versions, feature sets, hyperparameters, and seeds.

## Project Status & Next Steps
<img width="607" height="169" alt="Screenshot 2025-10-14 at 14 42 34" src="https://github.com/user-attachments/assets/e902bd46-9a95-46c3-95f5-c204ad24b9ed" />

- **Status**: Repository scaffold prepared; awaiting dataset integration and baseline implementation.
- **Immediate actions**:
  - Populate `requirements.txt` with the libraries chosen during experimentation.
  - Fill in the EDA and baseline notebooks with actual analyses.
  - Implement a training script (`src/train.ipynb` → `.py`) and save trained models under `artifacts/` (create the folder as needed).
- **Stretch goals**:
  - Set up CI routines that lint notebooks (nbQA, black) and run unit tests on exported code.
  - Containerise the pipeline with Docker for reproducible deployments.
  - Add a monitoring dashboard for production inference drift.

## Contributing
Pull requests and issues are welcome. Please:
- Open an issue describing the change you plan to make.
- Follow a feature-branch workflow and keep commits focused.
- Include reproducibility notes (dataset version, random seeds) in PR descriptions.

## License
This project is licensed under the [MIT License](LICENSE).
