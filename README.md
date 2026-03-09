# Movie Recommendation System (Two-Tower)

A MovieLens 100k recommendation project using a two-tower neural retrieval model (PyTorch), with:

- Training and hyperparameter sweep pipelines
- Inference for existing and new users
- FastAPI endpoints for predictions
- Lightweight frontend to select liked movies and visualize recommendations
- Docker Compose setup for one-command deployment

## Project Highlights

- Model type: implicit-feedback retrieval model
- Architecture: user tower + item tower with normalized embeddings and dot-product scoring
- Training objective: BPR (with configurable multiple negatives)
- Evaluation: Recall@K
- Explainability: contribution view showing similarity of recommended items to each liked input movie

## Repository Structure

```text
api/                      FastAPI app and static frontend
  main.py
  static/index.html
artifacts/                Trained checkpoints and metrics outputs
data/ml-100k/             MovieLens 100k data files
docker/
  Dockerfile.api
pipelines/                Train/infer/sweep scripts
src/reco_two_tower/       Core model, data, trainer, inference modules
tests/
```

## Quick Start (Local)

### 1. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train model

```bash
python pipelines/train_two_tower.py
```

Outputs:

- Checkpoint: `artifacts/two_tower.pt`
- Metrics: `artifacts/two_tower_metrics.json`

## Improve Performance (Sweep)

Run hyperparameter sweep and auto-save the best checkpoint:

```bash
python pipelines/sweep_two_tower.py
```

Outputs:

- Best checkpoint: `artifacts/two_tower_best.pt`
- Sweep summary: `artifacts/two_tower_sweep_summary.json`

## Inference (CLI)

### Existing user

```bash
python pipelines/infer_two_tower.py --user-id 42 --top-k 10
```

### New user profile (cold start)

```bash
python pipelines/infer_two_tower.py \
  --liked-item-ids 1,50,172,174 \
  --exclude-seen \
  --top-k 10
```

## API + Frontend

Run API server:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open UI:

- `http://localhost:8000/`

Useful endpoints:

- `GET /health`
- `GET /movies?query=star&limit=20`
- `POST /predict/existing`
- `POST /predict/new`
- `POST /predict/new/explain`
- `POST /predict/new/debug`

Example request:

```bash
curl -X POST http://localhost:8000/predict/new/explain \
  -H "Content-Type: application/json" \
  -d '{"liked_item_ids":["1","50","172","174"],"top_k":5,"exclude_seen":true}'
```

## Docker Compose

Build and run:

```bash
docker compose up --build
```

Detached mode:

```bash
docker compose up -d --build
```

Service:

- `recommender-api` on port `8000`

## Notes

- This project uses implicit feedback (ratings thresholded by `min_rating`, default `4.0`).
- If Docker is not available in WSL, enable WSL integration in Docker Desktop settings.
