# Two-Tower Recommendation Training

Train a two-tower retrieval model on MovieLens 100k split files.

## Quick start

From the project root:

```bash
/home/shreyash/projects/recommendation_ml/.venv/bin/python pipelines/train_two_tower.py
```

## Custom run

```bash
/home/shreyash/projects/recommendation_ml/.venv/bin/python pipelines/train_two_tower.py \
  --train-path data/ml-100k/u2.base \
  --test-path data/ml-100k/u2.test \
  --epochs 30 \
  --batch-size 512 \
  --learning-rate 0.001 \
  --loss-type bpr \
  --num-negatives 6 \
  --patience 8 \
  --k 10
```

For MovieLens 100k, multi-negative BPR is often more stable than in-batch contrastive loss for Recall@10.

## Outputs

- Model checkpoint: `artifacts/two_tower.pt`
- Metrics summary: `artifacts/two_tower_metrics.json`
- Best sweep checkpoint: `artifacts/two_tower_best.pt`
- Sweep summary: `artifacts/two_tower_sweep_summary.json`

The metrics file includes epoch loss and `recall@k` on the selected test split.

## Inference

### Existing user

```bash
/home/shreyash/projects/recommendation_ml/.venv/bin/python pipelines/infer_two_tower.py \
  --user-id 42 \
  --top-k 10
```

### New user (cold-start via liked items)

```bash
/home/shreyash/projects/recommendation_ml/.venv/bin/python pipelines/infer_two_tower.py \
  --liked-item-ids 1,50,172,174 \
  --exclude-seen \
  --top-k 10
```

In new-user mode, the script builds a temporary user embedding by averaging embeddings of liked items.

## Hyperparameter sweep

```bash
/home/shreyash/projects/recommendation_ml/.venv/bin/python pipelines/sweep_two_tower.py
```

This evaluates multiple BPR settings and automatically copies the best checkpoint to `artifacts/two_tower_best.pt`.
