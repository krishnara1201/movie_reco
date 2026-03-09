from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reco_two_tower import TrainConfig, build_interaction_data, train_two_tower


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a two-tower recommendation model on MovieLens 100k")
    parser.add_argument("--train-path", type=Path, default=ROOT / "data" / "ml-100k" / "u1.base")
    parser.add_argument("--test-path", type=Path, default=ROOT / "data" / "ml-100k" / "u1.test")
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--loss-type", type=str, choices=["bpr", "inbatch"], default="inbatch")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-negatives", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", type=Path, default=ROOT / "artifacts" / "two_tower.pt")
    parser.add_argument("--metrics-path", type=Path, default=ROOT / "artifacts" / "two_tower_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = build_interaction_data(
        train_path=args.train_path,
        test_path=args.test_path,
        min_rating=args.min_rating,
    )

    config = TrainConfig(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
        temperature=args.temperature,
        patience=args.patience,
        num_negatives=args.num_negatives,
        k=args.k,
        seed=args.seed,
    )

    metrics = train_two_tower(data, config, checkpoint_path=args.checkpoint_path)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training complete")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
