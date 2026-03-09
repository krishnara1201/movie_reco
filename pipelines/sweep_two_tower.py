from __future__ import annotations

import argparse
import json
import shutil
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reco_two_tower import TrainConfig, build_interaction_data, train_two_tower


def parse_csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for two-tower model")
    parser.add_argument("--train-path", type=Path, default=ROOT / "data" / "ml-100k" / "u1.base")
    parser.add_argument("--test-path", type=Path, default=ROOT / "data" / "ml-100k" / "u1.test")
    parser.add_argument("--min-rating", type=float, default=4.0)

    parser.add_argument("--embedding-dims", type=str, default="64,96,128")
    parser.add_argument("--learning-rates", type=str, default="0.001,0.0005")
    parser.add_argument("--num-negatives", type=str, default="4,6,10")

    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)

    parser.add_argument("--sweep-dir", type=Path, default=ROOT / "artifacts" / "sweep_bpr")
    parser.add_argument("--best-checkpoint", type=Path, default=ROOT / "artifacts" / "two_tower_best.pt")
    parser.add_argument("--summary-path", type=Path, default=ROOT / "artifacts" / "two_tower_sweep_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.sweep_dir.mkdir(parents=True, exist_ok=True)

    emb_values = parse_csv_ints(args.embedding_dims)
    lr_values = parse_csv_floats(args.learning_rates)
    neg_values = parse_csv_ints(args.num_negatives)

    all_results: list[dict] = []
    best_result: dict | None = None

    trial_no = 0
    total_trials = len(emb_values) * len(lr_values) * len(neg_values)

    for emb_dim, lr, num_neg in product(emb_values, lr_values, neg_values):
        trial_no += 1
        hidden_dim = emb_dim * 2
        tag = f"emb{emb_dim}_lr{lr}_neg{num_neg}"
        ckpt_path = args.sweep_dir / f"{tag}.pt"

        print(f"[{trial_no}/{total_trials}] training {tag}")
        data = build_interaction_data(
            train_path=args.train_path,
            test_path=args.test_path,
            min_rating=args.min_rating,
        )

        config = TrainConfig(
            embedding_dim=emb_dim,
            hidden_dim=hidden_dim,
            output_dim=emb_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=lr,
            weight_decay=1e-5,
            k=args.k,
            seed=args.seed,
            loss_type="bpr",
            temperature=0.07,
            patience=args.patience,
            num_negatives=num_neg,
        )

        result = train_two_tower(data, config, checkpoint_path=ckpt_path)
        result["tag"] = tag
        result["checkpoint"] = str(ckpt_path)

        metric_key = f"best_recall@{args.k}"
        score = float(result.get(metric_key, 0.0))
        result[metric_key] = score

        all_results.append(result)

        if best_result is None or score > float(best_result.get(metric_key, 0.0)):
            best_result = result
            args.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ckpt_path, args.best_checkpoint)
            print(f"  new best {metric_key}={score:.4f} ({tag})")

    all_results_sorted = sorted(
        all_results,
        key=lambda x: float(x.get(f"best_recall@{args.k}", 0.0)),
        reverse=True,
    )

    summary = {
        "search_space": {
            "embedding_dims": emb_values,
            "learning_rates": lr_values,
            "num_negatives": neg_values,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "min_rating": args.min_rating,
            "k": args.k,
        },
        "trials": all_results_sorted,
        "best": best_result,
        "best_checkpoint": str(args.best_checkpoint) if best_result else None,
    }

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Sweep complete. Summary: {args.summary_path}")
    if best_result is not None:
        print(f"Best checkpoint: {args.best_checkpoint}")


if __name__ == "__main__":
    main()
