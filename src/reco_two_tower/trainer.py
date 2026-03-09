from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal

import torch
from torch.nn import functional as F

from .data import InteractionData
from .model import TwoTowerModel


@dataclass
class TrainConfig:
    embedding_dim: int = 64
    hidden_dim: int = 128
    output_dim: int = 64
    batch_size: int = 512
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    k: int = 10
    seed: int = 42
    loss_type: Literal["bpr", "inbatch"] = "inbatch"
    temperature: float = 0.07
    patience: int = 0
    num_negatives: int = 4


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_negative_item(num_items: int, positives: set[int], rng: random.Random) -> int:
    candidate = rng.randrange(num_items)
    while candidate in positives:
        candidate = rng.randrange(num_items)
    return candidate


def _batch_triplets(
    data: InteractionData,
    batch_size: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    users: List[int] = []
    pos_items: List[int] = []
    neg_items: List[int] = []

    for _ in range(batch_size):
        u, i = data.train_pairs[rng.randrange(len(data.train_pairs))]
        j = _sample_negative_item(data.num_items, data.train_user_items[u], rng)
        users.append(u)
        pos_items.append(i)
        neg_items.append(j)

    return (
        torch.tensor(users, dtype=torch.long),
        torch.tensor(pos_items, dtype=torch.long),
        torch.tensor(neg_items, dtype=torch.long),
    )


def _iter_positive_batches(
    data: InteractionData,
    batch_size: int,
    rng: random.Random,
) -> List[List[tuple[int, int]]]:
    shuffled = list(data.train_pairs)
    rng.shuffle(shuffled)
    return [
        shuffled[start : start + batch_size]
        for start in range(0, len(shuffled), batch_size)
    ]


def recall_at_k(model: TwoTowerModel, data: InteractionData, k: int, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        item_ids = torch.arange(data.num_items, dtype=torch.long, device=device)
        item_vectors = model.encode_items(item_ids)

        recalls: List[float] = []
        for user_idx, positives in data.test_user_items.items():
            if not positives:
                continue
            user_id = torch.tensor([user_idx], dtype=torch.long, device=device)
            user_vec = model.encode_users(user_id)
            scores = torch.matmul(user_vec, item_vectors.T).squeeze(0)

            train_seen = data.train_user_items.get(user_idx, set())
            if train_seen:
                mask_indices = torch.tensor(list(train_seen), dtype=torch.long, device=device)
                scores[mask_indices] = -1e9

            topk = torch.topk(scores, k=min(k, data.num_items)).indices.tolist()
            hit_count = sum(1 for item in positives if item in topk)
            recalls.append(hit_count / len(positives))

    if not recalls:
        return 0.0
    return float(sum(recalls) / len(recalls))


def train_two_tower(
    data: InteractionData,
    config: TrainConfig,
    checkpoint_path: str | Path | None = None,
) -> Dict[str, object]:
    _set_seed(config.seed)
    rng = random.Random(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTowerModel(
        num_users=data.num_users,
        num_items=data.num_items,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: List[Dict[str, float]] = []
    best_recall = float("-inf")
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    no_improve = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        positive_batches = _iter_positive_batches(data, config.batch_size, rng)

        if config.loss_type == "bpr":
            for pos_batch in positive_batches:
                users = torch.tensor([u for u, _ in pos_batch], dtype=torch.long, device=device)
                pos_items = torch.tensor([i for _, i in pos_batch], dtype=torch.long, device=device)

                user_vec = model.encode_users(users)
                pos_vec = model.encode_items(pos_items)
                pos_scores = (user_vec * pos_vec).sum(dim=-1, keepdim=True)

                negative_batches: List[torch.Tensor] = []
                for _ in range(config.num_negatives):
                    neg_items = torch.tensor(
                        [
                            _sample_negative_item(data.num_items, data.train_user_items[u], rng)
                            for u, _ in pos_batch
                        ],
                        dtype=torch.long,
                        device=device,
                    )
                    neg_vec = model.encode_items(neg_items)
                    neg_scores = (user_vec * neg_vec).sum(dim=-1, keepdim=True)
                    negative_batches.append(neg_scores)

                neg_scores_all = torch.cat(negative_batches, dim=1)
                loss = -F.logsigmoid(pos_scores - neg_scores_all).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
        else:
            for pos_batch in positive_batches:
                users = torch.tensor([u for u, _ in pos_batch], dtype=torch.long, device=device)
                pos_items = torch.tensor([i for _, i in pos_batch], dtype=torch.long, device=device)

                user_vec = model.encode_users(users)
                item_vec = model.encode_items(pos_items)
                logits = torch.matmul(user_vec, item_vec.T) / config.temperature
                targets = torch.arange(logits.size(0), device=device)

                loss_ui = F.cross_entropy(logits, targets)
                loss_iu = F.cross_entropy(logits.T, targets)
                loss = 0.5 * (loss_ui + loss_iu)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        steps_per_epoch = max(1, len(positive_batches))
        avg_loss = running_loss / steps_per_epoch
        recall = recall_at_k(model, data, config.k, device)
        history.append({"epoch": float(epoch), "loss": avg_loss, f"recall@{config.k}": recall})
        print(f"Epoch {epoch:02d} | loss={avg_loss:.4f} | recall@{config.k}={recall:.4f}")

        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch
            no_improve = 0
            best_state = {name: param.detach().cpu().clone() for name, param in model.state_dict().items()}
        else:
            no_improve += 1

        if config.patience > 0 and no_improve >= config.patience:
            print(
                f"Early stopping at epoch {epoch:02d} (best epoch={best_epoch:02d}, recall@{config.k}={best_recall:.4f})"
            )
            break

    summary: Dict[str, object] = {
        "history": history,
        "config": asdict(config),
        "num_users": data.num_users,
        "num_items": data.num_items,
        "device": str(device),
        "best_epoch": best_epoch,
        f"best_recall@{config.k}": best_recall,
    }

    if best_state is not None:
        model.load_state_dict(best_state)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "user_to_index": data.user_to_index,
                "item_to_index": data.item_to_index,
                "index_to_user": data.index_to_user,
                "index_to_item": data.index_to_item,
                "best_epoch": best_epoch,
                f"best_recall@{config.k}": best_recall,
            },
            checkpoint_path,
        )
        summary["checkpoint"] = str(checkpoint_path)

    return summary
