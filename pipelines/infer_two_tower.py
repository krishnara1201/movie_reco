from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reco_two_tower.model import TwoTowerModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-tower recommendations for existing or new users")
    parser.add_argument("--checkpoint-path", type=Path, default=ROOT / "artifacts" / "two_tower_best.pt")
    parser.add_argument("--items-metadata-path", type=Path, default=ROOT / "data" / "ml-100k" / "u.item")
    parser.add_argument("--top-k", type=int, default=10)

    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="Existing user id from MovieLens (e.g. 42).",
    )
    parser.add_argument(
        "--liked-item-ids",
        type=str,
        default=None,
        help="Comma-separated MovieLens item ids for a new user profile (e.g. 1,50,172).",
    )
    parser.add_argument(
        "--exclude-seen",
        action="store_true",
        help="Exclude input liked items from recommendation output for new user mode.",
    )
    return parser.parse_args()


def _load_item_titles(path: Path) -> dict[str, str]:
    titles: dict[str, str] = {}
    with path.open("r", encoding="latin-1") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) >= 2:
                titles[parts[0]] = parts[1]
    return titles


def _parse_liked_item_ids(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return []
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def _build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> TwoTowerModel:
    config = checkpoint["config"]
    num_users = len(checkpoint["index_to_user"])
    num_items = len(checkpoint["index_to_item"])

    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        output_dim=int(config["output_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _existing_user_vector(
    model: TwoTowerModel,
    user_to_index: dict[str, int],
    raw_user_id: str,
    device: torch.device,
) -> torch.Tensor:
    if raw_user_id not in user_to_index:
        raise ValueError(f"User id {raw_user_id} not present in training mapping.")
    user_idx = user_to_index[raw_user_id]
    user_ids = torch.tensor([user_idx], dtype=torch.long, device=device)
    with torch.no_grad():
        return model.encode_users(user_ids).squeeze(0)


def _new_user_vector(
    model: TwoTowerModel,
    item_to_index: dict[str, int],
    liked_item_ids: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, list[int], list[str]]:
    known_indices: list[int] = []
    unknown_items: list[str] = []

    for item_id in liked_item_ids:
        if item_id in item_to_index:
            known_indices.append(item_to_index[item_id])
        else:
            unknown_items.append(item_id)

    if not known_indices:
        raise ValueError("None of the provided liked item ids were found in training item mapping.")

    item_ids = torch.tensor(known_indices, dtype=torch.long, device=device)
    with torch.no_grad():
        liked_vectors = model.encode_items(item_ids)
        profile = liked_vectors.mean(dim=0)
        profile = F.normalize(profile, p=2, dim=0)
    return profile, known_indices, unknown_items


def main() -> None:
    args = parse_args()

    if args.user_id is None and args.liked_item_ids is None:
        raise ValueError("Provide either --user-id for existing user inference, or --liked-item-ids for new user inference.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)

    model = _build_model_from_checkpoint(checkpoint, device)
    user_to_index = checkpoint["user_to_index"]
    item_to_index = checkpoint["item_to_index"]
    index_to_item = checkpoint["index_to_item"]

    item_titles = _load_item_titles(args.items_metadata_path)

    if args.user_id is not None:
        user_vector = _existing_user_vector(model, user_to_index, args.user_id, device)
        seen_item_indices: set[int] = set()
        mode_label = f"existing-user:{args.user_id}"
    else:
        liked_ids = _parse_liked_item_ids(args.liked_item_ids)
        user_vector, liked_item_indices, unknown_items = _new_user_vector(model, item_to_index, liked_ids, device)
        seen_item_indices = set(liked_item_indices) if args.exclude_seen else set()
        mode_label = f"new-user-from-liked-items:{','.join(liked_ids)}"
        if unknown_items:
            print(f"Ignored unknown item ids: {unknown_items}")

    with torch.no_grad():
        item_ids = torch.arange(len(index_to_item), dtype=torch.long, device=device)
        item_vectors = model.encode_items(item_ids)
        scores = torch.matmul(item_vectors, user_vector)

        if seen_item_indices:
            seen_tensor = torch.tensor(list(seen_item_indices), dtype=torch.long, device=device)
            scores[seen_tensor] = -1e9

        topk = torch.topk(scores, k=min(args.top_k, len(index_to_item)))
        top_scores = topk.values.cpu().tolist()
        top_indices = topk.indices.cpu().tolist()

    print(f"Inference mode: {mode_label}")
    print(f"Device: {device}")
    print(f"Top-{len(top_indices)} recommendations")

    for rank, (idx, score) in enumerate(zip(top_indices, top_scores), start=1):
        raw_item_id = index_to_item[idx]
        title = item_titles.get(raw_item_id, "<unknown-title>")
        print(f"{rank:02d}. item_id={raw_item_id:>4} | score={score:.4f} | {title}")


if __name__ == "__main__":
    main()
