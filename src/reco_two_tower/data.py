from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


@dataclass
class InteractionData:
    train_pairs: List[Tuple[int, int]]
    train_user_items: Dict[int, Set[int]]
    test_user_items: Dict[int, Set[int]]
    user_to_index: Dict[str, int]
    item_to_index: Dict[str, int]
    index_to_user: List[str]
    index_to_item: List[str]

    @property
    def num_users(self) -> int:
        return len(self.index_to_user)

    @property
    def num_items(self) -> int:
        return len(self.index_to_item)


def _read_ml_100k(path: Path) -> Iterable[Tuple[str, str, float]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            user_id, item_id, rating = parts[0], parts[1], float(parts[2])
            yield user_id, item_id, rating


def build_interaction_data(
    train_path: str | Path,
    test_path: str | Path,
    min_rating: float = 4.0,
) -> InteractionData:
    train_path = Path(train_path)
    test_path = Path(test_path)

    users: Dict[str, int] = {}
    items: Dict[str, int] = {}
    index_to_user: List[str] = []
    index_to_item: List[str] = []

    def user_idx(user_id: str) -> int:
        if user_id not in users:
            users[user_id] = len(index_to_user)
            index_to_user.append(user_id)
        return users[user_id]

    def item_idx(item_id: str) -> int:
        if item_id not in items:
            items[item_id] = len(index_to_item)
            index_to_item.append(item_id)
        return items[item_id]

    train_pairs: List[Tuple[int, int]] = []
    train_user_items: Dict[int, Set[int]] = {}

    for raw_user, raw_item, rating in _read_ml_100k(train_path):
        if rating < min_rating:
            continue
        u = user_idx(raw_user)
        i = item_idx(raw_item)
        train_pairs.append((u, i))
        if u not in train_user_items:
            train_user_items[u] = set()
        train_user_items[u].add(i)

    test_user_items: Dict[int, Set[int]] = {}
    for raw_user, raw_item, rating in _read_ml_100k(test_path):
        if rating < min_rating:
            continue
        if raw_user not in users or raw_item not in items:
            continue
        u = users[raw_user]
        i = items[raw_item]
        if u not in test_user_items:
            test_user_items[u] = set()
        test_user_items[u].add(i)

    return InteractionData(
        train_pairs=train_pairs,
        train_user_items=train_user_items,
        test_user_items=test_user_items,
        user_to_index=users,
        item_to_index=items,
        index_to_user=index_to_user,
        index_to_item=index_to_item,
    )
