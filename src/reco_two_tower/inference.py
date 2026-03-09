from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.nn import functional as F

from .model import TwoTowerModel


@dataclass
class Recommendation:
    item_id: str
    score: float
    title: str


@dataclass
class Contribution:
    liked_item_id: str
    liked_title: str
    similarity: float


@dataclass
class ExplainedRecommendation(Recommendation):
    contributions: list[Contribution]


class TwoTowerPredictor:
    def __init__(
        self,
        checkpoint_path: str | Path,
        items_metadata_path: str | Path,
        device: str | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.items_metadata_path = Path(items_metadata_path)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        config = checkpoint["config"]

        self.user_to_index: dict[str, int] = checkpoint["user_to_index"]
        self.item_to_index: dict[str, int] = checkpoint["item_to_index"]
        self.index_to_item: list[str] = checkpoint["index_to_item"]
        self.item_titles = self._load_item_titles(self.items_metadata_path)

        self.model = TwoTowerModel(
            num_users=len(checkpoint["index_to_user"]),
            num_items=len(self.index_to_item),
            embedding_dim=int(config["embedding_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            output_dim=int(config["output_dim"]),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            item_ids = torch.arange(len(self.index_to_item), dtype=torch.long, device=self.device)
            self.item_vectors = self.model.encode_items(item_ids)

    def list_movies(self, query: str | None = None, limit: int = 200) -> list[tuple[str, str]]:
        movies = [(item_id, self.item_titles.get(item_id, "<unknown-title>")) for item_id in self.index_to_item]
        if query:
            normalized = query.strip().lower()
            movies = [(item_id, title) for item_id, title in movies if normalized in title.lower()]
        return movies[: max(1, limit)]

    @staticmethod
    def _load_item_titles(path: Path) -> dict[str, str]:
        titles: dict[str, str] = {}
        with path.open("r", encoding="latin-1") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("|")
                if len(parts) >= 2:
                    titles[parts[0]] = parts[1]
        return titles

    def _topk(self, user_vector: torch.Tensor, top_k: int, excluded_indices: set[int] | None = None) -> list[Recommendation]:
        with torch.no_grad():
            scores = torch.matmul(self.item_vectors, user_vector)

            if excluded_indices:
                excluded = torch.tensor(list(excluded_indices), dtype=torch.long, device=self.device)
                scores[excluded] = -1e9

            topk = torch.topk(scores, k=min(top_k, len(self.index_to_item)))
            top_scores = topk.values.cpu().tolist()
            top_indices = topk.indices.cpu().tolist()

        recs: list[Recommendation] = []
        for idx, score in zip(top_indices, top_scores):
            raw_item_id = self.index_to_item[idx]
            recs.append(
                Recommendation(
                    item_id=raw_item_id,
                    score=float(score),
                    title=self.item_titles.get(raw_item_id, "<unknown-title>"),
                )
            )
        return recs

    def recommend_existing_user(self, user_id: str, top_k: int = 10) -> list[Recommendation]:
        if user_id not in self.user_to_index:
            raise ValueError(f"User id {user_id} not found in model mapping")

        user_idx = self.user_to_index[user_id]
        user_ids = torch.tensor([user_idx], dtype=torch.long, device=self.device)
        with torch.no_grad():
            user_vector = self.model.encode_users(user_ids).squeeze(0)

        return self._topk(user_vector, top_k=top_k)

    def recommend_new_user(
        self,
        liked_item_ids: Iterable[str],
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> tuple[list[Recommendation], list[str]]:
        known_indices: list[int] = []
        unknown_items: list[str] = []

        for item_id in liked_item_ids:
            raw_item = str(item_id)
            if raw_item in self.item_to_index:
                known_indices.append(self.item_to_index[raw_item])
            else:
                unknown_items.append(raw_item)

        if not known_indices:
            raise ValueError("No provided liked item ids were found in the model item mapping")

        item_ids = torch.tensor(known_indices, dtype=torch.long, device=self.device)
        with torch.no_grad():
            liked_vectors = self.model.encode_items(item_ids)
            user_vector = F.normalize(liked_vectors.mean(dim=0), p=2, dim=0)

        excluded = set(known_indices) if exclude_seen else set()
        return self._topk(user_vector, top_k=top_k, excluded_indices=excluded), unknown_items

    def recommend_new_user_explain(
        self,
        liked_item_ids: Iterable[str],
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> tuple[list[ExplainedRecommendation], list[str]]:
        known_indices: list[int] = []
        known_raw_ids: list[str] = []
        unknown_items: list[str] = []

        for item_id in liked_item_ids:
            raw_item = str(item_id)
            if raw_item in self.item_to_index:
                known_indices.append(self.item_to_index[raw_item])
                known_raw_ids.append(raw_item)
            else:
                unknown_items.append(raw_item)

        if not known_indices:
            raise ValueError("No provided liked item ids were found in the model item mapping")

        item_ids = torch.tensor(known_indices, dtype=torch.long, device=self.device)
        with torch.no_grad():
            liked_vectors = self.model.encode_items(item_ids)
            user_vector = F.normalize(liked_vectors.mean(dim=0), p=2, dim=0)

        excluded = set(known_indices) if exclude_seen else set()
        recs = self._topk(user_vector, top_k=top_k, excluded_indices=excluded)

        explained: list[ExplainedRecommendation] = []
        with torch.no_grad():
            for rec in recs:
                rec_index = self.item_to_index.get(rec.item_id)
                if rec_index is None:
                    rec_vec = None
                else:
                    rec_vec = self.item_vectors[rec_index]

                contributions: list[Contribution] = []
                if rec_vec is not None:
                    sims = torch.matmul(liked_vectors, rec_vec)
                    sims_list = sims.cpu().tolist()
                    for liked_id, sim in zip(known_raw_ids, sims_list):
                        contributions.append(
                            Contribution(
                                liked_item_id=liked_id,
                                liked_title=self.item_titles.get(liked_id, "<unknown-title>"),
                                similarity=float(sim),
                            )
                        )
                    contributions.sort(key=lambda c: c.similarity, reverse=True)

                explained.append(
                    ExplainedRecommendation(
                        item_id=rec.item_id,
                        score=rec.score,
                        title=rec.title,
                        contributions=contributions,
                    )
                )

        return explained, unknown_items

    def recommend_new_user_debug(
        self,
        liked_item_ids: Iterable[str],
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> dict:
        known_indices: list[int] = []
        known_raw_ids: list[str] = []
        unknown_items: list[str] = []

        for item_id in liked_item_ids:
            raw_item = str(item_id)
            if raw_item in self.item_to_index:
                known_indices.append(self.item_to_index[raw_item])
                known_raw_ids.append(raw_item)
            else:
                unknown_items.append(raw_item)

        if not known_indices:
            raise ValueError("No provided liked item ids were found in the model item mapping")

        item_ids = torch.tensor(known_indices, dtype=torch.long, device=self.device)
        with torch.no_grad():
            liked_vectors = self.model.encode_items(item_ids)
            user_vector = F.normalize(liked_vectors.mean(dim=0), p=2, dim=0)
            scores = torch.matmul(self.item_vectors, user_vector)

            if exclude_seen:
                excluded = torch.tensor(known_indices, dtype=torch.long, device=self.device)
                scores[excluded] = -1e9

            topk = torch.topk(scores, k=min(top_k, len(self.index_to_item)))
            top_scores = topk.values.cpu().tolist()
            top_indices = topk.indices.cpu().tolist()

        liked_norms = torch.norm(liked_vectors, dim=1).cpu().tolist()
        user_norm = float(torch.norm(user_vector).item())

        valid_scores = scores[scores > -1e8]
        score_min = float(valid_scores.min().item())
        score_max = float(valid_scores.max().item())
        score_mean = float(valid_scores.mean().item())

        top_candidates = []
        with torch.no_grad():
            for idx, score in zip(top_indices, top_scores):
                rec_id = self.index_to_item[idx]
                rec_title = self.item_titles.get(rec_id, "<unknown-title>")
                rec_vec = self.item_vectors[idx]
                sims = torch.matmul(liked_vectors, rec_vec).cpu().tolist()
                contributions = [
                    {
                        "liked_item_id": liked_id,
                        "liked_title": self.item_titles.get(liked_id, "<unknown-title>"),
                        "similarity": float(sim),
                    }
                    for liked_id, sim in zip(known_raw_ids, sims)
                ]
                contributions.sort(key=lambda x: x["similarity"], reverse=True)
                top_candidates.append(
                    {
                        "item_id": rec_id,
                        "title": rec_title,
                        "score": float(score),
                        "contributions": contributions,
                    }
                )

        liked_items = []
        weight = 1.0 / len(known_raw_ids)
        for liked_id, vec_norm in zip(known_raw_ids, liked_norms):
            liked_items.append(
                {
                    "item_id": liked_id,
                    "title": self.item_titles.get(liked_id, "<unknown-title>"),
                    "vector_norm": float(vec_norm),
                    "profile_weight": weight,
                }
            )

        return {
            "mode": "new-user-debug",
            "top_k": top_k,
            "unknown_item_ids": unknown_items,
            "liked_items": liked_items,
            "user_profile": {
                "method": "mean(liked_item_embeddings)->l2_normalize",
                "vector_norm": user_norm,
            },
            "score_summary": {
                "min": score_min,
                "max": score_max,
                "mean": score_mean,
            },
            "top_candidates": top_candidates,
        }
