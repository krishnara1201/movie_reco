from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reco_two_tower.inference import ExplainedRecommendation, Recommendation, TwoTowerPredictor


class ExistingUserRequest(BaseModel):
    user_id: str = Field(..., description="MovieLens user id")
    top_k: int = Field(10, ge=1, le=100)


class NewUserRequest(BaseModel):
    liked_item_ids: list[str] = Field(..., min_length=1, description="Liked MovieLens item ids")
    top_k: int = Field(10, ge=1, le=100)
    exclude_seen: bool = Field(True, description="Exclude already liked items from results")


class RecommendationItem(BaseModel):
    item_id: str
    score: float
    title: str


class ContributionItem(BaseModel):
    liked_item_id: str
    liked_title: str
    similarity: float


class ExplainedRecommendationItem(RecommendationItem):
    contributions: list[ContributionItem]


class MovieItem(BaseModel):
    item_id: str
    title: str


class PredictionResponse(BaseModel):
    mode: str
    top_k: int
    unknown_item_ids: list[str] = []
    predictions: list[RecommendationItem]


class ExplainedPredictionResponse(BaseModel):
    mode: str
    top_k: int
    unknown_item_ids: list[str] = []
    predictions: list[ExplainedRecommendationItem]


class DebugPredictionResponse(BaseModel):
    mode: str
    top_k: int
    unknown_item_ids: list[str] = []
    liked_items: list[dict]
    user_profile: dict
    score_summary: dict
    top_candidates: list[dict]


app = FastAPI(title="Two-Tower Recommender API", version="1.0.0")

predictor = TwoTowerPredictor(
    checkpoint_path=ROOT / "artifacts" / "two_tower_best.pt",
    items_metadata_path=ROOT / "data" / "ml-100k" / "u.item",
)


def _serialize_predictions(predictions: list[Recommendation]) -> list[RecommendationItem]:
    return [
        RecommendationItem(item_id=p.item_id, score=p.score, title=p.title)
        for p in predictions
    ]


def _serialize_explained_predictions(
    predictions: list[ExplainedRecommendation],
) -> list[ExplainedRecommendationItem]:
    return [
        ExplainedRecommendationItem(
            item_id=p.item_id,
            score=p.score,
            title=p.title,
            contributions=[
                ContributionItem(
                    liked_item_id=c.liked_item_id,
                    liked_title=c.liked_title,
                    similarity=c.similarity,
                )
                for c in p.contributions
            ],
        )
        for p in predictions
    ]


@app.get("/")
def home() -> FileResponse:
    return FileResponse(ROOT / "api" / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/movies", response_model=list[MovieItem])
def movies(query: str | None = None, limit: int = 200) -> list[MovieItem]:
    items = predictor.list_movies(query=query, limit=limit)
    return [MovieItem(item_id=item_id, title=title) for item_id, title in items]


@app.post("/predict/existing", response_model=PredictionResponse)
def predict_existing(request: ExistingUserRequest) -> PredictionResponse:
    try:
        predictions = predictor.recommend_existing_user(
            user_id=request.user_id,
            top_k=request.top_k,
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return PredictionResponse(
        mode="existing-user",
        top_k=request.top_k,
        predictions=_serialize_predictions(predictions),
    )


@app.post("/predict/new", response_model=PredictionResponse)
def predict_new(request: NewUserRequest) -> PredictionResponse:
    try:
        predictions, unknown_item_ids = predictor.recommend_new_user(
            liked_item_ids=request.liked_item_ids,
            top_k=request.top_k,
            exclude_seen=request.exclude_seen,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return PredictionResponse(
        mode="new-user",
        top_k=request.top_k,
        unknown_item_ids=unknown_item_ids,
        predictions=_serialize_predictions(predictions),
    )


@app.post("/predict/new/explain", response_model=ExplainedPredictionResponse)
def predict_new_explain(request: NewUserRequest) -> ExplainedPredictionResponse:
    try:
        predictions, unknown_item_ids = predictor.recommend_new_user_explain(
            liked_item_ids=request.liked_item_ids,
            top_k=request.top_k,
            exclude_seen=request.exclude_seen,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return ExplainedPredictionResponse(
        mode="new-user-explain",
        top_k=request.top_k,
        unknown_item_ids=unknown_item_ids,
        predictions=_serialize_explained_predictions(predictions),
    )


@app.post("/predict/new/debug", response_model=DebugPredictionResponse)
def predict_new_debug(request: NewUserRequest) -> DebugPredictionResponse:
    try:
        payload = predictor.recommend_new_user_debug(
            liked_item_ids=request.liked_item_ids,
            top_k=request.top_k,
            exclude_seen=request.exclude_seen,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return DebugPredictionResponse(**payload)
