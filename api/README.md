# Recommender API

## Run with Docker Compose

From project root:

```bash
docker compose up --build
```

API and UI are served on `http://localhost:8000`.

## Frontend UI

Open:

```bash
http://localhost:8000/
```

You can search movies, add liked movies, request recommendations, and view score/contribution charts.

## Run server

```bash
/home/shreyash/projects/recommendation_ml/.venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Health check

```bash
curl http://localhost:8000/health
```

## Predict for existing user

```bash
curl -X POST http://localhost:8000/predict/existing \
  -H "Content-Type: application/json" \
  -d '{"user_id":"42","top_k":5}'
```

## Predict for new user profile

```bash
curl -X POST http://localhost:8000/predict/new \
  -H "Content-Type: application/json" \
  -d '{"liked_item_ids":["1","50","172","174"],"top_k":5,"exclude_seen":true}'
```

## Predict with explanation data

```bash
curl -X POST http://localhost:8000/predict/new/explain \
  -H "Content-Type: application/json" \
  -d '{"liked_item_ids":["1","50","172","174"],"top_k":5,"exclude_seen":true}'
```

## List movies for picker

```bash
curl "http://localhost:8000/movies?query=star&limit=20"
```
