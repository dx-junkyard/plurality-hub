from fastapi import FastAPI
from app.schema import StructuralPayload, MatchResponse
import uuid

app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/v1/ingest")
async def ingest_data(payload: StructuralPayload):
    print(f"Received payload: {payload.dict()}")
    return {"status": "indexed", "id": "mock-uuid"}

@app.post("/api/v1/match", response_model=MatchResponse)
async def match_data(payload: StructuralPayload):
    # Mock logic for matching
    mock_matches = [
        {
            "id": "mock-match-1",
            "score": 0.95,
            "content": "Similar pain point detected."
        },
        {
            "id": "mock-match-2",
            "score": 0.88,
            "content": "System loop overlap found."
        }
    ]
    return MatchResponse(matches=mock_matches)
