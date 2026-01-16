import os
import random
import uuid
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.schema import StructuralPayload, MatchResponse

# Constants
COLLECTION_NAME = "civic_structures"
VECTOR_SIZE = 1536

# Environment variables
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Initialize Qdrant Client
# We will initialize it globally, but we might want to handle it better in lifespan if strictly needed there.
# For simplicity and standard FastAPI usage, global is often okay, but we need to ensure connection works.
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def get_mock_embedding() -> List[float]:
    """Generates a random mock embedding vector."""
    return [random.random() for _ in range(VECTOR_SIZE)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Check if collection exists, if not create it
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

    yield
    # Shutdown logic if any

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/v1/ingest")
async def ingest_data(payload: StructuralPayload):
    try:
        point_id = payload.session_hash  # Using session_hash as ID? Or should we generate a UUID?
        # The prompt says: "session_hash をIDとして使用（UUID生成でも可）"
        # session_hash is a string. Qdrant IDs can be int or UUID.
        # If session_hash is not a valid UUID string, we might need to hash it to a UUID or use a random UUID and store session_hash in payload.
        # Let's assume session_hash is unique enough or just generate a UUID for the point ID and store session_hash in payload.
        # Actually the prompt says "session_hash をIDとして使用".
        # If session_hash is arbitrary string, Qdrant requires UUID or Int.
        # Let's check if session_hash is a UUID. If not, generate a UUID based on it.
        # For safety, let's generate a deterministic UUID from the session_hash to keep it idempotent,
        # or just use a random UUID if we want to treat every ingestion as new.
        # "session_hash" suggests it is a hash, likely a string.
        # Let's try to use it as ID directly if it fits UUID format, otherwise generate one.
        # To be safe and simple compliant with "UUID生成でも可", let's generate a new UUID for the record
        # BUT the prompt says "session_hash をIDとして使用".
        # Let's assume the user knows session_hash is a valid UUID or use uuid5.

        # Let's use uuid5 to generate a consistent UUID from the session_hash string.
        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, payload.session_hash))

        vector = get_mock_embedding()

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_uuid,
                    vector=vector,
                    payload=payload.dict()
                )
            ]
        )
        return {"status": "indexed", "id": point_uuid}
    except Exception as e:
        # In a real app we would log this properly
        print(f"Error ingesting data: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/v1/match", response_model=MatchResponse)
async def match_data(payload: StructuralPayload):
    try:
        query_vector = get_mock_embedding()

        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=5
        )

        matches = []
        for scored_point in search_result:
            matches.append({
                "id": str(scored_point.id),
                "score": scored_point.score,
                "content": scored_point.payload  # returning the whole payload as content
            })

        return MatchResponse(matches=matches)
    except Exception as e:
        print(f"Error matching data: {e}")
        # Return empty list or error
        return MatchResponse(matches=[])
