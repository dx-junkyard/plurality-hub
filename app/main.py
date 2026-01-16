import os
import uuid
from typing import List
from contextlib import asynccontextmanager

import dotenv
from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.schema import StructuralPayload, MatchResponse
from app.ai_client import AsyncAIClient

# Load environment variables
dotenv.load_dotenv()

# Constants
COLLECTION_NAME = "civic_structures"
VECTOR_SIZE = 1536

# Environment variables
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Initialize Qdrant Client
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Initialize AI Client (will be set in lifespan or global if we want simple access,
# but let's initialize it globally to fail fast if env vars are missing, or inside lifespan)
# To follow the pattern of the qdrant client, we can init it here,
# but it relies on environment variables being loaded.
# Since we called load_dotenv() above, it should be fine.
# However, if OPENAI_API_KEY is missing, it will raise ValueError.
# This is desirable as we want to fail startup if config is wrong.
try:
    ai_client = AsyncAIClient()
except ValueError as e:
    # If we are running in an environment where we might not have the key yet (like CI without secrets),
    # we might want to delay this or just print a warning.
    # But for this task, the goal is to enforce it.
    print(f"Warning: {e}. Application might fail if OpenAI features are used.")
    ai_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Check if collection exists, if not create it
    try:
        if not client.collection_exists(collection_name=COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            print(f"Collection '{COLLECTION_NAME}' created.")
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists.")
    except Exception as e:
        print(f"Error connecting to Qdrant or checking collection: {e}")

    yield
    # Shutdown logic if any

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/v1/ingest")
async def ingest_data(payload: StructuralPayload):
    if not ai_client:
        raise HTTPException(status_code=500, detail="AI Client not initialized properly (missing API Key?)")

    try:
        point_id = payload.session_hash
        # Use uuid5 to generate a consistent UUID from the session_hash string
        point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, payload.session_hash))

        # Construct text to embed
        text_to_embed = f"Subject: {payload.agent_type}\nPain: {payload.pain_point}\nSystem Loop: {payload.system_loop}"

        # Generate embedding
        vector = await ai_client.get_embedding(text_to_embed)

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
        print(f"Error ingesting data: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/v1/match", response_model=MatchResponse)
async def match_data(payload: StructuralPayload):
    if not ai_client:
         raise HTTPException(status_code=500, detail="AI Client not initialized properly (missing API Key?)")

    try:
        # Construct text to embed (same logic as ingest for consistency)
        text_to_embed = f"Subject: {payload.agent_type}\nPain: {payload.pain_point}\nSystem Loop: {payload.system_loop}"

        query_vector = await ai_client.get_embedding(text_to_embed)

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
                "content": scored_point.payload
            })

        return MatchResponse(matches=matches)
    except Exception as e:
        print(f"Error matching data: {e}")
        return MatchResponse(matches=[])
