from typing import List, Optional
from pydantic import BaseModel

class StructuralPayload(BaseModel):
    session_hash: str
    agent_type: str
    pain_point: str
    system_loop: str
    values: Optional[List[str]] = None

class MatchResponse(BaseModel):
    matches: List[dict]
