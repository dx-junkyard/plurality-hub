import os
from typing import List
from openai import AsyncOpenAI

class AsyncAIClient:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    async def get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding vector for the given text using OpenAI API.
        Replaces newlines with spaces for best practices.
        """
        # Replace newlines with spaces as per OpenAI recommendation for embeddings
        clean_text = text.replace("\n", " ")

        response = await self.client.embeddings.create(
            input=clean_text,
            model=self.model
        )

        return response.data[0].embedding
