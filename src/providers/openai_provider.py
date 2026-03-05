import base64
import json
from typing import Any, Optional, cast

from loguru import logger
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from src.core.config import get_settings
from src.providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    BaseVisionProvider,
)

_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI ChatCompletion-based LLM provider."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        params: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            params["tools"] = tools

        response = await self._client.chat.completions.create(**params)
        return response.choices[0].message.content or ""

    async def generate_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> dict:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            tools=cast(list[ChatCompletionToolParam], tools),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        message = response.choices[0].message
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        return {
            "content": message.content,
            "tool_calls": tool_calls,
        }

    def get_model_name(self) -> str:
        return self._model


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embeddings provider."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embedding_model
        self._dimensions = _EMBEDDING_DIMENSIONS.get(self._model, 1536)

    async def embed_text(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # OpenAI supports up to 2048 items per request; batch in chunks of 512
        all_embeddings: list[list[float]] = []
        batch_size = 512
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            all_embeddings.extend([d.embedding for d in response.data])
        return all_embeddings

    def get_dimensions(self) -> int:
        return self._dimensions

    def get_model_name(self) -> str:
        return self._model


class OpenAIVisionProvider(BaseVisionProvider):
    """OpenAI GPT-4o vision provider for image analysis."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model  # gpt-4o supports vision

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    async def analyze_image(
        self,
        image_path: str,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        base64_image = self._encode_image(image_path)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=4096,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    async def extract_structured_data(
        self,
        image_path: str,
        prompt: str,
        **kwargs: Any,
    ) -> dict:
        raw_response = await self.analyze_image(image_path, prompt)
        try:
            # Try to extract JSON from the response (may be wrapped in markdown)
            text = raw_response.strip()
            if text.startswith("```"):
                # Remove markdown code block
                lines = text.split("\n")
                text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse structured data from vision response: {raw_response[:200]}"
            )
            return {"raw_response": raw_response, "parse_error": True}
