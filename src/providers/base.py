from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseLLMProvider(ABC):
    """Abstract base class for LLM (chat/completion) providers."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """Generate a text response from the model.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: Optional tool definitions for function calling.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            The generated text response.
        """
        ...

    @abstractmethod
    async def generate_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> dict:
        """Generate a response that may include tool calls.

        Args:
            messages: List of message dicts.
            tools: Tool definitions for function calling.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.

        Returns:
            Dict with 'content' (str or None) and 'tool_calls' (list or None).
        """
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model identifier being used."""
        ...


class BaseEmbeddingProvider(ABC):
    """Abstract base class for text embedding providers."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings.

        Args:
            texts: List of input texts.

        Returns:
            List of embedding vectors.
        """
        ...

    @abstractmethod
    def get_dimensions(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the embedding model identifier."""
        ...


class BaseVisionProvider(ABC):
    """Abstract base class for vision/image analysis providers."""

    @abstractmethod
    async def analyze_image(
        self,
        image_path: str,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Analyze an image and return a text description.

        Args:
            image_path: Path to the image file.
            prompt: Instruction prompt for the analysis.

        Returns:
            Text response from the vision model.
        """
        ...

    @abstractmethod
    async def extract_structured_data(
        self,
        image_path: str,
        prompt: str,
        **kwargs: Any,
    ) -> dict:
        """Extract structured data from an image.

        Args:
            image_path: Path to the image file.
            prompt: Instruction prompt requesting structured JSON output.

        Returns:
            Parsed dict of the structured data extracted from the image.
        """
        ...
