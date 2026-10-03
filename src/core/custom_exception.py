class AppError(Exception):
    """Base class for all application errors."""


class ExtractionError(AppError):
    """Raised when text cannot be extracted from an uploaded file."""


class VectorStoreError(AppError):
    """Raised when a vector store operation fails."""


class MemoryStoreError(AppError):
    """Raised when a cache memory operation fails."""


class LLMError(AppError):
    """Returns error occurred during llm invocation."""
