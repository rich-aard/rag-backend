from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings(
    model_name: str,
    expected_dimension: int,
) -> Embeddings:
    """Load and validate the embedding model."""

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )

    actual_dimension = len(embeddings.embed_query("dimension check"))

    if actual_dimension != expected_dimension:
        raise ValueError(
            f"Embedding model '{model_name}' outputs "
            f"{actual_dimension} dimensions, "
            f"but embedding dimension is set to {expected_dimension}."
        )

    return embeddings
