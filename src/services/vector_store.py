from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.core.custom_exception import VectorStoreError
from src.schemas.chat import RetrievedChunk


class VectorStore:
    """Store and search document chunks in Qdrant."""

    def __init__(
        self,
        url: str,
        collection: str,
        embeddings: Embeddings,
        dimension: int,
    ) -> None:

        self._collection = collection

        try:
            client = QdrantClient(url=url)

            if not client.collection_exists(collection):
                client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=Distance.COSINE,
                    ),
                )

            self._vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection,
                embedding=embeddings,
            )
        except Exception as exc:
            raise VectorStoreError("Could not initialise Qdrant") from exc

    def add_documents(
        self,
        chunks: list[str],
        doc_id: str,
        filename: str,
    ) -> None:
        """Embed and store document chunks in Qdrant."""
        metadatas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ]

        try:
            self._vector_store.add_texts(
                texts=chunks,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError("Could not add documents") from exc

    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Search Qdrant for relevant document chunks."""
        try:
            results = self._vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
            )
        except Exception as exc:
            raise VectorStoreError("Vector search failed") from exc

        return [
            RetrievedChunk(
                doc_id=str(document.metadata["doc_id"]),
                filename=str(document.metadata["filename"]),
                chunk_index=int(document.metadata["chunk_index"]),
                text=document.page_content,
                score=float(score),
            )
            for document, score in results
        ]
