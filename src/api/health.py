from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("")
def health() -> dict[str, str]:
    return {
        "status": "Healthy",
        "service": "RAG Backend",
        "version": "0.1.0",
    }
