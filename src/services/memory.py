from redis import Redis
from redis.exceptions import RedisError

from src.core.custom_exception import MemoryStoreError
from src.schemas.chat import ChatMessage


class ChatMemory:
    """Conversation memory with Redis."""

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int,
        max_messages: int,
    ) -> None:
        self._client = Redis.from_url(
            redis_url,
            decode_responses=True,
        )
        self._ttl = ttl_seconds
        self._max_messages = max_messages

    def _key(self, session_id: str) -> str:
        return f"chat:{session_id}"

    def append(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Saves the latest user or assistant message."""
        try:
            key = self._key(session_id)

            messages = [
                ChatMessage(
                    role="user",
                    content=user_message,
                ).model_dump_json(),
                ChatMessage(
                    role="assistant",
                    content=assistant_message,
                ).model_dump_json(),
            ]

            pipe = self._client.pipeline()
            pipe.rpush(key, *messages)
            pipe.ltrim(key, -self._max_messages, -1)
            pipe.expire(key, self._ttl)
            pipe.execute()

        except RedisError as exc:
            raise MemoryStoreError("Could not save chat history") from exc

    def get_history(self, session_id: str) -> list[ChatMessage]:
        """Get conversation history."""
        try:
            messages = self._client.lrange(self._key(session_id), 0, -1)

            return [ChatMessage.model_validate_json(message) for message in messages]

        except RedisError as exc:
            raise MemoryStoreError("Could not load chat history") from exc
