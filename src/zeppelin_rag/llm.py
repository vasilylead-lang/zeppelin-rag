"""Claude calls used by the graph nodes (official Anthropic SDK)."""

import os
from typing import Protocol

import anthropic
from pydantic import BaseModel, Field

from zeppelin_rag.config import Settings
from zeppelin_rag.knowledge import Chunk

ANSWER_SYSTEM = """Ты — справочник по дирижаблям и цеппелинам.
Отвечай на русском языке, опираясь только на фрагменты базы знаний из сообщения пользователя.
После каждого утверждения ставь номер фрагмента-источника в квадратных скобках, например [2].
Если во фрагментах нет ответа, прямо скажи, что в базе знаний нет этих данных, и не додумывай.
Числа приводи с единицами измерения. Отвечай по существу, без вступлений."""

GRADE_PROMPT = """Вопрос: {question}

Ниже пронумерованные фрагменты базы знаний. Верни номера тех фрагментов,
которые содержат сведения, полезные для ответа на вопрос (хотя бы частично).
Если полезных нет, верни пустой список.

{context}"""

REWRITE_PROMPT = """Поиск по базе знаний о дирижаблях не нашёл релевантных фрагментов.
Исходный вопрос: {question}
Предыдущий поисковый запрос: {query}

Переформулируй запрос для семантического поиска: раскрой сокращения, добавь
синонимы и ключевые термины (названия кораблей, физические величины).
Верни только новый запрос одной строкой."""


class RelevantChunks(BaseModel):
    relevant: list[int] = Field(description="Номера релевантных фрагментов")


class RefusalError(RuntimeError):
    pass


class RagLLM(Protocol):
    def grade(self, question: str, chunks: list[Chunk]) -> list[int]: ...
    def rewrite(self, question: str, query: str) -> str: ...
    def answer(self, question: str, chunks: list[Chunk]) -> str: ...


def client_options() -> dict:
    """Keys not scoped to a workspace must name one via the anthropic-workspace-id header."""
    workspace = os.getenv("ANTHROPIC_WORKSPACE_ID")
    return {"default_headers": {"anthropic-workspace-id": workspace}} if workspace else {}


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(f"[{i}] {chunk.text}" for i, chunk in enumerate(chunks, start=1))


def _text(message) -> str:
    return "".join(block.text for block in message.content if block.type == "text").strip()


class ClaudeRagLLM:
    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None):
        self.settings = settings
        self.client = client or anthropic.Anthropic(**client_options())

    def grade(self, question: str, chunks: list[Chunk]) -> list[int]:
        """Returns 0-based indices of chunks that help answer the question."""
        response = self.client.messages.parse(
            model=self.settings.fast_model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": GRADE_PROMPT.format(
                        question=question, context=format_context(chunks)
                    ),
                }
            ],
            output_format=RelevantChunks,
        )
        numbers = response.parsed_output.relevant if response.parsed_output else []
        return sorted({n - 1 for n in numbers if 1 <= n <= len(chunks)})

    def rewrite(self, question: str, query: str) -> str:
        response = self.client.messages.create(
            model=self.settings.fast_model,
            max_tokens=256,
            messages=[
                {"role": "user", "content": REWRITE_PROMPT.format(question=question, query=query)}
            ],
        )
        return _text(response) or question

    def answer(self, question: str, chunks: list[Chunk]) -> str:
        context = format_context(chunks) if chunks else "(фрагменты не найдены)"
        response = self.client.beta.messages.create(
            model=self.settings.answer_model,
            max_tokens=16000,
            output_config={"effort": self.settings.answer_effort},
            # On a safety decline the API reruns the request on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=ANSWER_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": f"Фрагменты базы знаний:\n\n{context}\n\nВопрос: {question}",
                }
            ],
        )
        if response.stop_reason == "refusal":
            raise RefusalError(f"Model declined the question: {question!r}")
        return _text(response)
