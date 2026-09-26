import numpy as np
import pytest

from zeppelin_rag.knowledge import Chunk


class KeywordEmbedder:
    """Deterministic bag-of-keywords embedder: no model download in tests."""

    VOCAB = ["водород", "гелий", "сопротивление", "удлинение", "гинденбург", "двигатель"]

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = [[text.lower().count(word) for word in self.VOCAB] + [0.01] for text in texts]
        return np.asarray(rows, dtype=np.float32)


class FakeLLM:
    """Scripted RagLLM: grade() answers come from a queue, one per call."""

    def __init__(self, grades: list[list[int]]):
        self.grades = list(grades)
        self.calls: list[str] = []

    def grade(self, question, chunks):
        self.calls.append("grade")
        return self.grades.pop(0)

    def rewrite(self, question, query):
        self.calls.append("rewrite")
        return f"{query} (уточнено)"

    def answer(self, question, chunks):
        self.calls.append("answer")
        return f"ответ по {len(chunks)} фрагментам"


def make_chunk(n: int, body: str) -> Chunk:
    return Chunk(
        id=f"doc.md#{n}", source="doc.md", document="Док", section=f"Раздел {n}", body=body
    )


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        make_chunk(1, "Водород легче, гелий не горит. Гелий и водород — несущие газы."),
        make_chunk(2, "Сопротивление зависит от удлинения. Удлинение и сопротивление связаны."),
        make_chunk(3, "Гинденбург имел четыре двигателя. Двигатель Daimler-Benz."),
    ]
