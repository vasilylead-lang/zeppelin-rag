"""Load the Markdown knowledge base and split it into section-level chunks."""

from dataclasses import dataclass
from pathlib import Path

from zeppelin_rag.config import KNOWLEDGE_DIR


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    document: str
    section: str
    body: str

    @property
    def text(self) -> str:
        """Text used for embedding and as LLM context: headings carry meaning."""
        return f"{self.document}. {self.section}\n{self.body}"


def split_markdown(markdown: str, source: str) -> list[Chunk]:
    """One chunk per `## ` section; the `# ` title is attached to every chunk."""
    document = source
    chunks: list[Chunk] = []
    section: str | None = None
    lines: list[str] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if section and body:
            chunks.append(
                Chunk(
                    id=f"{source}#{len(chunks) + 1}",
                    source=source,
                    document=document,
                    section=section,
                    body=body,
                )
            )

    for line in markdown.splitlines():
        if line.startswith("# "):
            document = line[2:].strip()
        elif line.startswith("## "):
            flush()
            section, lines = line[3:].strip(), []
        elif section is not None:
            lines.append(line)
    flush()
    return chunks


def load_chunks(directory: Path = KNOWLEDGE_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(split_markdown(path.read_text(encoding="utf-8"), path.name))
    if not chunks:
        raise FileNotFoundError(f"No Markdown sections found in {directory}")
    return chunks
