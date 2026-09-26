from zeppelin_rag.knowledge import load_chunks, split_markdown


def test_split_markdown_one_chunk_per_section():
    markdown = "# Заголовок\n\nвступление\n\n## Первый\nтекст 1\n\n## Второй\nтекст 2\n## Пустой\n"
    chunks = split_markdown(markdown, "a.md")

    assert [c.section for c in chunks] == ["Первый", "Второй"]
    assert chunks[0].document == "Заголовок"
    assert chunks[0].body == "текст 1"
    assert chunks[1].id == "a.md#2"
    assert chunks[0].text.startswith("Заголовок. Первый\n")


def test_real_knowledge_base_loads():
    chunks = load_chunks()

    assert len(chunks) >= 50
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(len(c.body) > 100 for c in chunks)
