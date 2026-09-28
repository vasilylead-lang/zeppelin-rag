from zeppelin_rag.llm import client_options


def test_client_options_adds_workspace_header(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test")

    assert client_options() == {"default_headers": {"anthropic-workspace-id": "wrkspc_test"}}


def test_client_options_empty_without_workspace(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)

    assert client_options() == {}
