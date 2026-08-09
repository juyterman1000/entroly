from __future__ import annotations

from dataclasses import dataclass, replace

from entroly.integrations.langchain import EntrolyCompressor, EntrolyDocumentCompressor


@dataclass(frozen=True)
class FakeMessage:
    content: str
    type: str
    id: str
    tool_calls: tuple[dict, ...] = ()

    def model_dump(self) -> dict:
        return {
            "content": self.content,
            "type": self.type,
            "id": self.id,
            "tool_calls": self.tool_calls,
        }

    def model_copy(self, *, update: dict):
        return replace(self, **update)


@dataclass(frozen=True)
class FakeDocument:
    page_content: str
    metadata: dict
    id: str

    def model_copy(self, *, update: dict):
        return replace(self, **update)


def test_message_compression_preserves_concrete_type_and_tool_metadata() -> None:
    messages = [
        FakeMessage("system " * 800, "system", "m1"),
        FakeMessage("result " * 800, "ai", "m2", ({"id": "call-7", "name": "lookup"},)),
        FakeMessage("What changed?", "human", "m3"),
    ]
    output = EntrolyCompressor(budget=120, preserve_last_n=1).invoke(messages)
    assert all(isinstance(item, FakeMessage) for item in output)
    assert [item.id for item in output] == ["m1", "m2", "m3"]
    assert output[1].tool_calls == messages[1].tool_calls
    assert output[-1].content == "What changed?"


def test_stream_surface_is_real_not_documentation_only() -> None:
    compressor = EntrolyCompressor(budget=100)
    assert list(compressor.stream("short")) == ["short"]


def test_document_compressor_preserves_metadata_and_identity() -> None:
    documents = [
        FakeDocument("invoice evidence " * 500, {"source": "ledger.csv", "page": 4}, "doc-1"),
        FakeDocument("campaign evidence " * 500, {"source": "brief.md"}, "doc-2"),
    ]
    output = EntrolyDocumentCompressor(budget=120).compress_documents(
        documents, "invoice evidence"
    )
    assert [item.id for item in output] == ["doc-1", "doc-2"]
    assert [item.metadata for item in output] == [item.metadata for item in documents]
    assert all(len(item.page_content) <= len(source.page_content) for item, source in zip(output, documents))
