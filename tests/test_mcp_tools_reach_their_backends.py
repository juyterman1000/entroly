"""Two MCP tools failed on every call: each passed its backend an argument
the backend never accepted.

Called through the official MCP client (TypeScript SDK 1.32.2) against
`entroly serve` with the full tool profile -- what a direct `entroly serve`
and `claude mcp add entroly -- entroly` expose:

* shared_memory_list -> "Error executing tool shared_memory_list:
  SharedMemoryStore.list_entries() got an unexpected keyword argument 'tag'"
* learn_from_failures -> "Error executing tool learn_from_failures:
  FailureMiner.mine() got an unexpected keyword argument 'sources'"

The other 86 tools, called once each with arguments built from their
schemas, either worked or rejected the dummy arguments with a message.
"""

from __future__ import annotations

import asyncio
import json

import pytest


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTROLY_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("ENTROLY_MCP_PROFILE", "full")
    monkeypatch.setenv("ENTROLY_MCP_PASSIVE", "1")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    from entroly.server import create_mcp_server

    server, _ = create_mcp_server()
    return server


def _call(server, name: str, args: dict):
    result = asyncio.run(server.call_tool(name, args))
    content = result[0] if isinstance(result, tuple) else result
    return json.loads(content[0].text)


def test_shared_memory_list_filters_by_agent_and_tag(mcp) -> None:
    _call(mcp, "shared_memory_write", {"content": "auth uses JWT", "agent_id": "a", "tags": "auth,decision"})
    _call(mcp, "shared_memory_write", {"content": "deploys run on Fridays", "agent_id": "b", "tags": "ops"})

    everything = _call(mcp, "shared_memory_list", {})
    tagged = _call(mcp, "shared_memory_list", {"tag": "auth"})
    by_agent = _call(mcp, "shared_memory_list", {"agent_id": "b"})

    assert {e["content"] for e in everything} == {"auth uses JWT", "deploys run on Fridays"}
    assert [e["content"] for e in tagged] == ["auth uses JWT"]
    assert [e["content"] for e in by_agent] == ["deploys run on Fridays"]


@pytest.mark.parametrize("deep", [False, True])
def test_learn_from_failures_runs(mcp, deep: bool) -> None:
    report = _call(mcp, "learn_from_failures", {"deep": deep})

    assert report.get("patterns") == [] or "patterns_found" in report, report


def test_failure_mining_reads_only_the_sources_asked_for(tmp_path) -> None:
    from entroly.learn import FailureMiner

    prism = tmp_path / "prism"
    prism.mkdir()
    (prism / "feedback.jsonl").write_text(
        "\n".join(json.dumps({"outcome": "negative", "entity": "auth.py", "error": f"boom {i}"}) for i in range(3)),
        encoding="utf-8",
    )
    miner = FailureMiner(tmp_path)

    assert [p.entity for p in miner.mine(sources=["prism"])] == ["auth.py"]
    assert miner.mine(sources=["vault"]) == []
    with pytest.raises(ValueError, match="unknown"):
        miner.mine(sources=["prism", "nonsense"])
