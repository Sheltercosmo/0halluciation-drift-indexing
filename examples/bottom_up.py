"""Runnable offline protocol demonstration; replace choose with your LLM."""

from pathlib import Path

from zero_index import build_index, retrieve


def choose(state):
    history = state["history"]
    if not history:
        return {"action": "find", "need": "visitor access opening and closing times"}
    last = history[-1]
    if last["action"] == "find":
        matches = last["result"].get("matches", [])
        return ({"action": "read", "node_id": matches[0]["node_id"]} if matches
                else {"action": "finish", "node_ids": []})
    if last["action"] == "read":
        return {"action": "up", "node_id": last["node_id"]}
    return {"action": "finish", "node_ids": [last["result"]["node_id"]]}


if __name__ == "__main__":
    source = Path(__file__).with_name("structured.md").read_text(encoding="utf-8")
    index = build_index(source, source_name="structured.md")
    result = retrieve(index, "When can visitors enter and leave?", choose)
    for item in result["evidence"]:
        print(item["text"])
        print(item["citation"])
