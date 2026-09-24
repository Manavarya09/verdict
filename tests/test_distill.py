import json

from verdict.distill import label_histogram, read_traces, suggest_question


def test_read_and_suggest(tmp_path):
    p = tmp_path / "t.jsonl"
    rows = [{"input": f"msg {i}", "answer": ["refund", "shipping"][i % 2]} for i in range(10)]
    rows.append({"prompt": "x", "output": {"choice": "refund"}})
    p.write_text("\n".join(json.dumps(r) for r in rows))
    tr = read_traces(p)
    assert len(tr) == 11 and tr[-1] == ("x", "refund")
    q = suggest_question(tr)
    assert q.kind == "choose" and set(q.option_labels) == {"refund", "shipping"}
    assert label_histogram(tr)["refund"] == 6
    assert suggest_question([("a", True), ("b", False)]).kind == "check"
    assert suggest_question([("a", 1), ("b", 5)]).kind == "score"
