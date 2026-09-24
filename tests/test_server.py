import pytest
from fastapi.testclient import TestClient

from verdict.server import build_app, jev_question_to_verdict


def test_jev_question_mapping():
    q = jev_question_to_verdict({"type": "choice", "instructions": "team?", "criteria": {"a": "x", "b": None}})
    assert q.kind == "choose" and q.option_labels == ["a", "b"]
    q = jev_question_to_verdict({"type": "score", "criteria": ["low", "mid", "high"]})
    assert q.kind == "score" and q.scale == (0, 2)
    q = jev_question_to_verdict({"type": "noul", "instructions": "asks for a human?"})
    assert q.kind == "check"
    from verdict.jev import JevFormatError

    with pytest.raises(JevFormatError):
        jev_question_to_verdict({"type": "choice", "criteria": {"only": None}})


@pytest.mark.slow
def test_systemone_and_decide():
    c = TestClient(build_app())
    r = c.post(
        "/v1/systemone",
        json={
            "state": {"body": "billed twice, refund please or we cancel"},
            "questions": {
                "dept": {"type": "choice", "instructions": "which team?", "criteria": {"billing": "refunds", "tech": "bugs"}},
                "urgency": {"type": "score", "criteria": ["low", "high"]},
                "churn": {"type": "noul", "instructions": "threatens to cancel"},
            },
        },
    )
    assert r.status_code == 200, r.text
    a = r.json()["answers"]
    assert a["dept"]["choice"] == "billing" and "abstain" in a["dept"]
    assert 0 <= a["urgency"]["score"] <= 1 and set(a["urgency"]["legend"]) == {"0", "1"}
    assert 0 <= a["churn"]["noul"] <= 1
    r = c.post(
        "/v1/decide",
        json={"decisions": [{"input": "the striker scored", "question": {"kind": "choose", "options": [{"label": "sports"}, {"label": "finance"}]}}]},
    )
    assert r.json()["answers"][0]["label"] == "sports"


@pytest.mark.slow
def test_systemone_batch():
    c = TestClient(build_app())
    r = c.post(
        "/v1/systemone/batch",
        json={
            "states": ["refund me now", "the app crashes on launch"],
            "questions": {"dept": {"type": "choice", "criteria": {"billing": "refunds", "tech": "bugs, crashes"}}},
        },
    )
    assert r.status_code == 200, r.text
    res = r.json()["results"]
    assert res[0]["answers"]["dept"]["choice"] == "billing" and res[1]["answers"]["dept"]["choice"] == "tech"
