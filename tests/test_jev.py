import pytest

from verdict.jev import JevFormatError, jev_question_to_verdict, render_state


def test_render_state_flattens_json_and_dicts():
    assert render_state('{"a": {"b": 1}, "tags": ["x", "y"]}') == "a.b: 1\ntags: x, y"
    assert render_state({"k": "v"}) == "k: v"
    assert render_state("plain text") == "plain text"
    assert render_state(None) == ""


def test_question_mapping_and_errors():
    assert jev_question_to_verdict({"type": "score", "criteria": ["lo", "hi"]}).scale == (0, 1)
    with pytest.raises(JevFormatError):
        jev_question_to_verdict({"type": "choice", "criteria": {"only": None}})
    with pytest.raises(JevFormatError):
        jev_question_to_verdict({"type": "bogus"})
