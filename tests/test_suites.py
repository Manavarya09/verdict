import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def suites(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from bench import suites

    return suites


@pytest.mark.parametrize("n_test", [0, 7])
def test_massive_it_registered_and_forwards_locale(suites, monkeypatch, n_test):
    result = object()
    calls = []

    def massive(lang, n_test):
        calls.append((lang, n_test))
        return result

    monkeypatch.setattr(suites, "massive", massive)
    assert suites.SUITES["massive_it"](n_test=n_test) is result
    assert calls == [("it", n_test)]


@pytest.mark.parametrize("n_test", [0, 2])
def test_massive_it_splits_and_labels(suites, monkeypatch, n_test):
    labels = ["music_query", "alarm_set"]
    dataset = {
        "train": {
            "text": [f"train {i}" for i in range(1005)],
            "label": [labels[i % 2] for i in range(1005)],
        },
        "test": {
            "text": ["metti un po' di musica", "imposta una sveglia", "riproduci musica"],
            "label": ["music_query", "alarm_set", "music_query"],
        },
    }

    def load_dataset(name, lang):
        assert (name, lang) == ("mteb/amazon_massive_intent", "it")
        return dataset

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    suite = suites.SUITES["massive_it"]
    q, train, calib, test = suite(n_test=n_test)

    assert q.kind == "choose"
    assert q.option_labels == ["alarm set", "music query"]
    expected_train = {
        (text, label.replace("_", " "))
        for text, label in zip(dataset["train"]["text"], dataset["train"]["label"])
    }
    assert len(train) == 5 and len(calib) == 1000
    assert set(train).isdisjoint(calib)
    assert set(train + calib) == expected_train
    expected_test = [
        (text, label.replace("_", " "))
        for text, label in zip(dataset["test"]["text"], dataset["test"]["label"])
    ]
    assert test == (expected_test[:n_test] if n_test else expected_test)
    assert {text for text, _ in train + calib}.isdisjoint(text for text, _ in test)
    assert suite(n_test=n_test) == (q, train, calib, test)
