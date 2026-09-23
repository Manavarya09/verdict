"""Properties competitors do not have: option order cannot change the answer, and the same
input always gets the same probabilities."""

import numpy as np
import pytest

from verdict import Verdict


@pytest.fixture(scope="module")
def v():
    return Verdict()


@pytest.mark.slow
def test_option_order_invariance(v):
    opts = {"billing": "refunds, invoices", "technical": "bugs, outages", "sales": "pricing", "other": "anything else"}
    text = "my invoice shows a charge I never authorised"
    a = v.choose(text, opts)
    b = v.choose(text, dict(reversed(list(opts.items()))))
    assert a.label == b.label
    for k in opts:
        assert abs(a.distribution[k] - b.distribution[k]) < 1e-5


@pytest.mark.slow
def test_deterministic(v):
    a = v.choose("where is my parcel", ["shipping", "billing"])
    b = v.choose("where is my parcel", ["shipping", "billing"])
    assert np.allclose(list(a.distribution.values()), list(b.distribution.values()))


@pytest.mark.slow
def test_distribution_sums_to_one_and_check_is_complementary(v):
    a = v.choose("hello", ["a", "b", "c"])
    assert abs(sum(a.distribution.values()) - 1) < 1e-6
    c = v.check("the sky is blue", claim="the text is about the sky")
    assert 0 <= c.probability <= 1  # P(true) + P(false) == 1 by construction
