"""Verdict: small, fast, calibrated decision models."""

from .core import Decider, Verdict
from .types import Check, Choice, Decision, Example, Option, Question, Score

__version__ = "0.1.0"
__all__ = ["Verdict", "Decider", "Question", "Option", "Decision", "Example", "Choice", "Score", "Check"]
