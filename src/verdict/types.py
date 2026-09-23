"""Core data types.

A *question* is a typed query about an *input*. Verdict never generates text; it
returns a decision with a probability distribution and an honest confidence.

Three question kinds cover the decisions that happen constantly inside software:

* ``choose``  pick one option from a closed list
* ``score``   place the input on an ordered scale
* ``check``   is a statement about the input true?
"""

from __future__ import annotations

from typing import Generic, Literal, Sequence, TypeVar

from pydantic import BaseModel, Field, model_validator

T = TypeVar("T", bound=str)

QuestionKind = Literal["choose", "score", "check"]


class Option(BaseModel):
    """One candidate answer. ``description`` is what the model reads; ``label`` is
    what you get back. Keep labels short and stable, put nuance in the description."""

    label: str
    description: str | None = None

    @property
    def text(self) -> str:
        return self.description or self.label.replace("_", " ").replace("-", " ")


class Question(BaseModel):
    """A typed question. Build one with the helpers on :class:`verdict.Verdict`,
    or construct directly when you want to serialise decisions as data."""

    kind: QuestionKind
    prompt: str | None = Field(
        default=None,
        description="Optional natural-language framing, e.g. 'What does the customer want?'",
    )
    options: list[Option] = Field(default_factory=list)
    scale: tuple[int, int] | None = None
    rubric: dict[int, str] | None = Field(
        default=None, description="For score: what each point on the scale means."
    )
    claim: str | None = Field(default=None, description="For check: the statement to test.")

    @model_validator(mode="after")
    def _validate(self) -> Question:
        if self.kind == "choose" and len(self.options) < 2:
            raise ValueError("choose needs at least two options")
        if self.kind == "score":
            if self.scale is None:
                raise ValueError("score needs a scale, e.g. (1, 5)")
            lo, hi = self.scale
            if hi - lo < 1:
                raise ValueError("scale must span at least two points")
        if self.kind == "check" and not self.claim:
            raise ValueError("check needs a claim")
        return self

    @property
    def option_labels(self) -> list[str]:
        return [o.label for o in self.options]


class Confidence(BaseModel):
    """Honest uncertainty. ``probability`` is calibrated (after temperature scaling).
    ``abstain`` is set when the calibrated conformal set has more than one member,
    i.e. the model cannot commit at the requested coverage."""

    probability: float
    margin: float = Field(description="Top-1 minus top-2 probability.")
    entropy: float
    abstain: bool = False
    calibrated: bool = Field(
        default=False,
        description="True when probabilities went through a temperature fitted on labelled data "
        "and abstain comes from a conformal threshold with a coverage guarantee.",
    )
    conformal_set: list[str] | None = Field(
        default=None,
        description="Labels that survive the conformal threshold. Length 1 = confident.",
    )


class Choice(BaseModel, Generic[T]):
    label: T
    confidence: Confidence
    distribution: dict[str, float]
    ranked: list[str]
    engine: str
    latency_ms: float

    def __bool__(self) -> bool:  # `if verdict.choose(...)` reads as "did it commit?"
        return not self.confidence.abstain


class Score(BaseModel):
    value: int
    expected: float = Field(description="Probability-weighted mean across the scale.")
    confidence: Confidence
    distribution: dict[int, float]
    engine: str
    latency_ms: float


class Check(BaseModel):
    verdict: bool
    probability: float = Field(description="P(claim is true).")
    confidence: Confidence
    engine: str
    latency_ms: float

    def __bool__(self) -> bool:
        return self.verdict


Answer = Choice | Score | Check


class Decision(BaseModel):
    """One (input, question) pair. The unit of batching, logging and training."""

    input: str
    question: Question
    context: dict[str, str] | None = Field(
        default=None, description="Extra fields folded into the input, e.g. user tier, channel."
    )

    def rendered_input(self) -> str:
        if not self.context:
            return self.input
        ctx = "\n".join(f"{k}: {v}" for k, v in self.context.items())
        return f"{ctx}\n\n{self.input}"


class Example(BaseModel):
    """A labelled decision, used for calibration, evaluation and distillation."""

    decision: Decision
    answer: str | int | bool
    source: str | None = None
    weight: float = 1.0


def options_from(seq: Sequence[str] | Sequence[Option] | dict[str, str]) -> list[Option]:
    if isinstance(seq, dict):
        return [Option(label=k, description=v) for k, v in seq.items()]
    out: list[Option] = []
    for o in seq:
        out.append(o if isinstance(o, Option) else Option(label=str(o)))
    return out
