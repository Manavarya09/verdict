"""HTTP server.

Two surfaces:

* ``POST /v1/systemone`` — the wire format TypeSafe's Jev introduced and Laya, impossibl
  and kev copied. Change a base URL and every existing integration talks to Verdict.
* ``POST /v1/decide`` — Verdict's native shape (list of decisions, full answers).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .core import Verdict
from .types import Decision, Option, Question


def render_state(state: Any) -> str:
    """Structured state -> readable ``key: value`` lines (nested keys joined with dots).
    Encoders read this far better than raw JSON braces and quotes. A string that parses as
    a JSON object is rendered the same way."""
    if state is None:
        return ""
    if isinstance(state, str):
        st = state.strip()
        if st.startswith("{") and st.endswith("}"):
            import json

            try:
                return render_state(json.loads(st))
            except Exception:
                return state
        return state
    if isinstance(state, dict):
        lines: list[str] = []

        def walk(prefix: str, v: Any) -> None:
            if isinstance(v, dict):
                for k, x in v.items():
                    walk(f"{prefix}.{k}" if prefix else str(k), x)
            elif isinstance(v, list) and v and all(not isinstance(x, (dict, list)) for x in v):
                lines.append(f"{prefix}: {', '.join(str(x) for x in v)}")
            elif isinstance(v, list):
                for i, x in enumerate(v):
                    walk(f"{prefix}[{i}]", x)
            else:
                lines.append(f"{prefix}: {v}")

        walk("", state)
        return "\n".join(lines)
    if isinstance(state, list):
        return "\n".join(render_state(x) for x in state)
    return str(state)


def _state_text(state: Any) -> str:
    return render_state(state)


def _instr(x: Any) -> str | None:
    if x is None:
        return None
    if isinstance(x, str):
        return x
    import json

    return json.dumps(x, ensure_ascii=False)


def jev_question_to_verdict(q: dict) -> Question:
    kind = q.get("type")
    instr = _instr(q.get("instructions"))
    crit = q.get("criteria")
    if kind == "choice":
        if isinstance(crit, list):
            crit = {str(c): None for c in crit}
        if not isinstance(crit, dict) or len(crit) < 2:
            raise HTTPException(422, "choice.criteria must map >=2 labels to descriptions")
        opts = [Option(label=str(k), description=_instr(v)) for k, v in crit.items()]
        return Question(kind="choose", prompt=instr, options=opts)
    if kind == "score":
        if not isinstance(crit, list) or len(crit) < 2:
            raise HTTPException(422, "score.criteria must be a list of >=2 level descriptions")
        return Question(
            kind="score",
            prompt=instr,
            scale=(0, len(crit) - 1),
            rubric={i: _instr(c) or f"level {i}" for i, c in enumerate(crit)},
        )
    if kind == "noul":
        claim = instr or "the statement holds"
        if isinstance(crit, dict) and crit.get("true"):
            claim = f"{claim} ({_instr(crit['true'])})"
        return Question(kind="check", claim=claim)
    raise HTTPException(422, f"unknown question type {kind!r}")


class SystemOneRequest(BaseModel):
    model: str | None = None
    state: Any = None
    questions: dict[str, dict]


class SystemOneBatchRequest(BaseModel):
    model: str | None = None
    states: list[Any]
    questions: dict[str, dict]


class DecideRequest(BaseModel):
    decisions: list[Decision]
    decider: str | None = None


def build_app(model: str | None = None, reranker: str | None = None, deciders: list[Path] | None = None) -> FastAPI:
    app = FastAPI(title="verdict", version="0.1.0")
    v = Verdict(**({"model": model} if model else {}), reranker=reranker)
    mounted = {p.stem: v.load(p) for p in (deciders or [])}

    @app.get("/health")
    def health():
        return {"status": "ok", "engine": v.engine.name, "deciders": list(mounted)}

    @app.post("/v1/systemone")
    def systemone(req: SystemOneRequest):
        text = _state_text(req.state)
        answers: dict[str, dict] = {}
        tokens = 0
        for qid, raw in req.questions.items():
            q = jev_question_to_verdict(raw)
            a = v.compile(q)(text)
            tokens += len(text.split())
            conf = a.confidence
            if q.kind == "choose":
                answers[qid] = {
                    "type": "choice",
                    "choice": a.label,  # type: ignore[union-attr]
                    "confidence": round(conf.probability, 4),
                    "probabilities": {k: round(p, 4) for k, p in a.distribution.items()},  # type: ignore[union-attr]
                    "abstain": conf.abstain,
                }
            elif q.kind == "score":
                legend = {str(i): raw["criteria"][i] for i in range(len(raw["criteria"]))}
                answers[qid] = {
                    "type": "score",
                    "score": round(a.expected, 4),  # type: ignore[union-attr]
                    "confidence": round(conf.probability, 4),
                    "probabilities": {str(k): round(p, 4) for k, p in a.distribution.items()},  # type: ignore[union-attr]
                    "legend": legend,
                    "abstain": conf.abstain,
                }
            else:
                answers[qid] = {"type": "noul", "noul": round(a.probability, 4), "abstain": conf.abstain}  # type: ignore[union-attr]
        return {
            "model": f"verdict/{v.engine.model_name.split('/')[-1]}",
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }

    @app.post("/v1/systemone/batch")
    def systemone_batch(req: SystemOneBatchRequest):
        """Many states, one question set: each question is one encoder pass over all states."""
        texts = [_state_text(s) for s in req.states]
        per_state: list[dict[str, dict]] = [{} for _ in texts]
        for qid, raw in req.questions.items():
            q = jev_question_to_verdict(raw)
            answers = v.compile(q).batch(texts)
            for i, a in enumerate(answers):
                conf = a.confidence
                if q.kind == "choose":
                    per_state[i][qid] = {"type": "choice", "choice": a.label, "confidence": round(conf.probability, 4), "probabilities": {k: round(p, 4) for k, p in a.distribution.items()}, "abstain": conf.abstain}  # type: ignore[union-attr]
                elif q.kind == "score":
                    per_state[i][qid] = {"type": "score", "score": round(a.expected, 4), "confidence": round(conf.probability, 4), "probabilities": {str(k): round(p, 4) for k, p in a.distribution.items()}, "abstain": conf.abstain}  # type: ignore[union-attr]
                else:
                    per_state[i][qid] = {"type": "noul", "noul": round(a.probability, 4), "abstain": conf.abstain}  # type: ignore[union-attr]
        return {"model": f"verdict/{v.engine.model_name.split('/')[-1]}", "results": [{"answers": a} for a in per_state], "usage": {"input_tokens": sum(len(t.split()) for t in texts) * len(req.questions), "output_tokens": 0}}

    @app.post("/v1/decide")
    def decide(req: DecideRequest):
        if req.decider:
            d = mounted.get(req.decider)
            if d is None:
                raise HTTPException(404, f"no decider named {req.decider!r}")
            return {"answers": [a.model_dump() for a in d.batch([x.input for x in req.decisions], [x.context for x in req.decisions])]}
        return {"answers": [a.model_dump() for a in v.decide(req.decisions)]}

    return app
