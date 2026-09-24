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
from .jev import JevFormatError, jev_question_to_verdict, render_state  # noqa: F401  (re-exported)
from .types import Decision


def _state_text(state: Any) -> str:
    return render_state(state)


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

    @app.exception_handler(JevFormatError)
    async def _bad_question(_req, exc):  # type: ignore[no-untyped-def]
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=422, content={"detail": str(exc)})

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
