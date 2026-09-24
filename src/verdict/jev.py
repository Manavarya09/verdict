"""Jev wire-format helpers: no server dependencies.

``jev_question_to_verdict`` converts a ``{"type": "choice"|"score"|"noul", ...}`` question into
a Verdict :class:`Question`; ``render_state`` turns a structured state into readable text."""

from __future__ import annotations

from typing import Any

from .types import Option, Question


class JevFormatError(ValueError):
    pass


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
            raise JevFormatError("choice.criteria must map >=2 labels to descriptions")
        opts = [Option(label=str(k), description=_instr(v)) for k, v in crit.items()]
        return Question(kind="choose", prompt=instr, options=opts)
    if kind == "score":
        if not isinstance(crit, list) or len(crit) < 2:
            raise JevFormatError("score.criteria must be a list of >=2 level descriptions")
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
    raise JevFormatError(f"unknown question type {kind!r}")


