"""``verdict`` command line."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich import print as rprint
from rich.table import Table

app = typer.Typer(help="Verdict: small, fast, calibrated decision models.", no_args_is_help=True)


def _load_examples(path: Path) -> list[tuple[str, str]]:
    """JSONL rows: {"input": ..., "answer": ...} or {"text": ..., "label": ...}."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows.append((r.get("input", r.get("text")), r.get("answer", r.get("label"))))
    return rows


@app.command()
def ask(
    input: str = typer.Argument(..., help="The text to decide about (or '-' for stdin)."),
    options: list[str] = typer.Option(None, "--option", "-o", help="Choice option; repeat."),
    claim: str | None = typer.Option(None, help="Check: the claim to test."),
    scale: str | None = typer.Option(None, help="Score: e.g. '1-5'."),
    prompt: str | None = typer.Option(None, help="Optional framing question."),
    model: str | None = typer.Option(None, help="Embedding model id."),
    reranker: str | None = typer.Option(None, help="NLI cross-encoder id for reranking."),
    decider: Path | None = typer.Option(None, help="A saved .verdict file (overrides the question)."),
    as_json: bool = typer.Option(False, "--json"),
):
    from .core import Verdict

    text = sys.stdin.read() if input == "-" else input
    v = Verdict(**({"model": model} if model else {}), reranker=reranker)
    if decider:
        ans = v.load(decider)(text)
    elif claim:
        ans = v.check(text, claim=claim)
    elif scale:
        lo, hi = (int(x) for x in scale.split("-"))
        ans = v.score(text, scale=(lo, hi), prompt=prompt)
    elif options and len(options) >= 2:
        ans = v.choose(text, options, prompt=prompt)
    else:
        raise typer.BadParameter("give --option (x2+), --claim, --scale, or --decider")
    if as_json:
        print(ans.model_dump_json(indent=2))
    else:
        rprint(ans.model_dump())


@app.command()
def fit(
    question: Path = typer.Argument(..., help="JSON file with the Question (kind/options/scale/claim)."),
    train: Path = typer.Argument(..., help="JSONL of {input, answer}."),
    out: Path = typer.Argument(..., help="Where to write the .verdict file."),
    calibrate: Path | None = typer.Option(None, help="Held-out JSONL for calibration."),
    coverage: float = typer.Option(0.9),
    head: str = typer.Option("auto", help="auto | prototype | linear"),
    model: str | None = typer.Option(None),
):
    from .core import Verdict
    from .types import Question

    v = Verdict(**({"model": model} if model else {}))
    d = v.compile(Question.model_validate_json(question.read_text()))
    d.fit(_load_examples(train), head=head)  # type: ignore[arg-type]
    rprint(f"[green]fit[/green] {d.meta['fit']}")
    if calibrate:
        d.calibrate(_load_examples(calibrate), coverage=coverage)
        rprint(f"[green]calibrated[/green] {d.meta['calibration']}")
    d.save(out)
    rprint(f"saved {out}")


@app.command()
def evaluate(
    decider: Path = typer.Argument(...),
    test: Path = typer.Argument(..., help="JSONL of {input, answer}."),
    model: str | None = typer.Option(None),
):
    from .core import Verdict

    v = Verdict(**({"model": model} if model else {}))
    ev = v.load(decider).evaluate(_load_examples(test))
    t = Table(title=str(decider))
    t.add_column("metric")
    t.add_column("value")
    for k, val in ev.items():
        t.add_row(k, f"{val:.4f}" if isinstance(val, float) else str(val))
    rprint(t)


@app.command()
def distill(
    traces: Path = typer.Argument(..., help="JSONL of your LLM's decisions: {input, answer}."),
    out: Path = typer.Argument(..., help="Where to write the .verdict file."),
    question: Path | None = typer.Option(None, help="Question JSON; inferred from the answers if omitted."),
    coverage: float = typer.Option(0.9),
    model: str | None = typer.Option(None),
):
    """Turn an LLM decision log into a Decider that agrees with it, in milliseconds."""
    from .core import Verdict
    from .distill import distill as _distill
    from .distill import label_histogram, read_traces, suggest_question
    from .types import Question

    rows = read_traces(traces)
    q = Question.model_validate_json(question.read_text()) if question else suggest_question(rows)
    rprint(f"[bold]{len(rows)}[/bold] traces, labels: {label_histogram(rows)}")
    v = Verdict(**({"model": model} if model else {}))
    d, report = _distill(v, q, rows, coverage=coverage)
    d.save(out)
    t = Table(title=f"distilled -> {out}")
    t.add_column("metric")
    t.add_column("value")
    for k, val in report.items():
        t.add_row(k, f"{val:.4f}" if isinstance(val, float) else str(val))
    rprint(t)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
    model: str | None = typer.Option(None),
    reranker: str | None = typer.Option(None),
    deciders: list[Path] = typer.Option(None, "--decider", help="Saved .verdict files to mount by name."),
):
    """HTTP server. Jev-compatible POST /v1/systemone plus native POST /v1/decide."""
    import uvicorn

    from .server import build_app

    uvicorn.run(build_app(model=model, reranker=reranker, deciders=deciders or []), host=host, port=port)


@app.command()
def bench(
    suite: str = typer.Argument("banking77", help="banking77 | clinc150 | massive | all"),
    n: int = typer.Option(1000, help="Test examples per suite (0 = all)."),
    shots: str = typer.Option("0,16", help="Comma list; 0 = zero-shot, -1 = full train."),
    model: str | None = typer.Option(None),
    reranker: str | None = typer.Option(None),
    out: Path | None = typer.Option(None, help="Write results JSON here."),
):
    from bench.run import run_bench  # type: ignore[import-not-found]

    res = run_bench(suite, n=n, shots=[int(s) for s in shots.split(",")], model=model, reranker=reranker)
    if out:
        out.write_text(json.dumps(res, indent=2))
    t = Table(title="verdict bench")
    for col in ["suite", "shots", "n", "accuracy", "ece", "automation_rate", "accuracy_when_committed", "ms_per_example"]:
        t.add_column(col)
    for r in res:
        t.add_row(*[f"{r.get(c):.3f}" if isinstance(r.get(c), float) else str(r.get(c)) for c in t.columns and ["suite", "shots", "n", "accuracy", "ece", "automation_rate", "accuracy_when_committed", "ms_per_example"]])
    rprint(t)


if __name__ == "__main__":
    app()
