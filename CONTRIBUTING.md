# Contributing

```bash
git clone https://github.com/Manavarya09/verdict && cd verdict
uv venv --python 3.12 && uv pip install -e ".[dev,serve,bench,train]"
uv run pytest -m "not slow"        # fast unit tests
uv run pytest                       # + downloads the 118M model and runs real inference
uv run ruff check src tests bench
```

What we care about, in order:

1. **Honest numbers.** Every accuracy claim comes from `bench/` with the no-leakage protocol
   (heads on train minus the calibration slice, calibration on that slice only, numbers on
   the untouched test split). If you add a suite, add it to `bench/suites.py` and post the
   JSON in `results/`.
2. **Calibration is a feature, not a metric.** Changes that raise accuracy but break ECE or
   coverage are regressions.
3. **CPU first.** The default path must run on a laptop CPU and in a browser tab.
4. **The wire format stays compatible** with `POST /v1/systemone`.

Good first issues: new bench suites (multilingual MASSIVE locales, tool routing), ONNX int8
export, long-input chunking, TypeScript SDK tests, a `verdict distill` command that trains
a Decider from LLM traces.
