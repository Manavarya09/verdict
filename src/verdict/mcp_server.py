"""MCP server: expose Verdict's three decisions as tools for any agent runtime.

    pip install "verdictml[mcp]"
    verdict mcp                      # stdio transport

Claude Desktop / Cursor / Claude Code config:
    {"mcpServers": {"verdict": {"command": "verdict", "args": ["mcp"]}}}
"""

from __future__ import annotations

from typing import Any


def build(model: str | None = None):
    try:
        from mcp.server.mcpserver import MCPServer as _Server  # mcp >= 2
    except ImportError:  # pragma: no cover
        from mcp.server.fastmcp import FastMCP as _Server  # mcp 1.x

    from .core import Verdict

    mcp = _Server("verdict", instructions="Typed decisions: choose one of N, score on a scale, check a claim. Fast, calibrated, abstains when unsure.")
    v = Verdict(**({"model": model} if model else {}))

    @mcp.tool()
    def choose(input: str, options: dict[str, str] | list[str], prompt: str | None = None) -> dict[str, Any]:
        """Pick one option for the input. options: {label: description} or [labels]. Returns label,
        calibrated-style probability, ranked labels, and abstain (true when unsure)."""
        a = v.choose(input, options, prompt=prompt)
        return {"label": a.label, "probability": a.confidence.probability, "ranked": a.ranked, "abstain": a.confidence.abstain, "distribution": a.distribution}

    @mcp.tool()
    def score(input: str, low: int = 1, high: int = 5, rubric: dict[int, str] | None = None, prompt: str | None = None) -> dict[str, Any]:
        """Place the input on an ordered scale low..high. rubric: {point: meaning}."""
        a = v.score(input, scale=(low, high), rubric=rubric, prompt=prompt)
        return {"value": a.value, "expected": a.expected, "probability": a.confidence.probability, "abstain": a.confidence.abstain, "distribution": a.distribution}

    @mcp.tool()
    def check(input: str, claim: str) -> dict[str, Any]:
        """Is the claim true about the input? Returns verdict and P(true)."""
        a = v.check(input, claim=claim)
        return {"verdict": a.verdict, "probability": a.probability, "abstain": a.confidence.abstain}

    return mcp


def main(model: str | None = None) -> None:
    build(model).run()
