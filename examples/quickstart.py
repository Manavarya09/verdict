"""Zero-shot in three lines, then teach it your decision in seconds."""

from verdict import Verdict

v = Verdict()  # multilingual-e5-small, ~118M params, runs on CPU

ticket = "Hi, we were billed twice for March. Refund the duplicate today or we cancel."

team = v.choose(
    ticket,
    {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, errors",
        "sales": "pricing, new contracts",
        "other": "everything else",
    },
    prompt="Which team should handle this?",
)
print(team.label, f"{team.confidence.probability:.2f}", team.ranked)

urgency = v.score(ticket, scale=(1, 3), rubric={1: "not urgent", 2: "soon", 3: "blocking or deadline"})
print(urgency.value, f"expected {urgency.expected:.2f}")

churn = v.check(ticket, claim="the customer threatens to cancel or leave")
print(churn.verdict, f"{churn.probability:.2f}")
