"""Verdict speaks the wire format Jev introduced. Start the server, then any Jev/Laya client
works by changing the base URL:

    verdict serve --port 8000
"""

import httpx

r = httpx.post(
    "http://localhost:8000/v1/systemone",
    json={
        "state": {"subject": "Duplicate charge on invoice #4411", "body": "Billed twice. Refund or we cancel."},
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this?",
                "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs", "sales": "pricing"},
            },
            "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["not urgent", "soon", "critical"]},
            "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
        },
    },
    timeout=60,
)
print(r.json())
