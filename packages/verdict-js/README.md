# @verdict-ai/sdk

Typed client for a Verdict server (`verdict serve`). Options flow into the return type.

```ts
import { Verdict } from "@verdict-ai/sdk";

const v = new Verdict({ baseUrl: "http://localhost:8000" });
const r = await v.choose("Billed twice, refund or we cancel", ["billing", "technical", "sales"]);
r.label;                       // "billing" | "technical" | "sales"
r.confidence.abstain;          // boolean
await v.check("Can I talk to a person?", "the user asks for a human");   // { verdict, probability, ... }
await v.score("Great product, slow delivery", { scale: [1, 5] });
```

Node 20+. No runtime dependencies.
