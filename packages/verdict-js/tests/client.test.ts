import { test } from "node:test";
import assert from "node:assert/strict";
import { Verdict } from "../src/index.ts";

test("choose posts a decide request and returns the first answer", async () => {
  let seen: any = null;
  const fetchMock = (async (url: string, init: any) => {
    seen = { url, body: JSON.parse(init.body) };
    return new Response(JSON.stringify({ answers: [{ label: "billing", confidence: { probability: 0.9, margin: 0.5, entropy: 0.2, abstain: false, calibrated: false, conformal_set: null }, distribution: { billing: 0.9, sales: 0.1 }, ranked: ["billing", "sales"], engine: "embed", latency_ms: 1 }] }), { status: 200 });
  }) as unknown as typeof fetch;
  const v = new Verdict({ baseUrl: "http://x/", fetch: fetchMock });
  const r = await v.choose("billed twice", { billing: "refunds", sales: "pricing" }, { prompt: "which team?" });
  assert.equal(r.label, "billing");
  assert.equal(seen.url, "http://x/v1/decide");
  assert.equal(seen.body.decisions[0].question.kind, "choose");
  assert.deepEqual(seen.body.decisions[0].question.options, [{ label: "billing", description: "refunds" }, { label: "sales", description: "pricing" }]);
});

test("check and score build the right question", async () => {
  const bodies: any[] = [];
  const fetchMock = (async (_u: string, init: any) => { bodies.push(JSON.parse(init.body)); return new Response(JSON.stringify({ answers: [{}] }), { status: 200 }); }) as unknown as typeof fetch;
  const v = new Verdict({ fetch: fetchMock });
  await v.check("hi", "the user greets");
  await v.score("great", { scale: [1, 5] });
  assert.equal(bodies[0].decisions[0].question.claim, "the user greets");
  assert.deepEqual(bodies[1].decisions[0].question.scale, [1, 5]);
});

test("non-2xx throws", async () => {
  const fetchMock = (async () => new Response("nope", { status: 500 })) as unknown as typeof fetch;
  await assert.rejects(() => new Verdict({ fetch: fetchMock }).check("a", "b"), /500/);
});
