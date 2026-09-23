/**
 * Verdict client. The options array's element type flows into the return type:
 *
 *   const r = await v.choose(text, ["billing", "technical", "sales"] as const);
 *   r.label  // "billing" | "technical" | "sales"
 */

export interface Confidence {
  probability: number;
  margin: number;
  entropy: number;
  abstain: boolean;
  calibrated: boolean;
  conformal_set: string[] | null;
}

export interface Choice<L extends string = string> {
  label: L;
  confidence: Confidence;
  distribution: Record<L, number>;
  ranked: L[];
  engine: string;
  latency_ms: number;
}

export interface Score {
  value: number;
  expected: number;
  confidence: Confidence;
  distribution: Record<string, number>;
  engine: string;
  latency_ms: number;
}

export interface Check {
  verdict: boolean;
  probability: number;
  confidence: Confidence;
  engine: string;
  latency_ms: number;
}

export type Options<L extends string> = readonly L[] | Readonly<Record<L, string>>;

type Question =
  | { kind: "choose"; prompt?: string | null; options: { label: string; description?: string | null }[] }
  | { kind: "score"; prompt?: string | null; scale: [number, number]; rubric?: Record<number, string> | null }
  | { kind: "check"; claim: string };

export interface VerdictClientOptions {
  baseUrl?: string;
  fetch?: typeof fetch;
  headers?: Record<string, string>;
}

export class Verdict {
  private baseUrl: string;
  private f: typeof fetch;
  private headers: Record<string, string>;

  constructor(opts: VerdictClientOptions = {}) {
    this.baseUrl = (opts.baseUrl ?? "http://localhost:8000").replace(/\/$/, "");
    this.f = opts.fetch ?? fetch;
    this.headers = { "content-type": "application/json", ...(opts.headers ?? {}) };
  }

  private async decide<T>(input: string, question: Question, context?: Record<string, string>): Promise<T> {
    const res = await this.f(`${this.baseUrl}/v1/decide`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify({ decisions: [{ input, question, context: context ?? null }] }),
    });
    if (!res.ok) throw new Error(`verdict: ${res.status} ${await res.text()}`);
    const data = (await res.json()) as { answers: T[] };
    return data.answers[0];
  }

  choose<const L extends string>(
    input: string,
    options: Options<L>,
    opts: { prompt?: string; context?: Record<string, string> } = {},
  ): Promise<Choice<L>> {
    const list = Array.isArray(options)
      ? (options as readonly L[]).map((label) => ({ label }))
      : (Object.entries(options) as [L, string][]).map(([label, description]) => ({ label, description }));
    return this.decide<Choice<L>>(input, { kind: "choose", prompt: opts.prompt ?? null, options: list }, opts.context);
  }

  score(
    input: string,
    opts: { scale?: [number, number]; rubric?: Record<number, string>; prompt?: string; context?: Record<string, string> } = {},
  ): Promise<Score> {
    return this.decide<Score>(
      input,
      { kind: "score", prompt: opts.prompt ?? null, scale: opts.scale ?? [1, 5], rubric: opts.rubric ?? null },
      opts.context,
    );
  }

  check(input: string, claim: string, opts: { context?: Record<string, string> } = {}): Promise<Check> {
    return this.decide<Check>(input, { kind: "check", claim }, opts.context);
  }
}

export default Verdict;
