/**
 * In-browser (or Node) Verdict with no server: the same bi-encoder maths as the Python
 * package, on top of @huggingface/transformers (optional peer dependency).
 *
 *   import { LocalVerdict } from "@verdict-ai/sdk/browser";
 *   const v = await LocalVerdict.load();                     // Xenova/multilingual-e5-small, int8
 *   const r = await v.choose("billed twice", ["billing", "sales"] as const);
 *   r.label  // "billing" | "sales"
 */

import type { Check, Choice, Options, Score } from "./index.js";

type Extractor = (texts: string[], opts: Record<string, unknown>) => Promise<{ data: Float32Array; dims: number[] }>;

const SCALE = 20;

function softmax(z: number[]): number[] {
  const m = Math.max(...z);
  const e = z.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}
const dot = (a: Float32Array, b: Float32Array) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const entropy = (p: number[]) => -p.reduce((s, x) => s + (x > 0 ? x * Math.log(x) : 0), 0) / Math.log(p.length);

export interface LocalOptions {
  model?: string;
  device?: "webgpu" | "wasm" | "cpu" | "auto";
  dtype?: "q8" | "fp32" | "fp16";
  queryPrefix?: string;
  passagePrefix?: string;
  abstainMargin?: number;
}

export class LocalVerdict {
  private cache = new Map<string, Float32Array>();
  private constructor(private extractor: Extractor, private opts: Required<LocalOptions>) {}

  static async load(opts: LocalOptions = {}): Promise<LocalVerdict> {
    const model = opts.model ?? "Xenova/multilingual-e5-small";
    const tf: any = await import("@huggingface/transformers");
    const device = opts.device ?? (typeof navigator !== "undefined" && (navigator as any).gpu ? "webgpu" : "wasm");
    const extractor = (await tf.pipeline("feature-extraction", model, { dtype: opts.dtype ?? "q8", device })) as Extractor;
    const isE5 = /e5/i.test(model);
    return new LocalVerdict(extractor, {
      model, device, dtype: opts.dtype ?? "q8",
      queryPrefix: opts.queryPrefix ?? (isE5 ? "query: " : ""),
      passagePrefix: opts.passagePrefix ?? (isE5 ? "passage: " : ""),
      abstainMargin: opts.abstainMargin ?? 0.1,
    });
  }

  private async embed(texts: string[], prefix: string): Promise<Float32Array[]> {
    const todo = texts.filter((t) => !this.cache.has(prefix + t));
    if (todo.length) {
      const res = await this.extractor(todo.map((t) => prefix + t), { pooling: "mean", normalize: true });
      const dim = res.dims[1];
      todo.forEach((t, i) => this.cache.set(prefix + t, res.data.slice(i * dim, (i + 1) * dim)));
    }
    return texts.map((t) => this.cache.get(prefix + t)!);
  }

  private async logits(input: string, optionTexts: string[]): Promise<number[]> {
    const [x] = await this.embed([input], this.opts.queryPrefix);
    const O = await this.embed(optionTexts, this.opts.passagePrefix);
    return O.map((o) => dot(o, x) * SCALE);
  }

  private confidence(p: number[]) {
    const s = [...p].sort((a, b) => b - a);
    const margin = s[0] - (s[1] ?? 0);
    return { probability: s[0], margin, entropy: entropy(p), abstain: margin < this.opts.abstainMargin, calibrated: false, conformal_set: null };
  }

  async choose<const L extends string>(input: string, options: Options<L>, opts: { prompt?: string } = {}): Promise<Choice<L>> {
    const t0 = performance.now();
    const list = Array.isArray(options)
      ? (options as readonly L[]).map((label) => ({ label, text: label.replace(/[_-]/g, " ") }))
      : (Object.entries(options) as [L, string][]).map(([label, text]) => ({ label, text }));
    const pre = opts.prompt ? opts.prompt + " " : "";
    const p = softmax(await this.logits(input, list.map((o) => pre + o.text)));
    const order = p.map((_, i) => i).sort((a, b) => p[b] - p[a]);
    const distribution = {} as Record<L, number>;
    list.forEach((o, i) => { distribution[o.label] = p[i]; });
    return { label: list[order[0]].label, confidence: this.confidence(p), distribution, ranked: order.map((i) => list[i].label), engine: `local:${this.opts.model}`, latency_ms: performance.now() - t0 };
  }

  async score(input: string, opts: { scale?: [number, number]; rubric?: Record<number, string>; prompt?: string } = {}): Promise<Score> {
    const t0 = performance.now();
    const [lo, hi] = opts.scale ?? [1, 5];
    const pre = opts.prompt ? opts.prompt + " " : "";
    const levels: number[] = []; for (let i = lo; i <= hi; i++) levels.push(i);
    const texts = levels.map((i) => pre + (opts.rubric?.[i] ?? `${i} out of ${hi}`));
    const p = softmax(await this.logits(input, texts));
    const top = p.indexOf(Math.max(...p));
    const distribution: Record<string, number> = {}; levels.forEach((v, i) => { distribution[String(v)] = p[i]; });
    return { value: levels[top], expected: levels.reduce((s, v, i) => s + v * p[i], 0), confidence: this.confidence(p), distribution, engine: `local:${this.opts.model}`, latency_ms: performance.now() - t0 };
  }

  async check(input: string, claim: string): Promise<Check> {
    const t0 = performance.now();
    const p = softmax(await this.logits(input, [`It is false that ${claim}`, `It is true that ${claim}`]));
    return { verdict: p[1] > p[0], probability: p[1], confidence: this.confidence(p), engine: `local:${this.opts.model}`, latency_ms: performance.now() - t0 };
  }
}
