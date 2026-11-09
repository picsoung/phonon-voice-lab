// Phonon's tokenizer (its tokenizer.5.0.0.json): the checkpoint's normalizer
// rules, then SentencePiece Unigram over Metaspace words, with byte fallback.
interface TokenizerJson {
  model: { vocab: [string, number][]; unk_id?: number };
  normalizer?: Normalizer | { type: "Sequence"; normalizers: Normalizer[] };
}
type Normalizer =
  | { type: "Replace"; pattern: { Regex?: string; String?: string }; content: string }
  | { type: "Strip"; strip_left: boolean; strip_right: boolean }
  | { type: "Prepend"; prepend: string };

const BYTE = /^<0x([0-9A-F]{2})>$/;

export class Tokenizer {
  readonly pieces: string[];
  private readonly score = new Map<string, [number, number]>();
  private readonly byte = new Map<number, number>();
  private readonly maxLen: number;
  private readonly unk: number;
  private readonly norm: ((s: string) => string)[];

  constructor(json: TokenizerJson) {
    const vocab = json.model.vocab;
    this.pieces = vocab.map((v) => v[0]);
    vocab.forEach(([p, s], i) => {
      const m = BYTE.exec(p);
      if (m) this.byte.set(Number.parseInt(m[1], 16), i);
      else this.score.set(p, [i, s]);
    });
    this.maxLen = Math.max(...this.pieces.map((p) => p.length));
    this.unk = json.model.unk_id ?? 0;
    const n = json.normalizer;
    const rules: Normalizer[] = !n ? [] : n.type === "Sequence" ? n.normalizers : [n];
    this.norm = rules.map((rule) => {
      switch (rule.type) {
        case "Replace": {
          // Rust's \x{HHHH} escapes are JavaScript's \u{HHHH}.
          const re =
            rule.pattern.Regex !== undefined
              ? new RegExp(rule.pattern.Regex.replaceAll(/\\x\{([0-9A-Fa-f]+)\}/g, "\\u{$1}"), "gu")
              : new RegExp(
                  (rule.pattern.String ?? "").replaceAll(/[.*+?^${}()|[\]\\]/g, "\\$&"),
                  "gu",
                );
          return (s: string) => s.replaceAll(re, rule.content);
        }
        case "Strip":
          return (s: string) =>
            rule.strip_left && rule.strip_right
              ? s.trim()
              : rule.strip_left
                ? s.trimStart()
                : s.trimEnd();
        case "Prepend":
          return (s: string) => (s ? rule.prepend + s : s);
      }
    });
  }

  normalize(text: string) {
    return this.norm.reduce((s, f) => f(s), text);
  }

  /** Ids for `text`, as the reference tokenizer gives them (no special tokens). */
  encode(text: string): number[] {
    // The normalizer prepends ▁. Metaspace turns spaces into ▁ and splits before each.
    let s = this.normalize(text).replaceAll(" ", "▁");
    if (!s.startsWith("▁")) s = "▁" + s;
    const ids: number[] = [];
    for (const word of s.split(/(?=▁)/u)) if (word) ids.push(...this.word(word));
    return ids;
  }

  /** Viterbi over one word: its best-scoring pieces, bytes for what no piece covers. */
  private word(w: string): number[] {
    // SentencePiece pieces are made of code points.
    const chars = Array.from(w);
    const n = chars.length;
    const best = new Float64Array(n + 1).fill(-Infinity);
    const from = new Int32Array(n + 1);
    const id = new Int32Array(n + 1);
    best[0] = 0;
    for (let i = 0; i < n; i++) {
      if (best[i] === -Infinity) continue;
      let s = "";
      for (let j = i; j < Math.min(n, i + this.maxLen); j++) {
        s += chars[j];
        const hit = this.score.get(s);
        if (hit && best[i] + hit[1] > best[j + 1]) {
          best[j + 1] = best[i] + hit[1];
          from[j + 1] = i;
          id[j + 1] = hit[0];
        }
      }
      if (best[i] - 100 > best[i + 1]) {
        best[i + 1] = best[i] - 100;
        from[i + 1] = i;
        id[i + 1] = -1;
      }
    }
    const out: number[][] = [];
    for (let k = n; k > 0; k = from[k]) {
      if (id[k] >= 0) out.push([id[k]]);
      else
        out.push(
          [...new TextEncoder().encode(chars[k - 1])].map((b) => this.byte.get(b) ?? this.unk),
        );
    }
    return out.reverse().flat();
  }
}
