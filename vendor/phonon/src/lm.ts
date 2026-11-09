// Phonon's flow LM as ops of a backend (graph.ts): a transformer (6 layers or 9) over the voice
// prompt, the text and the audio latents, and the flow head that samples each next latent. A
// voice's 125 prompt rows go through the layers once (`voice`), and a text then copies their keys
// and values in. The prefill runs the text and BOS rows padded to a multiple of 64: one fixed op
// set per size, so a text only writes its ids. A step feeds the position in the state buffer and
// makes one frame.
import { STATE, type AlignRows, type Cache, type Encoder, type Graph } from "./graph";
import type { Align } from "./files";

export const H = 512;
const FF = 2048;
export const PROMPT = 125;
/** Positions the LM keeps: the voice prompt, the text and the frames. */
export const S_MAX = 1024;
/** Frames one synthesis may make (61 s). */
export const F_MAX = 768;
export const TEXT_MAX = 240;
/** Prefill sizes: the text and BOS rows, padded. */
const PREFILL = [64, 128, 192, 256];
/** Token ids past any real one: BOS, and padding (a zero row). */
const BOS = 0xfffffffe;
const PAD = 0xffffffff;
const LN_EPS = "1e-5";
const FLOW_LN_EPS = "1e-6";
const ROPE_THETA = "10000.0";
const EOS_THRESHOLD = "4.0";
const SPLITS = 8;

/** Per-row activations of a transformer. */
export interface Rows<B> {
  x: B;
  h: B;
  qkv: B;
  q: B;
  att: B;
  ff: B;
}
export function rowBuffers<B>(g: Graph<B, unknown>, rows: number): Rows<B> {
  const r = (n: number) => g.buffer(rows * n * 4);
  return { x: r(H), h: r(H), qkv: r(3 * H), q: r(H), att: r(H), ff: r(FF) };
}

/** One transformer layer over `rows` rows. Row m sits at position pos0 + m, past the cache's
 *  own position (kernels.ts Cache). `out` takes the layer's result in place of `B.x`. */
export function layerRows<B, O>(
  g: Graph<B, O>,
  pre: string,
  o: {
    B: Rows<B>;
    rows: number;
    kc: B;
    vc: B;
    cache: Cache;
    pos0: number;
    /** A voice's last layer needs only its keys and values. */
    kvOnly?: boolean;
    out?: { buf: B; yo: number };
    /** A text's prefill: rows past the text and BOS are padding. */
    text?: boolean;
  },
): O[] {
  const { B, rows, cache } = o;
  const gate = cache.kind === "codec" ? "chunk" : o.text ? "text" : "none";
  const at = { rows, pos0: o.pos0 };
  const kv = [
    g.layernorm(B.x, B.h, rows, `${pre}.ln1.w`, `${pre}.ln1.b`, LN_EPS, gate),
    ...g.gemm(B.h, `${pre}.qkv`, B.qkv, { M: rows, K: H, N: 3 * H, gate }),
    g.ropeKV(B.qkv, B.q, o.kc, o.vc, at, { theta: ROPE_THETA, cache, gate }),
  ];
  if (o.kvOnly) return kv;
  return [
    ...kv,
    g.attention(B.q, o.kc, o.vc, B.att, at, { cache, gate }),
    ...g.gemm(B.att, `${pre}.o`, B.x, { M: rows, K: H, N: H, res: "add", rbuf: B.x, gate }),
    g.layernorm(B.x, B.h, rows, `${pre}.ln2.w`, `${pre}.ln2.b`, LN_EPS, gate),
    ...g.gemm(B.h, `${pre}.fc1`, B.ff, { M: rows, K: H, N: FF, act: "gelu", gate }),
    ...g.gemm(B.ff, `${pre}.fc2`, o.out?.buf ?? B.x, {
      M: rows,
      K: FF,
      N: H,
      res: "add",
      rbuf: B.x,
      yo: o.out?.yo ?? 0,
      gate,
    }),
  ];
}

export class LanguageModel<B, O> {
  /** The latents made so far, one row of 32 per frame. */
  readonly lats: B;
  /** One step: a latent, then the next position. */
  readonly step: O[];
  /** The alignment head's scores over the text, one row of TEXT_MAX per frame. */
  readonly align: B;
  private readonly g: Graph<B, O>;
  private readonly rows: Rows<B>;
  private readonly kc: B[];
  private readonly vc: B[];
  private readonly ids: B;
  /** The BOS row's output on its way to row 0, where the step's tail reads. */
  private readonly bos: B;
  private readonly voiceIn: B;
  private readonly voiceOps: O[];
  private readonly prefills: Map<number, O[]>;
  private readonly tail: O[];
  private readonly cond: B;
  private readonly ada: B;
  private readonly fh: B;
  private readonly ft: B;
  private readonly z: B;
  private readonly lat: B;
  private readonly parts: B;
  private readonly depth: number;
  /** The head that follows the text as it speaks (files.ts), its scores kept for word times. */
  private readonly alignLayer: number;
  private readonly alignAt: AlignRows;

  constructor(g: Graph<B, O>, depth: number, align: Align) {
    this.g = g;
    this.depth = depth;
    this.alignLayer = align.layer;
    this.alignAt = { lo: PROMPT, width: TEXT_MAX, head: align.head };
    const rows = PREFILL.at(-1)!;
    this.rows = rowBuffers(g, rows);
    this.kc = Array.from({ length: depth }, () => g.buffer(S_MAX * H * g.kvBytes));
    this.vc = Array.from({ length: depth }, () => g.buffer(S_MAX * H * g.kvBytes));
    this.cond = g.buffer(H * 4);
    this.ada = g.buffer(10240 * 4);
    this.fh = g.buffer(H * 4);
    this.ft = g.buffer(H * 4);
    this.z = g.buffer(32 * 4);
    this.lat = g.buffer(32 * 4);
    this.lats = g.buffer(F_MAX * 32 * 4);
    this.ids = g.buffer(rows * 4);
    this.bos = g.buffer(H * 4);
    this.parts = g.buffer(SPLITS * 8 * 66 * 4);
    this.voiceIn = g.buffer(PROMPT * H * 4);
    this.align = g.buffer(F_MAX * TEXT_MAX * 4);
    this.tail = this.tailOps();
    this.step = [...Array.from({ length: depth }, (_, i) => this.stepLayer(i)).flat(), ...this.tail];
    this.voiceOps = [
      ...g.gemm(this.voiceIn, "prompt_proj", this.rows.x, { M: PROMPT, K: H, N: H, gate: "none" }),
      ...this.layers(PROMPT),
    ];
    this.prefills = new Map(
      PREFILL.map((n) => [
        n,
        [g.embedRows(this.ids, this.rows.x, n, { bos: BOS, pad: PAD }), ...this.layers(n, PROMPT)],
      ]),
    );
  }

  /** The layers over `rows` rows from position `pos0`, in the prefill's way, and for a text's
   *  prefill frame 0's alignment row. */
  private layers(rows: number, pos0 = 0): O[] {
    const { g } = this;
    return Array.from({ length: this.depth }, (_, i) => [
      ...layerRows(g, `l${i}`, {
        B: this.rows,
        rows,
        kc: this.kc[i],
        vc: this.vc[i],
        cache: { kind: "lm" },
        pos0,
        kvOnly: !pos0 && i === this.depth - 1,
        text: pos0 > 0,
      }),
      ...(pos0 && i === this.alignLayer ? [g.alignRow(this.rows.q, this.kc[i], this.align, this.alignAt)] : []),
    ]).flat();
  }

  /** A voice's prompt rows through every layer, its keys and values kept in a new buffer. */
  voice(latents: Float32Array<ArrayBuffer>): B {
    const { g } = this;
    const bytes = PROMPT * H * g.kvBytes;
    const kv = g.buffer(2 * this.depth * bytes);
    g.write(this.voiceIn, 0, latents);
    const enc = g.encoder();
    enc.run(this.voiceOps);
    for (let i = 0; i < this.depth; i++) {
      enc.copy(this.kc[i], 0, kv, 2 * i * bytes, bytes);
      enc.copy(this.vc[i], 0, kv, (2 * i + 1) * bytes, bytes);
    }
    enc.submit();
    return kv;
  }

  /** Sets up a text in a voice (its keys and values from `voice`) and encodes its prefill, which
   *  makes frame 0. The state's position starts on BOS, and frame 0's step moves it past. */
  start(
    enc: Encoder<B, O>,
    ids: number[],
    voice: B,
    seed: number,
    temperature: number,
    framesAfterEos: number,
  ) {
    const { g } = this;
    const n = PREFILL.find((p) => p > ids.length)!;
    const padded = new Uint32Array(n).fill(PAD);
    padded.set(ids);
    padded[ids.length] = BOS;
    g.write(this.ids, 0, padded);
    const state = new Uint32Array(8);
    state[STATE.pos] = PROMPT + ids.length;
    state[STATE.after] = framesAfterEos;
    state[STATE.text] = ids.length;
    state[STATE.seed] = seed >>> 0;
    new Float32Array(state.buffer)[STATE.scale] = Math.sqrt(temperature);
    g.write(g.state, 0, state);
    const bytes = PROMPT * H * g.kvBytes;
    for (let i = 0; i < this.depth; i++) {
      enc.copy(voice, 2 * i * bytes, this.kc[i], 0, bytes);
      enc.copy(voice, (2 * i + 1) * bytes, this.vc[i], 0, bytes);
    }
    // The rows past BOS are padding: later rows never reach earlier ones, and each step writes
    // its own key and value before it reads the cache, so their keys and values are never read.
    enc.run(this.prefills.get(n)!);
    enc.copy(this.rows.x, ids.length * H * 4, this.bos, 0, H * 4);
    enc.copy(this.bos, 0, this.rows.x, 0, H * 4);
    enc.run(this.tail);
  }

  /** A step's layer: one row at the state's position, attention split over the keys. */
  private stepLayer(i: number): O[] {
    const { g } = this;
    const B = this.rows;
    const pre = `l${i}`;
    return [
      g.gemv(B.x, `${pre}.qkv`, B.qkv, {
        K: H,
        N: 3 * H,
        ln: { w: `${pre}.ln1.w`, b: `${pre}.ln1.b`, eps: LN_EPS },
      }),
      g.attnStep(B.qkv, this.kc[i], this.vc[i], this.parts, {
        splits: SPLITS,
        theta: ROPE_THETA,
        align: i === this.alignLayer ? { at: this.alignAt, buf: this.align } : undefined,
      }),
      g.gemv(null, `${pre}.o`, B.x, {
        K: H,
        N: H,
        res: "add",
        rbuf: B.x,
        combine: { splits: SPLITS, parts: this.parts },
      }),
      g.gemv(B.x, `${pre}.fc1`, B.ff, {
        K: H,
        N: FF,
        act: "gelu",
        ln: { w: `${pre}.ln2.w`, b: `${pre}.ln2.b`, eps: LN_EPS },
      }),
      g.gemv(B.ff, `${pre}.fc2`, B.x, { K: FF, N: H, res: "add", rbuf: B.x }),
    ];
  }

  /** From the final hidden state (row 0 of x): the flow head samples the next latent, then its
   *  EOS logit, the next input row and the next position. */
  private tailOps(): O[] {
    const { g } = this;
    const x = this.rows.x;
    const blocks = Array.from({ length: 6 }, (_, i) => {
      const a = i * 1536;
      return [
        g.gemv(this.fh, `fh${i}.fc1.w`, this.ft, {
          K: H,
          N: H,
          bias: `fh${i}.fc1.b`,
          act: "silu",
          ln: {
            w: `fh${i}.ln.w`,
            b: `fh${i}.ln.b`,
            eps: FLOW_LN_EPS,
            mod: this.ada,
            ho: a,
            so: a + 512,
          },
        }),
        g.gemv(this.ft, `fh${i}.fc2.w`, this.fh, {
          K: H,
          N: H,
          bias: `fh${i}.fc2.b`,
          res: "gate",
          rbuf: this.fh,
          gbuf: this.ada,
          go: a + 1024,
        }),
      ];
    }).flat();
    const fa = 6 * 1536;
    return [
      g.gemv(x, "fh.cond.w", this.cond, {
        K: H,
        N: H,
        bias: "fh.cond.b",
        act: "silu",
        ln: { w: "norm.w", b: "norm.b", eps: LN_EPS },
      }),
      g.gemv(this.cond, "fh.ada.w", this.ada, { K: H, N: 10240, bias: "fh.ada.b" }),
      g.noise(this.z),
      g.gemv(this.z, "fh.in.w", this.fh, { K: 32, N: H, bias: "fh.in.b" }),
      ...blocks,
      g.gemv(this.fh, "fh.out.w", this.lat, {
        K: H,
        N: 32,
        bias: "fh.out.b",
        res: "add",
        rbuf: this.z,
        ln: { eps: FLOW_LN_EPS, mod: this.ada, ho: fa, so: fa + 512 },
      }),
      g.finishStep(x, this.lat, this.lats, { eps: LN_EPS, threshold: EOS_THRESHOLD }),
      g.gemv(this.lat, "latent_proj", x, { K: 32, N: H, res: "vec", vname: "cond" }),
    ];
  }
}
