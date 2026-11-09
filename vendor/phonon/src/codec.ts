// The codec's decoder as ops of a backend (graph.ts): latents to 24 kHz audio, streamed in chunks of
// 1, 2, 4 or 8 frames over one shared state. The upsample and transformer run
// at 200 Hz from the chunk's position in the state buffer. Each convolution
// reads its input stage from the row where that stage's left context begins:
// the context is the rows the previous chunk left, zero before a text's first
// chunk, so convolutions never need a position. A chunk past the end of
// speech returns at once.
import { H, layerRows, rowBuffers } from "./lm";
import { STATE, type Encoder, type Graph, type Stage } from "./graph";

/** Frames per decoded chunk, the sizes a synthesis goes through. */
export const CHUNKS = [1, 2, 4, 8];
const C_MAX = 8;
/** Audio samples per frame. */
export const FRAME = 1920;
const WINDOW = 251;
/** Positions the codec transformer keeps: a ring over its window and a chunk. */
const RING = 512;

export class Codec<B, O> {
  /** The last chunk's samples, up to C_MAX * FRAME floats. */
  readonly audio: B;
  private readonly g: Graph<B, O>;
  private readonly ops: Map<number, O[]>;
  /** Each stage: `ctx` rows of left context (a reading convolution's (kernel - 1) * dilation),
   *  then `rate` rows per 200 Hz position. */
  private readonly stages: Stage<B>[];

  constructor(g: Graph<B, O>, lats: B) {
    this.g = g;
    const T = 16 * C_MAX;
    const stage = (ctx: number, rate: number, ch: number): Stage<B> => ({
      ctx,
      rate,
      ch,
      buf: g.buffer((ctx + rate * T) * ch * 4),
    });
    const tr = stage(6, 1, 512);
    const s0 = stage(1, 1, 512);
    const s1 = stage(2, 6, 256);
    const s1h = stage(0, 6, 128);
    const s1r = stage(1, 6, 256);
    const s2 = stage(2, 30, 128);
    const s2h = stage(0, 30, 64);
    const s2r = stage(1, 30, 128);
    const s3 = stage(2, 120, 64);
    const s3h = stage(0, 120, 32);
    const s3r = stage(2, 120, 64);
    this.stages = [tr, s0, s1, s1r, s2, s2r, s3, s3r];
    this.audio = g.buffer(C_MAX * FRAME * 4);
    const B = rowBuffers(g, T);
    const rings = [0, 1].map(() => ({ kc: g.buffer(RING * H * 4), vc: g.buffer(RING * H * 4) }));
    const gate = "chunk";
    const opsFor = (C: number): O[] => {
      const T = 16 * C;
      const conv = (
        name: string,
        src: Stage<B>,
        dst: Stage<B> | B,
        o: { cin: number; cout: number; k: number; elu: boolean; res?: Stage<B> },
      ) => {
        if (src.ctx !== o.k - 1) throw new Error(`${name} reads ${src.ctx} rows of context`);
        const out = dst === this.audio ? { buf: dst as B, ctx: 0 } : (dst as Stage<B>);
        return g.gemm(src.buf, `${name}.w`, out.buf, {
          M: src.rate * T,
          K: o.k * o.cin,
          N: o.cout,
          bias: `${name}.b`,
          load: "conv",
          conv: { cin: o.cin, k: o.k, dil: 1, elu: o.elu },
          res: o.res ? "add" : "none",
          rbuf: o.res?.buf,
          ro: o.res ? o.res.ctx * o.cout : 0,
          yo: out.ctx * o.cout,
          gate,
        });
      };
      const convT = (
        name: string,
        src: Stage<B>,
        dst: Stage<B>,
        o: { cin: number; cout: number; s: number },
      ) =>
        g.gemm(src.buf, `${name}.w`, dst.buf, {
          M: src.rate * T,
          K: 2 * o.cin,
          N: o.cout,
          bias: `${name}.b`,
          load: "convT",
          conv: { cin: o.cin, elu: true, stride: o.s },
          yo: dst.ctx * o.cout,
          gate,
        });
      return [
        g.upsample(lats, B.x, T),
        ...[0, 1].flatMap((i) =>
          layerRows(g, `d${i}`, {
            B,
            rows: T,
            ...rings[i],
            cache: { kind: "codec", window: WINDOW, ring: RING },
            pos0: 0,
            out: i === 1 ? { buf: tr.buf, yo: tr.ctx * 512 } : undefined,
          }),
        ),
        ...conv("s0", tr, s0, { cin: 512, cout: 512, k: 7, elu: false }),
        ...convT("s1", s0, s1, { cin: 512, cout: 256, s: 6 }),
        ...conv("s1r1", s1, s1h, { cin: 256, cout: 128, k: 3, elu: true }),
        ...conv("s1r2", s1h, s1r, { cin: 128, cout: 256, k: 1, elu: true, res: s1 }),
        ...convT("s2", s1r, s2, { cin: 256, cout: 128, s: 5 }),
        ...conv("s2r1", s2, s2h, { cin: 128, cout: 64, k: 3, elu: true }),
        ...conv("s2r2", s2h, s2r, { cin: 64, cout: 128, k: 1, elu: true, res: s2 }),
        ...convT("s3", s2r, s3, { cin: 128, cout: 64, s: 4 }),
        ...conv("s3r1", s3, s3h, { cin: 64, cout: 32, k: 3, elu: true }),
        ...conv("s3r2", s3h, s3r, { cin: 32, cout: 64, k: 1, elu: true, res: s3 }),
        ...conv("s4", s3r, this.audio, { cin: 64, cout: 1, k: 3, elu: true }),
        // Each stage keeps its last rows as the next chunk's left context.
        g.shiftRows(this.stages, T),
      ];
    };
    this.ops = new Map(CHUNKS.map((c) => [c, opsFor(c)]));
  }

  /** Before a text's first chunk: no context yet. */
  reset(enc: Encoder<B, O>) {
    for (const st of this.stages) enc.clear(st.buf, 0, st.ctx * st.ch * 4);
  }

  /** Decodes frames [f0, f0 + size) into `audio`. The position goes to the queue ahead of the
   *  submit that holds this encoder, so a submit holds one chunk at most. */
  encodeChunk(enc: Encoder<B, O>, f0: number, size: number) {
    this.g.write(this.g.state, STATE.chunk * 4, new Uint32Array([16 * f0]));
    enc.run(this.ops.get(size)!);
  }
}
