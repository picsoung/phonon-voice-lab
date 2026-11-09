// What a backend gives the model's graph: buffers, the ops the LM (lm.ts) and the codec
// (codec.ts) are made of, and an encoder that queues them. The WebGPU backend (gpu.ts) runs each
// op as a WGSL kernel (kernels.ts). The CPU backend (cpu.ts) runs it in WebAssembly (cpu/). Both
// run the same graph, built once when an engine starts, and the same engine loop (engine.ts).
//
// Activations are f32 rows of channels. The weights are one blob (checkpoint.ts): vectors and
// small matrices in f16, the large matrices in int8 with one f16 scale per Q8_GROUP values. Ops
// name tensors by their names in the blob.
//
// The state buffer (STATE) is shared by the LM, the codec and the engine. Work past the end of
// speech returns at once (Gate), so a submit that runs ahead of the engine costs little.

export type Act = "none" | "gelu" | "silu";
/** none | add (y = r + acc) | add_y (y += acc) | gate (y = r + g * acc) |
 *  gate_y (y += g * acc) | vec (y = acc + v, v from the blob) */
export type Residual = "none" | "add" | "add_y" | "gate" | "gate_y" | "vec";
/** The state buffer's u32 slots: the position the next LM step feeds, the frame it makes, the
 *  frames to keep once the end of speech is seen (0 before), the first 200 Hz position of the
 *  codec's chunk, the text's length in tokens, and the frames to keep after the end-of-speech
 *  frame, then the noise seed and its f32 scale (stored as u32 bits). */
export const STATE = { pos: 0, frame: 1, end: 2, chunk: 3, text: 4, after: 5, seed: 6, scale: 7 } as const;
/** Where the alignment head's scores go: one row per frame over the text tokens, which start
 *  at position `lo`, `width` floats per row, from head `head`. */
export interface AlignRows {
  lo: number;
  width: number;
  head: number;
}
/** Which work returns at once past the end: none, an LM step (its frame) or a codec chunk (its
 *  first frame). "text" marks a text's prefill, whose rows past the text and its BOS row are
 *  padding: a backend may skip them, as nothing reads what they make. */
export type Gate = "none" | "step" | "chunk" | "text";
/** The two transformers' key and value caches: the LM's at fixed positions, or the codec's, a
 *  ring at the chunk's position over a window, its work gated past the end. */
export type Cache = { kind: "lm" } | { kind: "codec"; window: number; ring: number };
/** int8 weights per f16 scale. */
export const Q8_GROUP = 32;

export interface ConvShape {
  cin: number;
  k?: number;
  dil?: number;
  /** ELU on the input as it is read. */
  elu?: boolean;
  /** Transposed convolutions: the stride, one GEMM per output phase. */
  stride?: number;
}

/** y = epilogue(W x) for one row (x at element `xo`), optionally normalizing x first. */
export interface Gemv<B> {
  K: number;
  N: number;
  xo?: number;
  act?: Act;
  bias?: string;
  res?: Residual;
  rbuf?: B;
  /** gate: y = r + g[go + n] * acc, g in `gbuf` */
  gbuf?: B;
  go?: number;
  /** vec: y = acc + the blob's tensor `vname` */
  vname?: string;
  /** LayerNorm over 512 values, with affine weights (w, b) and/or the adaLN modulation in `mod`
   *  (shift at `ho`, scale at `so`). */
  ln?: { w?: string; b?: string; eps: string; mod?: B; so?: number; ho?: number };
  /** The input is this many attention splits in `parts` (attnStep), merged. */
  combine?: { splits: number; parts: B };
}

/** Y = epilogue(X W^T) over M rows: a linear layer, a convolution or a transposed one. A
 *  convolution reads its input from row 0, where its (k - 1) * dil rows of left context begin.
 *  A transposed one reads one row of context and runs one GEMM per output phase. */
export interface Gemm<B> {
  M: number;
  K: number;
  N: number;
  gate: Gate;
  act?: Act;
  bias?: string;
  res?: Residual;
  rbuf?: B;
  /** Output and residual offsets, in elements. */
  yo?: number;
  ro?: number;
  load?: "linear" | "conv" | "convT";
  conv?: ConvShape;
}

/** Rows of a transformer op: row m sits at position pos0 + m, past the cache's own position
 *  (the codec's chunk). */
export interface Rows {
  rows: number;
  pos0: number;
}

/** A codec stage: `ctx` rows of left context, then `rate` rows per 200 Hz position. */
export interface Stage<B> {
  ctx: number;
  rate: number;
  ch: number;
  buf: B;
}

/** A buffer the engine reads results from once a submit is done. */
export interface Staging {
  /** Starts the read of everything submitted so far. */
  map(): Promise<void>;
  /** The bytes, after `map` settles. */
  range(): ArrayBuffer;
  unmap(): void;
}

/** Commands, run in order when submitted. */
export interface Encoder<B, O> {
  run(ops: O[]): void;
  copy(src: B, srcOffset: number, dst: B, dstOffset: number, bytes: number): void;
  clear(buf: B, offset: number, bytes: number): void;
  read(src: B, srcOffset: number, dst: Staging, dstOffset: number, bytes: number): void;
  submit(): void;
}

export interface Graph<B, O> {
  /** The weight blob. */
  readonly weights: B;
  readonly state: B;
  /** Submits the engine keeps in flight. */
  readonly depth: number;
  /** Bytes per value of the LM's key and value cache. */
  readonly kvBytes: number;
  buffer(bytes: number): B;
  /** Bytes into a buffer, before the next submit. */
  write(buf: B, offset: number, data: ArrayBufferView<ArrayBuffer>): void;
  encoder(): Encoder<B, O>;
  staging(bytes: number): Staging;
  /** Settles when every op built so far can run. */
  ready(): Promise<void>;

  /** An LM step's matrix-vector product with int8 weights: a step's work, gated. */
  gemv(x: B | null, w: string, y: B, o: Gemv<B>): O;
  gemm(x: B, w: string, y: B, o: Gemm<B>): O[];
  /** LayerNorm over 512 channels per row, with its affine weights. */
  layernorm(x: B, y: B, rows: number, w: string, b: string, eps: string, gate: Gate): O;
  /** RoPE on the q and k of each row's qkv, q to `q`, k and v to the cache at the row's position. */
  ropeKV(qkv: B, q: B, kc: B, vc: B, at: Rows, o: { theta: string; cache: Cache; gate?: Gate }): O;
  /** Causal (and for the codec windowed) attention of many rows over the cache. */
  attention(q: B, kc: B, vc: B, out: B, at: Rows, o: { cache: Cache; gate?: Gate }): O;
  /** An LM step's attention at the state's position, split over the keys: it rotates q and k,
   *  writes k and v to the cache, and leaves each split's unnormalized output, max and sum in
   *  `parts` ([split][head][66]) for a GEMV to merge (Gemv.combine). With `align`, the alignment
   *  head's scores over the text go to the row of the state's frame. */
  attnStep(qkv: B, kc: B, vc: B, parts: B, o: { splits: number; theta: string; align?: { at: AlignRows; buf: B } }): O;
  /** Frame 0's alignment row: the alignment head's scores of the BOS row (the last text row, at
   *  the state's position) over the text, from the prefill's q. */
  alignRow(q: B, kc: B, al: B, at: AlignRows): O;
  /** Token ids (u32, `bos` and `pad` past any real one) to embedding rows: a text's prefill
   *  (Gate "text"). */
  embedRows(ids: B, x: B, rows: number, o: { bos: number; pad: number }): O;
  /** The next latent's noise: Box-Muller over a counter hash of the seed and the frame. */
  noise(z: B): O;
  /** The end of an LM step: the EOS logit from LN(x row 0), the latent kept, the position and
   *  frame moved on, and the end set once the logit passes `threshold`. */
  finishStep(x: B, lat: B, lats: B, o: { eps: string; threshold: string }): O;
  /** The codec's input: latents denormalized, projected and upsampled ×16 for the chunk at the
   *  state's codec position. */
  upsample(lats: B, y: B, T: number): O;
  /** Every stage's last `ctx` rows of a chunk of T positions to its front. */
  shiftRows(stages: Stage<B>[], T: number): O;
}
