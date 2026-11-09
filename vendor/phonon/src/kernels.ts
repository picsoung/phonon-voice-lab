// WGSL for the WebGPU backend (gpu.ts): one kernel per op of graph.ts. The weights are one blob
// read as plain u32 words: vectors and small matrices are f16 pairs, the large matrices int8 with
// one f16 scale per Q8_GROUP weights (checkpoint.ts), so no shader needs 16-bit types. The LM's
// key and value cache is f16, the codec's f32. Each generator returns the shader for one
// configuration, so loops and branches on its constants fold away. Every kernel touches all of
// its bindings (`_ = &x;`) and declares them all, because gpu.ts builds each pipeline's layout
// from those declarations.
import { Q8_GROUP, STATE, type Act, type AlignRows, type Cache, type ConvShape, type Gate, type Residual } from "./graph";
export { STATE };
/** Each kernel's uniform fields, in order: its WGSL struct and the values gpu.ts writes. */
export const UNIFORMS = {
  layernorm: ["go", "bo"],
  rows: ["rows", "pos0"],
  embedRows: ["eo", "eso", "bo"],
  upsample: ["T", "wo", "wso", "mo", "so"],
  shiftRows: ["T"],
  gemm: ["M", "yo", "ro", "rows", "wo", "bo", "wso"],
  gemmReduce: ["rows", "yo", "ro", "stride", "bo"],
  gemv: ["xo", "yo", "ro", "go", "so", "ho", "wo", "bo", "vo", "lgo", "lbo", "wso"],
  finishStep: ["ewo", "ebo", "ngo", "nbo"],
} as const;
export type Fields = readonly string[];
/** The WGSL struct `P` of `fields`, u32 unless `types` says, padded to whole vec4s. */
function struct(fields: Fields, types: Record<string, string> = {}): string {
  const pad = Array.from({ length: (4 - (fields.length % 4)) % 4 }, (_, i) => `p${i}`);
  return `struct P { ${[...fields, ...pad].map((f) => `${f}: ${types[f] ?? "u32"}`).join(", ")} }`;
}
export interface GemmOptions {
  K: number;
  N: number;
  act?: Act;
  bias?: boolean;
  res?: Residual;
  load?: "linear" | "conv" | "convT";
  conv?: ConvShape | null;
  splits?: number;
  /** int8 weights, else f16. */
  q8: boolean;
  gate: Gate;
}
export interface GemvOptions {
  K: number;
  N: number;
  act?: Act;
  bias?: boolean;
  res?: Residual;
  ln?: { affine: boolean; modulate: boolean; eps: string } | null;
  combine?: number;
  /** 32-lane subgroup sums, else shared memory. */
  sg: boolean;
}

const { pos: POS, frame: FRAME, end: END, chunk: CHUNK, text: TEXT, after: AFTER, seed: SEED, scale: SCALE } = STATE;
const GATES: Record<Gate, string> = {
  none: "",
  step: `if (st[${END}] != 0u && st[${FRAME}] >= st[${END}]) { return; }`,
  chunk: `if (st[${END}] != 0u && st[${CHUNK}] / 16u >= st[${END}]) { return; }`,
  text: "",
};

/** The blob's f16 values: one by half index, or four from a half index that is a multiple of 4. */
const BLOB = `
fn hf(i: u32) -> f32 { let f = unpack2x16float(wv[i >> 3u][(i >> 1u) & 3u]); return select(f.x, f.y, (i & 1u) == 1u); }
fn h4(i: u32) -> vec4<f32> { let q = wv[i >> 3u]; return select(vec4<f32>(unpack2x16float(q.x), unpack2x16float(q.y)), vec4<f32>(unpack2x16float(q.z), unpack2x16float(q.w)), (i & 4u) != 0u); }`;
/** Element e of an int8 tensor at word `o` with scales at half `so`. */
const int8At = (o: string, so: string, e: string) =>
  `(f32(unpack4xI8(wv[(${o}) / 4u + (${e}) / 16u][((${e}) / 4u) % 4u])[(${e}) % 4u]) * hf(${so} + (${e}) / ${Q8_GROUP}u))`;

const ACT = `
fn gelu(v: f32) -> f32 {
  // erf via Abramowitz-Stegun 7.1.26 is too coarse; use a high-precision rational erf.
  let z = v * 0.70710678118654752;
  let a = abs(z);
  let t = 1.0 / (1.0 + 0.5 * a);
  let r = t * exp(-a * a - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  let erfz = select(r - 1.0, 1.0 - r, z >= 0.0);
  return 0.5 * v * (1.0 + erfz);
}
fn silu(v: f32) -> f32 { return v / (1.0 + exp(-v)); }`;

function actCall(act: Act, v: string): string {
  if (act === "gelu") return `gelu(${v})`;
  if (act === "silu") return `silu(${v})`;
  return v;
}

/** The epilogue's residual, for output index `o` and residual index `ri`. */
function residual(res: Residual, ri: string): string {
  return {
    none: "",
    add: `v = r[${ri}] + v;`,
    add_y: "v = y[o] + v;",
    gate: `v = r[${ri}] + gv[p.go + n] * v;`,
    gate_y: "v = y[o] + gv[p.go + n] * v;",
    vec: "v = v + hf(p.vo + n);",
  }[res];
}

/** Sum over each segment of W subgroup lanes (W a power of 2, from 2 to 32). */
function subgroupSum(W: number): string {
  if (W === 32) return "fn sumSeg(v: f32, li: u32) -> f32 { return subgroupAdd(v); }";
  let body = "var t = v;";
  for (let s = W / 2; s >= 1; s /= 2) body += ` t += subgroupShuffleXor(t, ${s}u);`;
  return `fn sumSeg(v: f32, li: u32) -> f32 { ${body} return t; }`;
}

/** LayerNorm over 512 channels per row (one workgroup each), with its affine weights. */
export function layernorm({ eps, gate }: { eps: string; gate: Gate }): string {
  return `
${struct(UNIFORMS.layernorm)}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<storage, read> st: array<u32>;
var<workgroup> red: array<f32, 128>;
${BLOB}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = &x; _ = &y; _ = &wv; _ = &st;
  ${GATES[gate]}
  let base = wid.x * 512u;
  var v: array<f32, 4>;
  var s = 0.0;
  for (var i = 0u; i < 4u; i++) { v[i] = x[base + li + i * 128u]; s += v[i]; }
  red[li] = s; workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (li < k) { red[li] += red[li + k]; } workgroupBarrier(); }
  let mean = red[0] / 512.0;
  workgroupBarrier();
  var q = 0.0;
  for (var i = 0u; i < 4u; i++) { let d = v[i] - mean; q += d * d; }
  red[li] = q; workgroupBarrier();
  for (var k = 64u; k > 0u; k >>= 1u) { if (li < k) { red[li] += red[li + k]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / 512.0 + ${eps});
  for (var i = 0u; i < 4u; i++) {
    let c = li + i * 128u;
    y[base + c] = (v[i] - mean) * inv * hf(p.go + c) + hf(p.bo + c);
  }
}`;
}

/** The parts of a transformer kernel its cache decides. */
function cacheParts(cache: Cache) {
  const codec = cache.kind === "codec";
  return {
    kv16: !codec,
    /** Row 0's position past `pos0`. */
    at: codec ? `st[${CHUNK}]` : "0u",
    gate: GATES[codec ? "chunk" : "none"],
    slot: (e: string) => (codec ? `((${e}) % ${cache.ring}u)` : `(${e})`),
    window: codec ? cache.window : 0,
  };
}

/** RoPE on q and k of `rows` rows, writing q to qb and k, v to the cache at their positions
 *  (row m at pos0 + m, past the cache's own position). */
export function ropeKV({ theta, cache }: { theta: string; cache: Cache }): string {
  const { kv16, at, gate, slot: slotOf } = cacheParts(cache);
  const slot = slotOf("pos");
  // Each thread rotates two adjacent dims, so an f16 cache gets whole words.
  const store = kv16
    ? `let kw = (${slot} * 512u + h * 64u) / 2u + t;
  kc[kw] = pack2x16float(vec2<f32>(rot(base + 512u, d0, c0, s0), rot(base + 512u, d1, c1, s1)));
  vc[kw] = pack2x16float(vec2<f32>(qkv[base + 1024u + d0], qkv[base + 1024u + d1]));`
    : `let kb = ${slot} * 512u + h * 64u;
  kc[kb + d0] = rot(base + 512u, d0, c0, s0); kc[kb + d1] = rot(base + 512u, d1, c1, s1);
  vc[kb + d0] = qkv[base + 1024u + d0]; vc[kb + d1] = qkv[base + 1024u + d1];`;
  return `
${struct(UNIFORMS.rows)}
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> qb: array<f32>;
@group(0) @binding(2) var<storage, read_write> kc: array<${kv16 ? "u32" : "f32"}>;
@group(0) @binding(3) var<storage, read_write> vc: array<${kv16 ? "u32" : "f32"}>;
@group(0) @binding(4) var<storage, read> st: array<u32>;
@group(0) @binding(5) var<uniform> p: P;
fn rot(base: u32, d: u32, c: f32, s: f32) -> f32 {
  if (d < 32u) { return qkv[base + d] * c - qkv[base + d + 32u] * s; }
  return qkv[base + d] * c + qkv[base + d - 32u] * s;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = &qkv; _ = &qb; _ = &kc; _ = &vc; _ = &st;
  ${gate}
  let t = gid.x % 32u;
  let h = (gid.x / 32u) % 8u;
  let m = gid.x / 256u;
  if (m >= p.rows) { return; }
  let pos = p.pos0 + ${at} + m;
  let base = m * 1536u + h * 64u;
  let d0 = 2u * t; let d1 = d0 + 1u;
  let a0 = f32(pos) * pow(${theta}, -f32(d0 % 32u) / 32.0);
  let a1 = f32(pos) * pow(${theta}, -f32(d1 % 32u) / 32.0);
  let c0 = cos(a0); let s0 = sin(a0); let c1 = cos(a1); let s1 = sin(a1);
  qb[m * 512u + h * 64u + d0] = rot(base, d0, c0, s0);
  qb[m * 512u + h * 64u + d1] = rot(base, d1, c1, s1);
  ${store}
}`;
}

/** Causal (optionally windowed) attention for many query rows, flash style: a workgroup takes
 *  16 query rows of one head and walks the keys in tiles of 16 through shared memory with an
 *  online softmax, so each key is read once per 16 rows. Thread (r, c): row r, key c for the
 *  scores and dims 4c..4c+3 for the output. */
export function attention({ cache }: { cache: Cache }): string {
  const { kv16, at, gate, slot, window } = cacheParts(cache);
  const kT = kv16 ? "vec2<u32>" : "vec4<f32>";
  const rd = (b: string, i: string) => (kv16 ? `kv(${b}[${i}])` : `${b}[${i}]`);
  return `
${struct(UNIFORMS.rows)}
@group(0) @binding(0) var<storage, read> qb: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kc: array<${kT}>;
@group(0) @binding(2) var<storage, read> vc: array<${kT}>;
@group(0) @binding(3) var<storage, read_write> o: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> st: array<u32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> qs: array<array<vec4<f32>, 16>, 16>;
var<workgroup> ks: array<array<vec4<f32>, 16>, 16>;
var<workgroup> vs: array<array<vec4<f32>, 16>, 16>;
var<workgroup> sc: array<array<f32, 17>, 16>;
fn kv(u: vec2<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(u.x), unpack2x16float(u.y)); }
fn lo(pos: u32) -> u32 { return ${window ? `select(0u, pos + 1u - ${window}u, pos + 1u > ${window}u)` : "0u"}; }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = &qb; _ = &kc; _ = &vc; _ = &o; _ = &st;
  ${gate}
  let h = wid.y; let m0 = wid.x * 16u;
  let r = li / 16u; let c = li % 16u;
  let m = m0 + r;
  let last = min(m0 + 15u, p.rows - 1u);
  let base = p.pos0 + ${at};
  let pos = base + m;
  let rowOk = m < p.rows;
  let mylo = lo(pos);
  qs[r][c] = select(vec4<f32>(0.0), qb[m * 128u + h * 16u + c] * 0.125, rowOk);
  var mx = -1e30; var l = 0.0; var acc = vec4<f32>(0.0);
  let k0 = lo(base + m0); let k1 = base + last + 1u;   // keys [k0, k1) cover the 16 rows
  for (var t = k0; t < k1; t += 16u) {
    workgroupBarrier();
    let kp = t + r;   // this thread loads key t + r, dims 4c..4c+3
    if (kp < k1) {
      let s = ${slot("kp")} * 128u + h * 16u + c;
      ks[r][c] = ${rd("kc", "s")}; vs[r][c] = ${rd("vc", "s")};
    } else { ks[r][c] = vec4<f32>(0.0); vs[r][c] = vec4<f32>(0.0); }
    workgroupBarrier();
    var d = 0.0;
    for (var e = 0u; e < 16u; e++) { d += dot(qs[r][e], ks[c][e]); }
    let kc2 = t + c;
    sc[r][c] = select(-1e30, d, rowOk && kc2 <= pos && kc2 >= mylo && kc2 < k1);
    workgroupBarrier();
    var tm = mx;
    for (var j = 0u; j < 16u; j++) { tm = max(tm, sc[r][j]); }
    let scale = exp(mx - tm);
    l *= scale; acc *= scale;
    for (var j = 0u; j < 16u; j++) {
      let s = sc[r][j];
      let pj = select(0.0, exp(s - tm), s > -1e29);
      l += pj; acc += pj * vs[j][c];
    }
    mx = tm;
  }
  if (rowOk) { o[m * 128u + h * 16u + c] = acc / l; }
}`;
}

/** Token ids padded with BOS and PAD (lm.ts) to embedding rows: int8 token rows, the f16 BOS
 *  row, zero rows for PAD. */
export function embedRows({ bos, pad }: { bos: number; pad: number }): string {
  return `
${struct(UNIFORMS.embedRows)}
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
${BLOB}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = &ids; _ = &wv; _ = &x;
  let c = gid.x % 512u; let i = gid.x / 512u;
  let id = ids[i];
  var v = 0.0;
  if (id == ${bos}u) { v = hf(p.bo + c); } else if (id != ${pad}u) { v = ${int8At("p.eo", "p.eso", "id * 512u + c")}; }
  x[i * 512u + c] = v;
}`;
}

/** Noise for the next latent: Box-Muller over a counter hash of the seed and frame. */
export function noise(): string {
  return `
@group(0) @binding(0) var<storage, read_write> z: array<f32>;
@group(0) @binding(1) var<storage, read> st: array<u32>;
fn pcg(v: u32) -> u32 { var s = v * 747796405u + 2891336453u; let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) i: u32) {
  _ = &z; _ = &st;
  ${GATES.step}
  let f = st[${FRAME}];
  let a = pcg(st[${SEED}] ^ pcg(f * 64u + i * 2u)); let b = pcg(a ^ 0x9e3779b9u);
  let u1 = (f32(a >> 8u) + 1.0) / 16777217.0; let u2 = f32(b >> 8u) / 16777216.0;
  z[i] = sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2) * bitcast<f32>(st[${SCALE}]);
}`;
}

/** Codec input: denormalize latents and run the folded latent_proj + depthwise upsample (int8),
 *  for the chunk starting at the state's codec position (200 Hz). */
export function upsample(): string {
  return `
${struct(UNIFORMS.upsample)}
@group(0) @binding(0) var<storage, read> lats: array<f32>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<storage, read> st: array<u32>;
${BLOB}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = &lats; _ = &wv; _ = &y; _ = &st;
  ${GATES.chunk}
  let c = gid.x % 512u; let tl = gid.x / 512u;
  if (tl >= p.T) { return; }
  let t = st[${CHUNK}] + tl; let ph = t % 16u; let q = t / 16u;
  var acc = 0.0;
  for (var j = 0u; j < 2u; j++) {
    if (q < j) { continue; }
    let src = q - j;
    for (var i = 0u; i < 32u; i++) {
      let v = lats[src * 32u + i] * hf(p.so + i) + hf(p.mo + i);
      acc += ${int8At("p.wo", "p.wso", "((c * 16u + ph) * 2u + j) * 32u + i")} * v;
    }
  }
  y[tl * 512u + c] = acc;
}`;
}

/** Moves every stage's left context to its front in one dispatch. Stages are independent;
 *  each source starts after its destination, so no invocation can overwrite another's input. */
export function shiftRows(stages: { ctx: number; rate: number; ch: number }[]): string {
  let offset = 0;
  const copies = stages.map(({ ctx, rate, ch }, i) => {
    const start = offset;
    offset += ctx * ch;
    return `if (gid.x < ${offset}u) {
    let j = gid.x - ${start}u;
    b${i}[j] = b${i}[p.T * ${rate * ch}u + j];
    return;
  }`;
  });
  return `
${struct(UNIFORMS.shiftRows)}
${stages.map((_, i) => `@group(0) @binding(${i}) var<storage, read_write> b${i}: array<f32>;`).join("\n")}
@group(0) @binding(${stages.length}) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${stages.map((_, i) => `_ = &b${i};`).join(" ")}
  ${copies.join("\n  ")}
}`;
}

/** GEMM: 64x64 tiles, K in steps of 32, optional split-K. With splits > 1 it writes raw
 *  partial sums, and gemmReduce finishes them. A convolution reads its input stage from row 0,
 *  where its (k - 1) * dil rows of left context begin; a transposed one has one row of context
 *  and runs one GEMM per output phase. Each thread loads one 16-byte word of weights per K step
 *  (8 f16, or 16 int8 for half the threads). */
export function gemm({
  K,
  N,
  act = "none",
  bias = false,
  res = "none",
  load = "linear",
  conv = null,
  splits = 1,
  q8,
  gate,
}: GemmOptions): string {
  const isT = load === "convT";
  const shape: Required<ConvShape> = { cin: 0, k: 1, dil: 1, elu: false, stride: 1, ...conv };
  const stride = isT ? shape.stride : 1;
  const cin = shape.cin;
  const elu4 = (v: string) => `select(exp(${v}) - vec4<f32>(1.0), ${v}, ${v} > vec4<f32>(0.0))`;
  const row =
    load === "linear"
      ? `m * ${K}u + k`
      : load === "conv"
        ? `(m + (k / ${cin}u) * ${shape.dil}u) * ${cin}u + k % ${cin}u`
        : `(m + 1u - k / ${cin}u) * ${cin}u + k % ${cin}u`;
  const aLoad = `fn loadA(m: u32, k: u32) -> vec4<f32> {
  if (m >= p.M) { return vec4<f32>(0.0); }
  let v = x[(${row}) / 4u];
  return ${shape.elu && load !== "linear" ? elu4("v") : "v"};
}`;
  const Ks = Math.ceil(K / splits / 32) * 32;
  const wrow = isT ? stride * K : K;
  // Tiles in shared memory, k-major: As[row][k/4] and Bs[n][k/4] as vec4 along k (16 KB, the
  // portable limit). Each thread stores the whole vec4s it loaded.
  const bLoad = q8
    ? `
    if (li < 128u) {
      let row = li / 2u; let kq = (li % 2u) * 16u;
      let n = tn + row;
      var q = vec4<u32>(0u); var sc = 0.0;
      if (n < ${N}u) {
        let col = wbase + kt + kq;
        q = wv[p.wo / 4u + (n * ${wrow}u + col) / 16u];
        sc = hf(p.wso + n * ${wrow / Q8_GROUP}u + col / ${Q8_GROUP}u);
      }
      let k4 = kq / 4u;
      Bs[row][k4] = vec4<f32>(unpack4xI8(q.x)) * sc; Bs[row][k4 + 1u] = vec4<f32>(unpack4xI8(q.y)) * sc;
      Bs[row][k4 + 2u] = vec4<f32>(unpack4xI8(q.z)) * sc; Bs[row][k4 + 3u] = vec4<f32>(unpack4xI8(q.w)) * sc;
    }`
    : `
    {
      let row = li / 4u; let kq = (li % 4u) * 8u;
      let n = tn + row;
      var q = vec4<u32>(0u);
      if (n < ${N}u) { q = wv[(p.wo * 2u + n * ${wrow}u + wbase + kt + kq) / 8u]; }
      Bs[row][kq / 4u] = vec4<f32>(unpack2x16float(q.x), unpack2x16float(q.y));
      Bs[row][kq / 4u + 1u] = vec4<f32>(unpack2x16float(q.z), unpack2x16float(q.w));
    }`;
  const epilogue =
    splits > 1
      ? `y[(sp * p.rows + orow) * ${N}u + n] = v;`
      : `${bias ? "v += hf(p.bo + n);" : ""}
      v = ${actCall(act, "v")};
      let o = p.yo + orow * ${N}u + n;
      ${residual(res, `p.ro + orow * ${N}u + n`)}
      y[o] = v;`;
  return `
${struct(UNIFORMS.gemm)}
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> r: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> st: array<u32>;
var<workgroup> As: array<array<vec4<f32>, 8>, 64>;
var<workgroup> Bs: array<array<vec4<f32>, 8>, 64>;
${BLOB}
${ACT}
${aLoad}
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  _ = &x; _ = &wv; _ = &y; _ = &r; _ = &st;
  ${GATES[gate]}
  let tm = wid.y * 64u;
  let tn = wid.x * 64u;
  let ph = wid.z / ${splits}u;
  let sp = wid.z % ${splits}u;
  let wbase = ${isT ? `ph * ${2 * cin}u` : "0u"};
  let li = lid.y * 16u + lid.x;
  let k0 = sp * ${Ks}u;
  let k1 = min(${K}u, k0 + ${Ks}u);
  var acc: array<vec4<f32>, 4>;
  for (var kt = k0; kt < k1; kt += 32u) {
    for (var e = 0u; e < 2u; e++) {
      let idx = li + e * 256u;        // 512 vec4 = 64 rows x 8 vec4 (32 k)
      let row = idx / 8u; let k4 = idx % 8u;
      As[row][k4] = loadA(tm + row, kt + k4 * 4u);
    }
    ${bLoad}
    workgroupBarrier();
    for (var k4 = 0u; k4 < 8u; k4++) {
      let a0 = As[lid.y * 4u][k4]; let a1 = As[lid.y * 4u + 1u][k4]; let a2 = As[lid.y * 4u + 2u][k4]; let a3 = As[lid.y * 4u + 3u][k4];
      let b0 = Bs[lid.x * 4u][k4]; let b1 = Bs[lid.x * 4u + 1u][k4]; let b2 = Bs[lid.x * 4u + 2u][k4]; let b3 = Bs[lid.x * 4u + 3u][k4];
      acc[0] += vec4<f32>(dot(a0, b0), dot(a0, b1), dot(a0, b2), dot(a0, b3));
      acc[1] += vec4<f32>(dot(a1, b0), dot(a1, b1), dot(a1, b2), dot(a1, b3));
      acc[2] += vec4<f32>(dot(a2, b0), dot(a2, b1), dot(a2, b2), dot(a2, b3));
      acc[3] += vec4<f32>(dot(a3, b0), dot(a3, b1), dot(a3, b2), dot(a3, b3));
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let m = tm + lid.y * 4u + i;
    if (m >= p.M) { continue; }
    let orow = ${isT ? `(m * ${stride}u + ph)` : "m"};
    for (var j = 0u; j < 4u; j++) {
      let n = tn + lid.x * 4u + j;
      if (n >= ${N}u) { continue; }
      var v = acc[i][j];
      ${epilogue}
    }
  }
}`;
}

/** Finishes a split-K GEMM: sums the partials, then bias, activation and residual. */
export function gemmReduce({
  N,
  splits,
  act = "none",
  bias = false,
  res = "none",
  gate,
}: {
  N: number;
  splits: number;
  act?: Act;
  bias?: boolean;
  res?: Residual;
  gate: Gate;
}): string {
  return `
${struct(UNIFORMS.gemmReduce)}
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> r: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> st: array<u32>;
${BLOB}
${ACT}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  _ = &part; _ = &y; _ = &wv; _ = &r; _ = &st;
  ${GATES[gate]}
  let i = gid.x;
  if (i >= p.rows * ${N}u) { return; }
  let n = i % ${N}u; let row = i / ${N}u;
  var v = 0.0;
  for (var s = 0u; s < ${splits}u; s++) { v += part[(s * p.stride + row) * ${N}u + n]; }
  ${bias ? "v += hf(p.bo + n);" : ""}
  v = ${actCall(act, "v")};
  let o = p.yo + i;
  ${residual(res, "p.ro + i")}
  y[o] = v;
}`;
}

/** A GEMV's layout for K inputs: 128 threads, 2 rows per 32 lanes (more when a row needs
 *  fewer than 32 lanes), 16-byte weight loads. */
export function gemvShape(K: number) {
  const WG = 128;
  const R = 2;
  const U = K / 16; // 16-byte loads per row
  const LPR = Math.min(32, U); // lanes per row
  const RPP = 32 / LPR; // rows per pass of 32 lanes
  const RS = R * RPP; // rows per 32 lanes
  return { WG, R, U, LPR, RPP, RS, rowsPerGroup: (WG / 32) * RS };
}

/** GEMV for one row of an LM step: y[n] = epilogue(sum_k x[k] w[n, k]) with int8 weights.
 *  32 lanes per output row, 2 rows per 32 lanes, 16-byte weight loads. Optional prologues:
 *  LayerNorm (affine and/or adaLN modulation) or the attention splits' merge. Without
 *  subgroups, both rows share one vec2 reduction with the same per-row summation order;
 *  only each segment's first lane uses its sums, and the scratch is not reused. */
export function gemv({
  K,
  N,
  act = "none",
  bias = false,
  res = "none",
  ln = null,
  combine = 0,
  sg,
}: GemvOptions): string {
  const { WG, R, U, LPR, RPP, RS } = gemvShape(K);
  const K4 = K / 4;
  const xo = "p.xo / 4u";
  const pre = ln
    ? `
  // one pass: sum and sum of squares (f32 is ample for 512 values of this size)
  var s1 = 0.0; var s2 = 0.0;
  for (var i = li; i < 128u; i += ${WG}u) { let v = x[${xo} + i]; s1 += v.x + v.y + v.z + v.w; s2 += dot(v, v); }
  ${
    sg
      ? `s1 = subgroupAdd(s1); s2 = subgroupAdd(s2);
  if ((li & 31u) == 0u) { lnr[li >> 5u] = vec2<f32>(s1, s2); }
  workgroupBarrier();`
      : `lnr[li] = vec2<f32>(s1, s2);
  workgroupBarrier();
  for (var s = ${WG / 2}u; s > 0u; s >>= 1u) { if (li < s) { lnr[li] += lnr[li + s]; } workgroupBarrier(); }`
  }
  var t = vec2<f32>(0.0);
  ${sg ? `for (var i = 0u; i < ${WG / 32}u; i++) { t += lnr[i]; }` : "t = lnr[0];"}
  let mean = t.x / 512.0;
  let inv = inverseSqrt(max(t.y / 512.0 - mean * mean, 0.0) + ${ln.eps});
  for (var i = li; i < 128u; i += ${WG}u) {
    var o = (x[${xo} + i] - vec4<f32>(mean)) * inv;
    ${ln.affine ? "o = o * h4(p.lgo + i * 4u) + h4(p.lbo + i * 4u);" : ""}
    ${ln.modulate ? "o = o * (vec4<f32>(1.0) + md[p.so / 4u + i]) + md[p.ho / 4u + i];" : ""}
    xs[i] = o;
  }
  workgroupBarrier();`
    : combine
      ? `
  // part: [split][head] blocks of 66 floats: 64 dims, max, sum
  for (var i = li; i < 128u; i += ${WG}u) {
    let h = i / 16u; let d = (i % 16u) * 4u;
    var mx = -1e30;
    for (var s = 0u; s < ${combine}u; s++) { mx = max(mx, r[(s * 8u + h) * 66u + 64u]); }
    var num = vec4<f32>(0.0); var den = 0.0;
    for (var s = 0u; s < ${combine}u; s++) {
      let b = (s * 8u + h) * 66u;
      let f = exp(r[b + 64u] - mx);
      num += vec4<f32>(r[b + d], r[b + d + 1u], r[b + d + 2u], r[b + d + 3u]) * f;
      den += r[b + 65u] * f;
    }
    xs[i] = num / den;
  }
  workgroupBarrier();`
      : `
  for (var i = li; i < ${K4}u; i += ${WG}u) { xs[i] = x[${xo} + i]; }
  workgroupBarrier();`;
  return `${sg ? "enable subgroups;" : ""}
${struct(UNIFORMS.gemv)}
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> r: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read> gv: array<f32>;
@group(0) @binding(6) var<storage, read> md: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> st: array<u32>;
var<workgroup> xs: array<vec4<f32>, ${Math.max(K4, 128)}>;
${ln ? `var<workgroup> lnr: array<vec2<f32>, ${sg ? WG / 32 : WG}>;` : ""}
${BLOB}
${ACT}
${sg ? subgroupSum(LPR) : `var<workgroup> r32: array<vec2<f32>, ${WG}>;`}
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = &x; _ = &wv; _ = &y; _ = &r; _ = &gv; _ = &md; _ = &st;
  ${GATES.step}
  let lane = li & 31u;
  let ul = lane % ${LPR}u; let sub = lane / ${LPR}u;
  let n0 = (wid.x * ${WG / 32}u + (li >> 5u)) * ${RS}u + sub;
  ${pre}
  var acc: array<f32, ${R}>;
  for (var j = 0u; j < ${R}u; j++) {
    let n = n0 + j * ${RPP}u;
    if (n < ${N}u) {
      let wb = p.wo / 4u + n * ${U}u;
      for (var i = 0u; i < ${U / LPR}u; i++) {
        let u = ul + i * ${LPR}u;
        let q = wv[wb + u];
        let sc = hf(p.wso + n * ${K / Q8_GROUP}u + (u * 16u) / ${Q8_GROUP}u);
        let a = dot(xs[u * 4u], vec4<f32>(unpack4xI8(q.x))) + dot(xs[u * 4u + 1u], vec4<f32>(unpack4xI8(q.y)))
              + dot(xs[u * 4u + 2u], vec4<f32>(unpack4xI8(q.z))) + dot(xs[u * 4u + 3u], vec4<f32>(unpack4xI8(q.w)));
        acc[j] += a * sc;
      }
    }
  }
  ${sg ? "" : `r32[li] = vec2<f32>(acc[0], acc[1]); workgroupBarrier();
  for (var s = ${LPR / 2}u; s > 0u; s >>= 1u) { if (ul < s) { r32[li] += r32[li + s]; } workgroupBarrier(); }
  `}for (var j = 0u; j < ${R}u; j++) {
    let tot = ${sg ? "sumSeg(acc[j], li)" : "r32[li][j]"};
    let n = n0 + j * ${RPP}u;
    if (ul == 0u && n < ${N}u) {
      var v = tot${bias ? " + hf(p.bo + n)" : ""};
      v = ${actCall(act, "v")};
      let o = p.yo + n;
      ${residual(res, "p.ro + n")}
      y[o] = v;
    }
  }
}`;
}

/** The end of an LM step, with the final LayerNorm fused: the EOS logit from LN(x row 0), the
 *  latent kept, the position and frame moved on, and the end set once the logit passes
 *  `threshold` (keeping the state's `after` more frames). */
export function finishStep({ eps, threshold }: { eps: string; threshold: string }): string {
  return `
${struct(UNIFORMS.finishStep)}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> lat: array<f32>;
@group(0) @binding(3) var<storage, read_write> lats: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<u32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> red: array<f32, 256>;
var<workgroup> ended: u32;
${BLOB}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  _ = &x; _ = &wv; _ = &lat; _ = &lats; _ = &st;
  // st is read_write here, so the gate goes through workgroup memory to stay uniform.
  if (li == 0u) { ended = select(0u, 1u, st[${END}] != 0u && st[${FRAME}] >= st[${END}]); }
  if (workgroupUniformLoad(&ended) == 1u) { return; }
  let a = x[li]; let b2 = x[li + 256u];
  red[li] = a + b2; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let mean = red[0] / 512.0; workgroupBarrier();
  red[li] = (a - mean) * (a - mean) + (b2 - mean) * (b2 - mean); workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let inv = inverseSqrt(red[0] / 512.0 + ${eps}); workgroupBarrier();
  let c2 = li + 256u;
  red[li] = ((a - mean) * inv * hf(p.ngo + li) + hf(p.nbo + li)) * hf(p.ewo + li)
          + ((b2 - mean) * inv * hf(p.ngo + c2) + hf(p.nbo + c2)) * hf(p.ewo + c2);
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  let f = st[${FRAME}];
  if (li < 32u) { lats[f * 32u + li] = lat[li]; }
  let logit = red[0] + hf(p.ebo);
  workgroupBarrier();
  storageBarrier();
  if (li == 0u) {
    st[${POS}] = st[${POS}] + 1u; st[${FRAME}] = f + 1u;
    if (st[${END}] == 0u && logit > ${threshold}) { st[${END}] = f + st[${AFTER}] + 1u; }
  }
}`;
}

/** Split-K decode attention (flash-decoding) for an LM step: workgroup (split, head) of 128
 *  threads rotates q, and the split owning the current position writes its k and v (f16)
 *  first. 8 lanes per key for the scores, 16 lanes x 8 key groups for the values. It leaves
 *  each split's unnormalized output, max and sum for the next GEMV to merge. */
export function attnSplit({
  splits,
  theta,
  sg,
  align,
}: {
  splits: number;
  theta: string;
  sg: boolean;
  /** Keep the alignment head's scores over the text, for word times. */
  align?: AlignRows;
}): string {
  const pack = (v: string) => `vec2<u32>(pack2x16float(${v}.xy), pack2x16float(${v}.zw))`;
  return `${sg ? "enable subgroups;" : ""}
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> kc: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> vc: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
@group(0) @binding(4) var<storage, read> st: array<u32>;
${align ? "@group(0) @binding(5) var<storage, read_write> al: array<f32>;" : ""}
fn kv(u: vec2<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(u.x), unpack2x16float(u.y)); }
var<workgroup> qf: array<f32, 64>;
var<workgroup> kf: array<f32, 64>;
var<workgroup> sc: array<f32, 256>;
var<workgroup> pr: array<f32, 128>;
var<workgroup> red: array<f32, 128>;
var<workgroup> acc4: array<vec4<f32>, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = &qkv; _ = &kc; _ = &vc; _ = &part; _ = &st;${align ? " _ = &al;" : ""}
  ${GATES.step}
  let sp = wid.x; let h = wid.y;
  let pos = st[${POS}];
  let hb = h * 64u;
  let n = pos + 1u;
  let per = (n + ${splits - 1}u) / ${splits}u;
  let j0 = sp * per; let j1 = min(n, j0 + per);
  let own = pos >= j0 && pos < j1;
  if (li < 32u) {
    let ang = f32(pos) * pow(${theta}, -f32(li) / 32.0);
    let c = cos(ang); let s = sin(ang);
    let q1 = qkv[hb + li]; let q2 = qkv[hb + li + 32u];
    qf[li] = (q1 * c - q2 * s) * 0.125; qf[li + 32u] = (q2 * c + q1 * s) * 0.125;
    if (own) {
      let k1 = qkv[512u + hb + li]; let k2 = qkv[512u + hb + li + 32u];
      kf[li] = k1 * c - k2 * s; kf[li + 32u] = k2 * c + k1 * s;
    }
  } else if (own && li < 48u) {
    let d = (li - 32u) * 4u;
    let vv = vec4<f32>(qkv[1024u + hb + d], qkv[1025u + hb + d], qkv[1026u + hb + d], qkv[1027u + hb + d]);
    vc[(pos * 512u + hb + d) / 4u] = ${pack("vv")};
  }
  workgroupBarrier();
  if (own && li < 16u) { let k4 = vec4<f32>(kf[li * 4u], kf[li * 4u + 1u], kf[li * 4u + 2u], kf[li * 4u + 3u]); kc[(pos * 512u + hb) / 4u + li] = ${pack("k4")}; }
  storageBarrier();
  workgroupBarrier();
  // scores: a lane group of 8 per key, each lane 8 dims
  let kk = li / 8u; let part8 = li % 8u;
  let qa = vec4<f32>(qf[part8 * 8u], qf[part8 * 8u + 1u], qf[part8 * 8u + 2u], qf[part8 * 8u + 3u]);
  let qb = vec4<f32>(qf[part8 * 8u + 4u], qf[part8 * 8u + 5u], qf[part8 * 8u + 6u], qf[part8 * 8u + 7u]);
  let cnt = j1 - min(j0, j1);
  for (var base = 0u; base < cnt; base += 16u) {
    let j = j0 + base + kk;
    var d = 0.0;
    if (base + kk < cnt) {
      let kb = (j * 512u + hb) / 4u + part8 * 2u;
      d = dot(qa, kv(kc[kb])) + dot(qb, kv(kc[kb + 1u]));
    }
    ${
      sg
        ? `d += subgroupShuffleXor(d, 4u); d += subgroupShuffleXor(d, 2u); d += subgroupShuffleXor(d, 1u);
    if (part8 == 0u && base + kk < cnt) { sc[base + kk] = d; }`
        : `pr[li] = d; workgroupBarrier();
    if (li < 16u && base + li < cnt) { var t = 0.0; for (var e = 0u; e < 8u; e++) { t += pr[li * 8u + e]; } sc[base + li] = t; }
    workgroupBarrier();`
    }
  }
  workgroupBarrier();
  var mx = -1e30;
  for (var j = li; j < cnt; j += 128u) { mx = max(mx, sc[j]); }
  red[li] = mx; workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) { if (li < s) { red[li] = max(red[li], red[li + s]); } workgroupBarrier(); }
  mx = red[0]; workgroupBarrier();
  var sum = 0.0;
  for (var j = li; j < cnt; j += 128u) {
    ${
      align
        ? `let k = j0 + j - ${align.lo}u;
    if (h == ${align.head}u && j0 + j >= ${align.lo}u && k < st[${TEXT}]) { al[st[${FRAME}] * ${align.width}u + k] = sc[j]; }`
        : ""
    }
    let e = exp(sc[j] - mx); sc[j] = e; sum += e;
  }
  red[li] = sum; workgroupBarrier();
  for (var s = 64u; s > 0u; s >>= 1u) { if (li < s) { red[li] += red[li + s]; } workgroupBarrier(); }
  sum = red[0];
  // values: 16 lanes over the head's 64 dims (vec4), 8 key groups
  let dv = li % 16u; let g = li / 16u;
  var a = vec4<f32>(0.0);
  for (var j = g; j < cnt; j += 8u) { a += sc[j] * kv(vc[((j0 + j) * 512u + hb) / 4u + dv]); }
  acc4[li] = a; workgroupBarrier();
  let pb = (sp * 8u + h) * 66u;
  if (li < 16u) {
    var t = vec4<f32>(0.0);
    for (var e = 0u; e < 8u; e++) { t += acc4[e * 16u + li]; }
    part[pb + li * 4u] = t.x; part[pb + li * 4u + 1u] = t.y; part[pb + li * 4u + 2u] = t.z; part[pb + li * 4u + 3u] = t.w;
  }
  if (li == 0u) { part[pb + 64u] = select(mx, -1e30, j1 <= j0); part[pb + 65u] = sum; }
}`;
}


/** Frame 0's alignment row, in the prefill after the alignment layer's attention: the alignment
 *  head's scores of the BOS row (the last text row, at the state's position) over the text. */
export function alignRow({ lo, head }: AlignRows): string {
  return `
@group(0) @binding(0) var<storage, read> qb: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> kc: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> al: array<f32>;
@group(0) @binding(3) var<storage, read> st: array<u32>;
fn kv(u: vec2<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(u.x), unpack2x16float(u.y)); }
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  _ = &qb; _ = &kc; _ = &al; _ = &st;
  let m = st[${POS}] - ${lo}u;
  for (var i = li; i < st[${TEXT}]; i += 256u) {
    var d = 0.0;
    for (var e = 0u; e < 16u; e++) { d += dot(qb[m * 128u + ${head * 16}u + e] * 0.125, kv(kc[(${lo}u + i) * 128u + ${head * 16}u + e])); }
    al[i] = d;
  }
}`;
}
