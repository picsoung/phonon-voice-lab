// src/pace.ts
var H = 512;
function matvec(w, x, rows) {
  const cols = x.length;
  const y = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = 0;
    for (let c = 0; c < cols; c++) s += w[r * cols + c] * x[c];
    y[r] = s;
  }
  return y;
}
function paceVectors(p, delta) {
  const half = 64;
  const phases = new Float32Array(2 * half);
  for (let i = 0; i < half; i++) {
    const a = 1e3 * delta / 1e4 ** (i / (half - 1));
    phases[i] = Math.cos(a);
    phases[half + i] = Math.sin(a);
  }
  const cond = matvec(p.speakersProj, p.speakers, H);
  const dur = matvec(p.durationProj, phases, H);
  for (let i = 0; i < H; i++) cond[i] += dur[i];
  const bos = matvec(p.latentProj, p.bos, H);
  for (let i = 0; i < H; i++) bos[i] += cond[i];
  return { cond, bos };
}

// src/graph.ts
var STATE = { pos: 0, frame: 1, end: 2, chunk: 3, text: 4, after: 5, seed: 6, scale: 7 };

// src/checkpoint.ts
var lmLayers = (m) => Object.keys(m.tensors).filter((n) => /^l\d+\.qkv$/.test(n)).length;
function toF16(values) {
  const f = new Float32Array(1);
  const u = new Uint32Array(f.buffer);
  const out = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) {
    f[0] = values[i];
    const x = u[0];
    const sign = x >>> 16 & 32768;
    const exp = x >>> 23 & 255;
    let mant = x & 8388607;
    if (exp === 255) {
      out[i] = sign | 31744 | (mant ? 512 : 0);
      continue;
    }
    const e = exp - 127 + 15;
    if (e >= 31) {
      out[i] = sign | 31744;
      continue;
    }
    if (e <= 0) {
      if (e < -10) {
        out[i] = sign;
        continue;
      }
      mant |= 8388608;
      const shift = 14 - e;
      let half2 = mant >> shift;
      const rest2 = mant & (1 << shift) - 1;
      const mid = 1 << shift - 1;
      if (rest2 > mid || rest2 === mid && half2 & 1) half2++;
      out[i] = sign | half2;
      continue;
    }
    let half = e << 10 | mant >> 13;
    const rest = mant & 8191;
    if (rest > 4096 || rest === 4096 && half & 1) half++;
    out[i] = sign | half;
  }
  return out;
}
function unpackModel(buffer) {
  const length = new DataView(buffer).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, length)));
  let o = Math.ceil((4 + length) / 4) * 4;
  const pace = {};
  for (const [k, n] of header.pace) {
    pace[k] = new Float32Array(buffer, o, n);
    o += n * 4;
  }
  return { tensors: header.tensors, pace, blob: new Uint8Array(buffer, o) };
}
function withoutWeights(m) {
  const pace = Object.fromEntries(Object.entries(m.pace).map(([k, v]) => [k, v.slice()]));
  return { tensors: m.tensors, pace, blob: new Uint8Array() };
}

// src/files.ts
var hubFile = (repo, revision, path) => `https://huggingface.co/${repo}/resolve/${revision}/${path}`;
var MODEL_CACHE = "phonon-webgpu";
var FORTY_REPO = "blixt/phonon-40m-webgpu";
var FORTY_REVISION = "dde41aa27ed6bed7c874621d8b17e77dfece21bf";
var fortyM = (name, revision, languages, align) => ({
  source: { repo: `gradium/${name}`, revision },
  weights: hubFile(FORTY_REPO, FORTY_REVISION, `${name}.int8.bin`),
  tokenizer: hubFile(FORTY_REPO, FORTY_REVISION, `${name}.tokenizer.5.0.0.json`),
  languages,
  license: "CC-BY-NC-4.0",
  align
});
var SOURCE_31M = { repo: "gradium/phonon-31m-en", revision: "92741c5b07d085ff840e26fea8ce43110b87852f" };
var MODELS = {
  "phonon-31m-en": {
    source: SOURCE_31M,
    weights: hubFile("blixt/phonon-31m-en-webgpu", "f4635a6dddaa6072451e0155c03c97fbe889b6ec", "phonon-31m-en.int8.bin"),
    tokenizer: hubFile(SOURCE_31M.repo, SOURCE_31M.revision, "tokenizer.5.0.0.json"),
    languages: ["en"],
    license: "Apache-2.0",
    align: { layer: 3, head: 0, lag: 3 }
  },
  "phonon-40m-en": fortyM("phonon-40m-en", "bc7e8ae7e543a7a3e4604753e42fbda76ddf0781", ["en"], { layer: 6, head: 0, lag: 3 }),
  "phonon-40m-fr": fortyM("phonon-40m-fr", "d91086970206b88efd2c88ca971d0628b5968f0f", ["fr"], { layer: 4, head: 1, lag: 3 }),
  "phonon-40m-de": fortyM("phonon-40m-de", "79cfc7b47fcfc0b666657b981eb366415023a862", ["de"], { layer: 3, head: 5, lag: 2 }),
  "phonon-40m-es": fortyM("phonon-40m-es", "e5e052b26e31abd5918ade71d3135e69d51a66f8", ["es"], { layer: 4, head: 6, lag: 3 }),
  "phonon-40m-pt": fortyM("phonon-40m-pt", "4cbfd742a1555d60eb5399c6d296e9f2e07389e9", ["pt"], { layer: 4, head: 2, lag: 3 }),
  "phonon-40m-multilingual": fortyM("phonon-40m-multilingual", "1537ce46c64f42903e0d608b132ccb073dac46d2", ["en", "fr", "de", "es", "pt"], { layer: 4, head: 1, lag: 3 })
};
var VOICES_REPO = "gradium/phonon-voices";
var VOICES_REVISION = "0ee17205f254b5abc43b266c8a0f215565c352f7";
var CATALOG_URL = hubFile(VOICES_REPO, VOICES_REVISION, "voices.jsonl");

// src/cache.ts
async function open() {
  try {
    return await caches.open(MODEL_CACHE);
  } catch {
    return null;
  }
}
async function lookup(key) {
  try {
    return await (await open())?.match(key);
  } catch {
    return void 0;
  }
}
async function store(key, body) {
  try {
    await (await open())?.put(key, new Response(body));
  } catch {
  }
}

// src/lm.ts
var H2 = 512;
var FF = 2048;
var PROMPT = 125;
var S_MAX = 1024;
var F_MAX = 768;
var TEXT_MAX = 240;
var PREFILL = [64, 128, 192, 256];
var BOS = 4294967294;
var PAD = 4294967295;
var LN_EPS = "1e-5";
var FLOW_LN_EPS = "1e-6";
var ROPE_THETA = "10000.0";
var EOS_THRESHOLD = "4.0";
var SPLITS = 8;
function rowBuffers(g, rows) {
  const r = (n) => g.buffer(rows * n * 4);
  return { x: r(H2), h: r(H2), qkv: r(3 * H2), q: r(H2), att: r(H2), ff: r(FF) };
}
function layerRows(g, pre, o) {
  const { B, rows, cache } = o;
  const gate = cache.kind === "codec" ? "chunk" : o.text ? "text" : "none";
  const at = { rows, pos0: o.pos0 };
  const kv = [
    g.layernorm(B.x, B.h, rows, `${pre}.ln1.w`, `${pre}.ln1.b`, LN_EPS, gate),
    ...g.gemm(B.h, `${pre}.qkv`, B.qkv, { M: rows, K: H2, N: 3 * H2, gate }),
    g.ropeKV(B.qkv, B.q, o.kc, o.vc, at, { theta: ROPE_THETA, cache, gate })
  ];
  if (o.kvOnly) return kv;
  return [
    ...kv,
    g.attention(B.q, o.kc, o.vc, B.att, at, { cache, gate }),
    ...g.gemm(B.att, `${pre}.o`, B.x, { M: rows, K: H2, N: H2, res: "add", rbuf: B.x, gate }),
    g.layernorm(B.x, B.h, rows, `${pre}.ln2.w`, `${pre}.ln2.b`, LN_EPS, gate),
    ...g.gemm(B.h, `${pre}.fc1`, B.ff, { M: rows, K: H2, N: FF, act: "gelu", gate }),
    ...g.gemm(B.ff, `${pre}.fc2`, o.out?.buf ?? B.x, {
      M: rows,
      K: FF,
      N: H2,
      res: "add",
      rbuf: B.x,
      yo: o.out?.yo ?? 0,
      gate
    })
  ];
}
var LanguageModel = class {
  /** The latents made so far, one row of 32 per frame. */
  lats;
  /** One step: a latent, then the next position. */
  step;
  /** The alignment head's scores over the text, one row of TEXT_MAX per frame. */
  align;
  g;
  rows;
  kc;
  vc;
  ids;
  /** The BOS row's output on its way to row 0, where the step's tail reads. */
  bos;
  voiceIn;
  voiceOps;
  prefills;
  tail;
  cond;
  ada;
  fh;
  ft;
  z;
  lat;
  parts;
  depth;
  /** The head that follows the text as it speaks (files.ts), its scores kept for word times. */
  alignLayer;
  alignAt;
  constructor(g, depth, align) {
    this.g = g;
    this.depth = depth;
    this.alignLayer = align.layer;
    this.alignAt = { lo: PROMPT, width: TEXT_MAX, head: align.head };
    const rows = PREFILL.at(-1);
    this.rows = rowBuffers(g, rows);
    this.kc = Array.from({ length: depth }, () => g.buffer(S_MAX * H2 * g.kvBytes));
    this.vc = Array.from({ length: depth }, () => g.buffer(S_MAX * H2 * g.kvBytes));
    this.cond = g.buffer(H2 * 4);
    this.ada = g.buffer(10240 * 4);
    this.fh = g.buffer(H2 * 4);
    this.ft = g.buffer(H2 * 4);
    this.z = g.buffer(32 * 4);
    this.lat = g.buffer(32 * 4);
    this.lats = g.buffer(F_MAX * 32 * 4);
    this.ids = g.buffer(rows * 4);
    this.bos = g.buffer(H2 * 4);
    this.parts = g.buffer(SPLITS * 8 * 66 * 4);
    this.voiceIn = g.buffer(PROMPT * H2 * 4);
    this.align = g.buffer(F_MAX * TEXT_MAX * 4);
    this.tail = this.tailOps();
    this.step = [...Array.from({ length: depth }, (_, i) => this.stepLayer(i)).flat(), ...this.tail];
    this.voiceOps = [
      ...g.gemm(this.voiceIn, "prompt_proj", this.rows.x, { M: PROMPT, K: H2, N: H2, gate: "none" }),
      ...this.layers(PROMPT)
    ];
    this.prefills = new Map(
      PREFILL.map((n) => [
        n,
        [g.embedRows(this.ids, this.rows.x, n, { bos: BOS, pad: PAD }), ...this.layers(n, PROMPT)]
      ])
    );
  }
  /** The layers over `rows` rows from position `pos0`, in the prefill's way, and for a text's
   *  prefill frame 0's alignment row. */
  layers(rows, pos0 = 0) {
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
        text: pos0 > 0
      }),
      ...pos0 && i === this.alignLayer ? [g.alignRow(this.rows.q, this.kc[i], this.align, this.alignAt)] : []
    ]).flat();
  }
  /** A voice's prompt rows through every layer, its keys and values kept in a new buffer. */
  voice(latents) {
    const { g } = this;
    const bytes = PROMPT * H2 * g.kvBytes;
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
  start(enc, ids, voice, seed, temperature, framesAfterEos) {
    const { g } = this;
    const n = PREFILL.find((p) => p > ids.length);
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
    const bytes = PROMPT * H2 * g.kvBytes;
    for (let i = 0; i < this.depth; i++) {
      enc.copy(voice, 2 * i * bytes, this.kc[i], 0, bytes);
      enc.copy(voice, (2 * i + 1) * bytes, this.vc[i], 0, bytes);
    }
    enc.run(this.prefills.get(n));
    enc.copy(this.rows.x, ids.length * H2 * 4, this.bos, 0, H2 * 4);
    enc.copy(this.bos, 0, this.rows.x, 0, H2 * 4);
    enc.run(this.tail);
  }
  /** A step's layer: one row at the state's position, attention split over the keys. */
  stepLayer(i) {
    const { g } = this;
    const B = this.rows;
    const pre = `l${i}`;
    return [
      g.gemv(B.x, `${pre}.qkv`, B.qkv, {
        K: H2,
        N: 3 * H2,
        ln: { w: `${pre}.ln1.w`, b: `${pre}.ln1.b`, eps: LN_EPS }
      }),
      g.attnStep(B.qkv, this.kc[i], this.vc[i], this.parts, {
        splits: SPLITS,
        theta: ROPE_THETA,
        align: i === this.alignLayer ? { at: this.alignAt, buf: this.align } : void 0
      }),
      g.gemv(null, `${pre}.o`, B.x, {
        K: H2,
        N: H2,
        res: "add",
        rbuf: B.x,
        combine: { splits: SPLITS, parts: this.parts }
      }),
      g.gemv(B.x, `${pre}.fc1`, B.ff, {
        K: H2,
        N: FF,
        act: "gelu",
        ln: { w: `${pre}.ln2.w`, b: `${pre}.ln2.b`, eps: LN_EPS }
      }),
      g.gemv(B.ff, `${pre}.fc2`, B.x, { K: FF, N: H2, res: "add", rbuf: B.x })
    ];
  }
  /** From the final hidden state (row 0 of x): the flow head samples the next latent, then its
   *  EOS logit, the next input row and the next position. */
  tailOps() {
    const { g } = this;
    const x = this.rows.x;
    const blocks = Array.from({ length: 6 }, (_, i) => {
      const a = i * 1536;
      return [
        g.gemv(this.fh, `fh${i}.fc1.w`, this.ft, {
          K: H2,
          N: H2,
          bias: `fh${i}.fc1.b`,
          act: "silu",
          ln: {
            w: `fh${i}.ln.w`,
            b: `fh${i}.ln.b`,
            eps: FLOW_LN_EPS,
            mod: this.ada,
            ho: a,
            so: a + 512
          }
        }),
        g.gemv(this.ft, `fh${i}.fc2.w`, this.fh, {
          K: H2,
          N: H2,
          bias: `fh${i}.fc2.b`,
          res: "gate",
          rbuf: this.fh,
          gbuf: this.ada,
          go: a + 1024
        })
      ];
    }).flat();
    const fa = 6 * 1536;
    return [
      g.gemv(x, "fh.cond.w", this.cond, {
        K: H2,
        N: H2,
        bias: "fh.cond.b",
        act: "silu",
        ln: { w: "norm.w", b: "norm.b", eps: LN_EPS }
      }),
      g.gemv(this.cond, "fh.ada.w", this.ada, { K: H2, N: 10240, bias: "fh.ada.b" }),
      g.noise(this.z),
      g.gemv(this.z, "fh.in.w", this.fh, { K: 32, N: H2, bias: "fh.in.b" }),
      ...blocks,
      g.gemv(this.fh, "fh.out.w", this.lat, {
        K: H2,
        N: 32,
        bias: "fh.out.b",
        res: "add",
        rbuf: this.z,
        ln: { eps: FLOW_LN_EPS, mod: this.ada, ho: fa, so: fa + 512 }
      }),
      g.finishStep(x, this.lat, this.lats, { eps: LN_EPS, threshold: EOS_THRESHOLD }),
      g.gemv(this.lat, "latent_proj", x, { K: 32, N: H2, res: "vec", vname: "cond" })
    ];
  }
};

// src/simd.ts
var RELAXED = new Uint8Array([
  0,
  97,
  115,
  109,
  1,
  0,
  0,
  0,
  1,
  5,
  1,
  96,
  0,
  1,
  123,
  3,
  2,
  1,
  0,
  10,
  15,
  1,
  13,
  0,
  65,
  0,
  253,
  15,
  65,
  0,
  253,
  15,
  253,
  128,
  2,
  11
]);
var SIMD = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);
function simdLevel() {
  if (WebAssembly.validate(RELAXED)) return "relaxed";
  if (WebAssembly.validate(SIMD)) return "simd";
  return null;
}

// src/cpu.ts
var TILES = [
  { dot: true, rows: 2, cols: 4 },
  { dot: true, rows: 4, cols: 4 },
  { dot: true, rows: 2, cols: 8 },
  { dot: false, rows: 8, cols: 8 },
  { dot: false, rows: 4, cols: 16 }
];
var KIND = { copy: 0, clear: 1, gemv: 2, gemm: 3, layernorm: 4, rope: 5, attention: 6, attnStep: 7, alignRow: 8, embed: 9, noise: 10, finish: 11, upsample: 12, shift: 13, elu: 14 };
var FIELDS = {
  gemv: ["x", "xo", "w", "ws", "y", "K", "N", "act", "bias", "res", "r", "g", "go", "v", "ln", "lnw", "lnb", "eps", "mod", "so", "ho", "splits", "parts", "state"],
  gemm: ["x", "w", "ws", "y", "M", "K", "N", "act", "bias", "res", "r", "yo", "ro", "load", "cin", "stride", "gate", "state"],
  layernorm: ["x", "y", "rows", "w", "b", "eps", "gate", "state"],
  rope: ["qkv", "q", "kc", "vc", "rows", "pos0", "freqs", "codec", "ring", "state", "table", "tableRows", "gate"],
  attention: ["q", "kc", "vc", "out", "rows", "pos0", "codec", "window", "ring", "state", "gate"],
  attnStep: ["qkv", "kc", "vc", "parts", "splits", "freqs", "align", "al", "lo", "width", "head", "state", "table", "tableRows"],
  alignRow: ["q", "kc", "al", "lo", "head", "state"],
  embed: ["ids", "x", "rows", "bos", "pad", "w", "ws", "bosv", "state"],
  noise: ["z", "state"],
  finish: ["x", "lat", "lats", "state", "ew", "eb", "nw", "nb", "eps", "threshold"],
  upsample: ["lats", "y", "T", "w", "ws", "mean", "std", "state"],
  elu: ["src", "dst", "count", "gate", "state"]
};
var ACT = { none: 0, gelu: 1, silu: 2 };
var GATE = { none: 0, step: 1, chunk: 2, text: 3 };
var LOAD = { linear: 0, conv: 1, convT: 2 };
var PAGE = 65536;
var SCRATCH_PER = 256 * 1024;
var STACK = 256 * 1024;
var MAX_PAGES = 16384;
function unshare(bytes) {
  let at = 8;
  const leb = () => {
    let v = 0;
    let shift = 0;
    for (; ; ) {
      const b = bytes[at++];
      v |= (b & 127) << shift;
      if (!(b & 128)) return v >>> 0;
      shift += 7;
    }
  };
  while (at < bytes.length) {
    const id = bytes[at++];
    const size = leb();
    const end = at + size;
    if (id === 2) {
      const skip = () => {
        const n = leb();
        at += n;
      };
      for (let n = leb(); n > 0; n--) {
        skip();
        skip();
        const kind = bytes[at++];
        if (kind === 2) {
          if (bytes[at] & 2) bytes[at] &= ~2;
          return bytes;
        }
        if (kind === 0) leb();
        else if (kind === 1) {
          at++;
          const flags = bytes[at++];
          leb();
          if (flags & 1) leb();
        } else if (kind === 3) at += 2;
        else throw new Error("Unexpected import");
      }
    }
    at = end;
  }
  return bytes;
}
async function instantiate(wasm, shared, bytes) {
  const module = await WebAssembly.compile(shared ? wasm : unshare(wasm));
  const need = Math.ceil(bytes / PAGE);
  let memory = null;
  for (const maximum of [MAX_PAGES, 8192, 6144, 4096]) {
    try {
      memory = new WebAssembly.Memory({ initial: Math.min(need, maximum), maximum, shared });
      break;
    } catch {
    }
  }
  if (!memory) throw new Error("No memory for the CPU engine.");
  const instance = await WebAssembly.instantiate(module, { env: { memory } });
  return { module, memory, exports: { ...instance.exports, memory } };
}
var f32bits = (v) => new Uint32Array(new Float32Array([v]).buffer)[0];
var Staging = class {
  bytes;
  constructor(bytes) {
    this.bytes = new ArrayBuffer(bytes);
  }
  map() {
    return new Promise((done) => {
      const c = new MessageChannel();
      c.port1.onmessage = () => {
        c.port1.close();
        done();
      };
      c.port2.postMessage(null);
    });
  }
  range() {
    return this.bytes;
  }
  unmap() {
  }
};
var Encoder = class {
  g;
  commands = [];
  constructor(g) {
    this.g = g;
  }
  run(ops) {
    for (const op of ops) this.commands.push({ op });
  }
  copy(src, so, dst, dO, bytes) {
    this.commands.push({ op: this.g.temp(KIND.copy, [src.ptr + so, dst.ptr + dO, bytes]) });
  }
  clear(buf, offset, bytes) {
    this.commands.push({ op: this.g.temp(KIND.clear, [buf.ptr + offset, bytes]) });
  }
  read(src, so, dst, dO, bytes) {
    this.commands.push({
      read: () => new Uint8Array(dst.bytes, dO, bytes).set(this.g.bytes(src.ptr + so, bytes))
    });
  }
  submit() {
    const ops = [];
    for (const c of this.commands) {
      if ("op" in c) ops.push(c.op);
      else {
        this.g.exec(ops.splice(0));
        c.read();
      }
    }
    this.g.exec(ops);
    this.g.endSubmit();
  }
};
var CpuGraph = class {
  weights;
  state;
  depth = 1;
  kvBytes = 4;
  threads;
  x;
  tensors;
  top;
  list;
  /** Descriptors of one submit's copies and clears, reused by the next. */
  temps;
  tempAt = 0;
  freqs = /* @__PURE__ */ new Map();
  table = 0;
  tableTheta = "";
  /** ELU'd inputs of convolutions: one buffer for all, and the descriptor fields that name it. */
  elu = null;
  eluRefs = [];
  /** `x` is the instantiated module, `stacks` the bytes the pool threads' stacks took from the
   *  heap base, and `blob` the weights, copied into the memory. */
  constructor(x, model, o) {
    this.x = x;
    this.tensors = model.tensors;
    this.threads = o.threads;
    this.top = o.heap;
    this.weights = this.buffer(model.blob.byteLength);
    this.bytes(this.weights.ptr, model.blob.byteLength).set(model.blob);
    this.state = this.buffer(32);
    this.list = this.buffer(4 * 1024);
    this.temps = this.buffer(4 * 4096);
    const scratch = this.buffer(SCRATCH_PER * o.threads);
    x.setup(o.threads, scratch.ptr, SCRATCH_PER);
  }
  /** Bytes from the heap, 64-byte aligned, zeroed. The memory grows as needed. */
  buffer(size) {
    const ptr = Math.ceil(this.top / 64) * 64;
    this.top = ptr + Math.max(size, 4);
    const mem = this.x.memory;
    if (this.top > mem.buffer.byteLength) mem.grow(Math.ceil((this.top - mem.buffer.byteLength) / PAGE));
    return { ptr, size };
  }
  /** The memory's bytes (a fresh view: the memory can grow). */
  bytes(ptr, n) {
    return new Uint8Array(this.x.memory.buffer, ptr, n);
  }
  write(buf, offset, data) {
    this.bytes(buf.ptr + offset, data.byteLength).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  }
  encoder() {
    return new Encoder(this);
  }
  staging(bytes) {
    return new Staging(bytes);
  }
  async ready() {
  }
  labels = /* @__PURE__ */ new Map();
  times = /* @__PURE__ */ new Map();
  /** Kernel variants that compute the same ops (cpu/src/pool.rs), for the fastest on a device. */
  tune(v) {
    this.x.set_tile(v.tile.dot ? 1 : 0, v.tile.rows, v.tile.cols);
    this.x.set_gemv(v.gemv === "int16" ? 1 : 0);
  }
  /** The fastest GEMM tile here: each of TILES on two of the codec's products, best of three. */
  fastestTile() {
    const M = 32;
    const x = this.buffer((M + 6) * 512 * 4);
    const h = this.buffer(M * 2048 * 4);
    const ops = [
      ...this.gemm(x, "d0.fc1", h, { M, K: 512, N: 2048, gate: "none" }),
      ...this.gemm(x, "s0.w", h, { M, K: 7 * 512, N: 512, gate: "none", load: "conv", conv: { cin: 512, k: 7 } })
    ];
    let best = { tile: TILES[0], ms: Infinity };
    for (const tile2 of TILES) {
      this.tune({ tile: tile2, gemv: "int16" });
      this.exec(ops);
      for (let i = 0; i < 3; i++) {
        const t = performance.now();
        this.exec(ops);
        const ms = performance.now() - t;
        if (ms < best.ms) best = { tile: tile2, ms };
      }
    }
    return best.tile;
  }
  /** Runs ops now, in one call. */
  exec(ops) {
    if (!ops.length) return;
    if (false) {
      for (const op of ops) {
        new Uint32Array(this.x.memory.buffer, this.list.ptr, 1)[0] = op;
        const t = performance.now();
        this.x.run(this.list.ptr, 1);
        const label = this.labels.get(op) ?? "copy";
        this.times.set(label, (this.times.get(label) ?? 0) + performance.now() - t);
      }
      return;
    }
    if (ops.length * 4 > this.list.size) this.list = this.buffer(ops.length * 8);
    new Uint32Array(this.x.memory.buffer, this.list.ptr, ops.length).set(ops);
    this.x.run(this.list.ptr, ops.length);
  }
  endSubmit() {
    this.tempAt = 0;
  }
  /** A descriptor for this submit only. */
  temp(kind, values) {
    const words = 1 + values.length;
    if ((this.tempAt + words) * 4 > this.temps.size) throw new Error("Too many copies in one submit");
    const ptr = this.temps.ptr + this.tempAt * 4;
    new Uint32Array(this.x.memory.buffer, ptr, words).set([kind, ...values]);
    this.tempAt += words;
    return ptr;
  }
  desc(kind, values, extra = []) {
    const fields = FIELDS[kind];
    const given = values;
    const words = [KIND[kind], ...fields.map((f) => given[f] ?? 0), ...extra];
    const b = this.buffer(words.length * 4);
    new Uint32Array(this.x.memory.buffer, b.ptr, words.length).set(words.map((v) => v >>> 0));
    if (false) this.labels.set(b.ptr, kind === "gemv" || kind === "gemm" ? this.label : kind);
    return b.ptr;
  }
  /** The name the next ops are timed under, in a profile build. */
  label = "";
  tensor(name) {
    const t = this.tensors[name];
    if (!t) throw new Error(`No tensor ${name}`);
    return t;
  }
  /** A tensor's address, and its scales' (0 for f16). */
  t(name) {
    const t = this.tensor(name);
    return { w: this.weights.ptr + t.offset, ws: t.scales === void 0 ? 0 : this.weights.ptr + t.scales };
  }
  half(name) {
    if (!name) return 0;
    const t = this.tensor(name);
    if (t.scales !== void 0) throw new Error(`${name} is int8`);
    return this.weights.ptr + t.offset;
  }
  q8(name) {
    const t = this.t(name);
    if (!t.ws) throw new Error(`${name} is not int8`);
    return t;
  }
  /** The 32 rotary frequencies of `theta` in f32, and a table of their cos and sin for the
   *  LM's positions. */
  rope(theta) {
    let f = this.freqs.get(theta);
    if (f === void 0) {
      const b = this.buffer(32 * 4);
      new Float32Array(this.x.memory.buffer, b.ptr, 32).set(
        Array.from({ length: 32 }, (_, d) => Math.fround(Number(theta) ** (-d / 32)))
      );
      f = b.ptr;
      this.freqs.set(theta, f);
    }
    if (!this.table) {
      const t = this.buffer(S_MAX * 64 * 4);
      this.x.rope_table(f, t.ptr, S_MAX);
      this.table = t.ptr;
      this.tableTheta = theta;
    }
    const table = this.tableTheta === theta ? this.table : 0;
    return { freqs: f, table, tableRows: table ? S_MAX : 0 };
  }
  gemv(x, wname, y, o) {
    const { ln, combine } = o;
    const res = o.res ?? "none";
    const w = this.q8(wname);
    if (false) this.label = `gemv ${wname.replace(/\d+/g, "")}`;
    return this.desc("gemv", {
      x: x?.ptr,
      xo: o.xo,
      ...w,
      y: y.ptr,
      K: o.K,
      N: o.N,
      act: ACT[o.act ?? "none"],
      bias: this.half(o.bias),
      res: { none: 0, add: 1, add_y: 1, gate: 2, gate_y: 2, vec: 3 }[res],
      r: o.rbuf?.ptr,
      g: o.gbuf?.ptr,
      go: o.go,
      v: this.half(o.vname),
      ln: ln ? 1 | (ln.w ? 2 : 0) | (ln.mod ? 4 : 0) : 0,
      lnw: this.half(ln?.w),
      lnb: this.half(ln?.b),
      eps: ln ? f32bits(Number(ln.eps)) : 0,
      mod: ln?.mod?.ptr,
      so: ln?.so,
      ho: ln?.ho,
      splits: combine?.splits,
      parts: combine?.parts.ptr,
      state: this.state.ptr
    });
  }
  gemm(x, wname, y, o) {
    const load2 = o.load ?? "linear";
    const conv = o.conv;
    const cin = conv?.cin ?? 0;
    const gate = GATE[o.gate];
    const ops = [];
    let src = x.ptr;
    if (conv?.elu && load2 !== "linear") {
      const rows = load2 === "conv" ? o.M + (conv.k ?? 1) - 1 : o.M + 1;
      const count = rows * cin;
      if (!this.elu || this.elu.size < count * 4) {
        this.elu = this.buffer(count * 4);
        const words = new Uint32Array(this.x.memory.buffer);
        for (const at of this.eluRefs) words[at / 4] = this.elu.ptr;
      }
      const op2 = this.desc("elu", { src: x.ptr, dst: this.elu.ptr, count, gate, state: this.state.ptr });
      this.eluRefs.push(op2 + 8);
      ops.push(op2);
      src = this.elu.ptr;
    }
    if (false) this.label = `${wname.replace(/\d+/g, "")} M${o.M}`;
    const op = this.desc("gemm", {
      x: src,
      ...this.t(wname),
      y: y.ptr,
      M: o.M,
      K: o.K,
      N: o.N,
      act: ACT[o.act ?? "none"],
      bias: this.half(o.bias),
      res: o.res && o.res !== "none" ? 1 : 0,
      r: o.rbuf?.ptr ?? y.ptr,
      yo: o.yo,
      ro: o.ro,
      load: LOAD[load2],
      cin,
      stride: conv?.stride ?? 1,
      gate,
      state: this.state.ptr
    });
    if (src !== x.ptr) this.eluRefs.push(op + 4);
    ops.push(op);
    return ops;
  }
  layernorm(x, y, rows, w, b, eps, gate) {
    return this.desc("layernorm", {
      x: x.ptr,
      y: y.ptr,
      rows,
      w: this.half(w),
      b: this.half(b),
      eps: f32bits(Number(eps)),
      gate: GATE[gate],
      state: this.state.ptr
    });
  }
  ropeKV(qkv, q, kc, vc, at, o) {
    const codec = o.cache.kind === "codec";
    return this.desc("rope", {
      qkv: qkv.ptr,
      q: q.ptr,
      kc: kc.ptr,
      vc: vc.ptr,
      rows: at.rows,
      pos0: at.pos0,
      ...this.rope(o.theta),
      codec: codec ? 1 : 0,
      ring: o.cache.kind === "codec" ? o.cache.ring : 0,
      state: this.state.ptr,
      gate: GATE[o.gate ?? "none"]
    });
  }
  attention(q, kc, vc, out, at, o) {
    const c = o.cache;
    return this.desc("attention", {
      q: q.ptr,
      kc: kc.ptr,
      vc: vc.ptr,
      out: out.ptr,
      rows: at.rows,
      pos0: at.pos0,
      codec: c.kind === "codec" ? 1 : 0,
      window: c.kind === "codec" ? c.window : 0,
      ring: c.kind === "codec" ? c.ring : 0,
      state: this.state.ptr,
      gate: GATE[o.gate ?? "none"]
    });
  }
  attnStep(qkv, kc, vc, parts, o) {
    return this.desc("attnStep", {
      qkv: qkv.ptr,
      kc: kc.ptr,
      vc: vc.ptr,
      parts: parts.ptr,
      splits: o.splits,
      ...this.rope(o.theta),
      align: o.align ? 1 : 0,
      al: o.align?.buf.ptr,
      lo: o.align?.at.lo,
      width: o.align?.at.width,
      head: o.align?.at.head,
      state: this.state.ptr
    });
  }
  alignRow(q, kc, al, at) {
    return this.desc("alignRow", { q: q.ptr, kc: kc.ptr, al: al.ptr, lo: at.lo, head: at.head, state: this.state.ptr });
  }
  embedRows(ids, x, rows, o) {
    return this.desc("embed", { ids: ids.ptr, x: x.ptr, rows, bos: o.bos, pad: o.pad, ...this.q8("embed"), bosv: this.half("bos_embed"), state: this.state.ptr });
  }
  noise(z) {
    return this.desc("noise", { z: z.ptr, state: this.state.ptr });
  }
  finishStep(x, lat, lats, o) {
    return this.desc("finish", {
      x: x.ptr,
      lat: lat.ptr,
      lats: lats.ptr,
      state: this.state.ptr,
      ew: this.half("eos.w"),
      eb: this.half("eos.b"),
      nw: this.half("norm.w"),
      nb: this.half("norm.b"),
      eps: f32bits(Number(o.eps)),
      threshold: f32bits(Number(o.threshold))
    });
  }
  upsample(lats, y, T) {
    return this.desc("upsample", {
      lats: lats.ptr,
      y: y.ptr,
      T,
      ...this.q8("up"),
      mean: this.half("lat_mean"),
      std: this.half("lat_std"),
      state: this.state.ptr
    });
  }
  shiftRows(stages, T) {
    const words = [KIND.shift, T, stages.length, ...stages.flatMap((s) => [s.buf.ptr, s.ctx, s.rate, s.ch])];
    const b = this.buffer(words.length * 4);
    new Uint32Array(this.x.memory.buffer, b.ptr, words.length).set(words);
    return b.ptr;
  }
};

// src/codec.ts
var CHUNKS = [1, 2, 4, 8];
var C_MAX = 8;
var FRAME = 1920;
var WINDOW = 251;
var RING = 512;
var Codec = class {
  /** The last chunk's samples, up to C_MAX * FRAME floats. */
  audio;
  g;
  ops;
  /** Each stage: `ctx` rows of left context (a reading convolution's (kernel - 1) * dilation),
   *  then `rate` rows per 200 Hz position. */
  stages;
  constructor(g, lats) {
    this.g = g;
    const T = 16 * C_MAX;
    const stage = (ctx, rate, ch) => ({
      ctx,
      rate,
      ch,
      buf: g.buffer((ctx + rate * T) * ch * 4)
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
    const rings = [0, 1].map(() => ({ kc: g.buffer(RING * H2 * 4), vc: g.buffer(RING * H2 * 4) }));
    const gate = "chunk";
    const opsFor = (C) => {
      const T2 = 16 * C;
      const conv = (name, src, dst, o) => {
        if (src.ctx !== o.k - 1) throw new Error(`${name} reads ${src.ctx} rows of context`);
        const out = dst === this.audio ? { buf: dst, ctx: 0 } : dst;
        return g.gemm(src.buf, `${name}.w`, out.buf, {
          M: src.rate * T2,
          K: o.k * o.cin,
          N: o.cout,
          bias: `${name}.b`,
          load: "conv",
          conv: { cin: o.cin, k: o.k, dil: 1, elu: o.elu },
          res: o.res ? "add" : "none",
          rbuf: o.res?.buf,
          ro: o.res ? o.res.ctx * o.cout : 0,
          yo: out.ctx * o.cout,
          gate
        });
      };
      const convT = (name, src, dst, o) => g.gemm(src.buf, `${name}.w`, dst.buf, {
        M: src.rate * T2,
        K: 2 * o.cin,
        N: o.cout,
        bias: `${name}.b`,
        load: "convT",
        conv: { cin: o.cin, elu: true, stride: o.s },
        yo: dst.ctx * o.cout,
        gate
      });
      return [
        g.upsample(lats, B.x, T2),
        ...[0, 1].flatMap(
          (i) => layerRows(g, `d${i}`, {
            B,
            rows: T2,
            ...rings[i],
            cache: { kind: "codec", window: WINDOW, ring: RING },
            pos0: 0,
            out: i === 1 ? { buf: tr.buf, yo: tr.ctx * 512 } : void 0
          })
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
        g.shiftRows(this.stages, T2)
      ];
    };
    this.ops = new Map(CHUNKS.map((c) => [c, opsFor(c)]));
  }
  /** Before a text's first chunk: no context yet. */
  reset(enc) {
    for (const st of this.stages) enc.clear(st.buf, 0, st.ctx * st.ch * 4);
  }
  /** Decodes frames [f0, f0 + size) into `audio`. The position goes to the queue ahead of the
   *  submit that holds this encoder, so a submit holds one chunk at most. */
  encodeChunk(enc, f0, size) {
    this.g.write(this.g.state, STATE.chunk * 4, new Uint32Array([16 * f0]));
    enc.run(this.ops.get(size));
  }
};

// src/engine.ts
var TEMPERATURE = 0.3;
var FRAMES_AFTER_EOS = 1;
var FRAMES_AFTER_EOS_SHORT = 3;
var SHORT_TEXT_WORDS = 4;
var FRAMES_PER_SECOND = 12.5;
var AUDIO_AT = 16;
var ROWS = Math.max(...CHUNKS);
var PhononEngine = class {
  g;
  model;
  lm;
  codec;
  /** Each voice's prompt keys and values (lm.ts). */
  voices = /* @__PURE__ */ new Map();
  stages;
  rowsAt;
  pace = Number.NaN;
  /** Settles when the text in progress has drained. */
  turn = Promise.resolve();
  /** `g` holds the model's weights. */
  constructor(g, model, align) {
    this.model = model;
    this.g = g;
    this.lm = new LanguageModel(this.g, lmLayers(model), align);
    this.codec = new Codec(this.g, this.lm.lats);
    this.rowsAt = AUDIO_AT + ROWS * FRAME * 4;
    this.stages = Array.from({ length: g.depth }, () => g.staging(this.rowsAt + ROWS * TEXT_MAX * 4));
    this.setPace(0);
  }
  /** Settles when every shader has compiled. */
  ready() {
    return this.g.ready();
  }
  /** Registers a voice by its speaker latents, [125][512]. Its prefill uses the LM's caches, so
   *  it runs in the queue, before a text (`synthesize`'s `latents`). */
  addVoice(name, latents) {
    if (latents.length !== PROMPT * H2)
      throw new Error(`Voice ${name} has ${latents.length} values`);
    this.voices.set(name, this.lm.voice(latents));
  }
  hasVoice(name) {
    return this.voices.has(name);
  }
  /** The LM's pace (pace.ts): its condition and BOS row are written into the blob, so it is set
   *  in the queue, before a text (`synthesize`'s `pace`). */
  setPace(delta) {
    if (delta === this.pace) return;
    this.pace = delta;
    const { cond, bos } = paceVectors(this.model.pace, delta);
    for (const [name, values] of [
      ["cond", cond],
      ["bos_embed", bos]
    ])
      this.g.write(this.g.weights, this.model.tensors[name].offset, toF16(values));
  }
  /** Text ids to 24 kHz audio in one voice, in chunks as they are made, with the alignment
   *  head's scores of each kept frame over the text as they are made. Null when cancelled. One
   *  text at a time: a call waits until the one before has drained, cancelled or not. */
  synthesize(ids, voice, options = {}) {
    const next = this.turn.then(() => this.run(ids, voice, options));
    this.turn = next.then(() => {
    }, () => {
    });
    return next;
  }
  async run(ids, voice, { seed = 1, words = Infinity, pace = 0, latents, cancelled: cancelled2 = () => false, onChunk, onRows }) {
    if (!this.voices.has(voice)) {
      if (!latents) throw new Error(`Unknown voice ${voice}`);
      this.addVoice(voice, latents);
    }
    const kv = this.voices.get(voice);
    this.setPace(pace);
    if (!ids.length) return { chunks: [] };
    if (ids.length > TEXT_MAX) throw new Error(`Text too long: ${ids.length} tokens`);
    const { g, lm, codec } = this;
    const maxFrames = Math.min(
      F_MAX,
      S_MAX - PROMPT - ids.length - 1,
      Math.ceil((ids.length / 3 + 2) * FRAMES_PER_SECOND)
    );
    const after = words <= SHORT_TEXT_WORDS ? FRAMES_AFTER_EOS_SHORT : FRAMES_AFTER_EOS;
    let enc = g.encoder();
    codec.reset(enc);
    lm.start(enc, ids, kv, seed, TEMPERATURE, after);
    enc.submit();
    let issued = 1;
    let decoded = 0;
    let made = 0;
    let rowsCopied = 0;
    let end = null;
    const chunks = [];
    const free = [...this.stages];
    const inflight = [];
    const issue = () => {
      const stage = free.pop();
      enc = g.encoder();
      const size = CHUNKS[Math.min(made, CHUNKS.length - 1)];
      const target = Math.min(maxFrames, end ?? Math.max(issued, decoded + size));
      const steps = [];
      for (; issued < target; issued++) steps.push(...lm.step);
      enc.run(steps);
      const avail = end ?? issued;
      let f0 = -1;
      if (decoded < avail && (decoded + size <= avail || end !== null)) {
        f0 = decoded;
        codec.encodeChunk(enc, f0, size);
        decoded += size;
        made++;
      }
      enc.read(g.state, 0, stage, 0, 16);
      if (f0 >= 0) enc.read(codec.audio, 0, stage, AUDIO_AT, size * FRAME * 4);
      const r0 = rowsCopied;
      if (issued > r0) {
        if (issued - r0 > ROWS) throw new Error(`${issued - r0} alignment rows in one submit`);
        const bytes = TEXT_MAX * 4;
        enc.read(lm.align, r0 * bytes, stage, this.rowsAt, (issued - r0) * bytes);
        rowsCopied = issued;
      }
      enc.submit();
      inflight.push({ stage, f0, size, n: issued, r0, ready: stage.map() });
    };
    const drain = async () => {
      const job = inflight.shift();
      await job.ready;
      const m = job.stage.range();
      const keep = new Uint32Array(m, 0, 4)[STATE.end];
      if (end === null && keep) end = Math.min(keep, maxFrames);
      if (end === null && job.n >= maxFrames) end = maxFrames;
      const r1 = Math.min(job.n, end ?? Infinity);
      if (onRows && r1 > job.r0) {
        const n = ids.length;
        const rows = new Float32Array(m, this.rowsAt, (job.n - job.r0) * TEXT_MAX);
        const out = new Float32Array((r1 - job.r0) * n);
        for (let r = 0; r < r1 - job.r0; r++) out.set(rows.subarray(r * TEXT_MAX, r * TEXT_MAX + n), r * n);
        onRows(out);
      }
      if (job.f0 >= 0 && (end === null || job.f0 < end)) {
        const frames = Math.min(job.size, (end ?? Infinity) - job.f0);
        const chunk = new Float32Array(m, AUDIO_AT, frames * FRAME).slice();
        chunks.push(chunk);
        onChunk?.(chunk);
      }
      job.stage.unmap();
      free.push(job.stage);
    };
    const finished = () => end !== null && decoded >= end && issued >= end;
    for (; ; ) {
      while (!cancelled2() && inflight.length < g.depth && !finished()) issue();
      if (!inflight.length) break;
      await drain();
    }
    if (cancelled2()) return null;
    return { chunks };
  }
};

// src/cpu-worker.ts
var scope = self;
var workers = [];
async function startPool(pool, url, n, stacks) {
  workers.push(...Array.from({ length: n - 1 }, () => new Worker(url, { type: "module" })));
  const up = await Promise.all(
    workers.map(
      (w, k) => new Promise((done) => {
        const timer = setTimeout(() => done(false), 1e4);
        w.onmessage = () => {
          clearTimeout(timer);
          done(true);
        };
        w.onerror = () => {
          clearTimeout(timer);
          done(false);
        };
        w.postMessage({ role: "pool", ...pool, index: k + 1, stack: stacks + (k + 1) * STACK });
      })
    )
  );
  const count = up.indexOf(false) < 0 ? up.length : up.indexOf(false);
  workers.splice(count).forEach((w) => w.terminate());
  return 1 + count;
}
async function tile(g, key) {
  const known = await (await lookup(key))?.json().catch(() => null);
  if (known && TILES.some((t) => JSON.stringify(t) === JSON.stringify(known))) return known;
  const best = g.fastestTile();
  await store(key, JSON.stringify(best));
  return best;
}
var engine = null;
var graph = null;
var cancelled = /* @__PURE__ */ new Set();
async function load(m) {
  const level = simdLevel();
  if (!level) throw new Error("This browser has no WebAssembly SIMD.");
  const shared = scope.crossOriginIsolated && m.threads > 1;
  const url = new URL(`./phonon-cpu-${level}.wasm`, import.meta.url);
  const model = unpackModel(m.packed);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`The CPU engine's module could not download (${response.status}).`);
  const wasm = new Uint8Array(await response.arrayBuffer());
  const { module, memory, exports: x } = await instantiate(wasm, shared, model.blob.byteLength + 64 * 2 ** 20);
  const base = Math.ceil(x.__heap_base.value / 64) * 64;
  const threads = shared ? await startPool({ module, memory }, import.meta.url, m.threads, base) : 1;
  const g = new CpuGraph(x, model, { threads, heap: base + (shared ? m.threads - 1 : 0) * STACK });
  g.tune({ tile: await tile(g, `${url}?bytes=${wasm.byteLength}&threads=${threads}`), gemv: "int16" });
  graph = g;
  engine = new PhononEngine(g, withoutWeights(model), m.align);
  await engine.ready();
  return { threads, simd: level, memory: memory.buffer.byteLength };
}
scope.onmessage = async (e) => {
  const m = e.data;
  if ("role" in m) {
    const instance = await WebAssembly.instantiate(m.module, { env: { memory: m.memory } });
    const x = instance.exports;
    x.__stack_pointer.value = m.stack;
    scope.postMessage({ up: true });
    x.worker(m.index);
    return;
  }
  if (m.type === "load") {
    try {
      scope.postMessage({ type: "loaded", ...await load(m) });
    } catch (err) {
      workers.splice(0).forEach((w) => w.terminate());
      scope.postMessage({ type: "error", message: String(err?.message ?? err) });
    }
    return;
  }
  if (m.type === "close") {
    workers.splice(0).forEach((w) => w.terminate());
    scope.postMessage({ type: "closed" });
    return;
  }
  if (m.type === "cancel") {
    cancelled.add(m.id);
    return;
  }
  if (m.type === "speak") {
    const { id } = m;
    try {
      const out = await engine.synthesize(m.ids, m.voice, {
        ...m.options,
        latents: m.latents,
        cancelled: () => cancelled.has(id),
        onRows: (rows) => scope.postMessage({ type: "rows", id, rows }, [rows.buffer]),
        onChunk: (samples) => scope.postMessage({ type: "chunk", id, samples }, [samples.buffer])
      });
      if (false) {
        console.log(JSON.stringify([...graph.times].sort((a, b) => b[1] - a[1]).map(([k, v]) => [k, +v.toFixed(2)])));
        graph.times.clear();
      }
      scope.postMessage({ type: "done", id, cancelled: !out });
    } catch (err) {
      scope.postMessage({ type: "error", id, message: String(err?.message ?? err) });
    }
    cancelled.delete(id);
  }
};
