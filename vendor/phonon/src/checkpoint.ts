// A Gradium Phonon checkpoint (one safetensors file: the flow LM in bf16,
// the codec's decoder in f32) turned into the engine's weight blob: matrices
// of Q8_LEAST values or more in int8 with one f16 scale per Q8_GROUP along
// each row, everything else in f16. Linear
// weights stay [out][in]. Conv weights become [out][tap][in]. Transposed
// convs (kernel = 2 * stride) become [out][phase][tap][in], so a shader
// gathers instead of scattering. q and k rows move from interleaved rotary
// pairs to halves, as the kernels rotate. Constants the flow head adds every
// step (its time embedding for one LSD step from 0 to 1) are computed here.
import { paceVectors, type PaceWeights } from "./pace";
import { Q8_GROUP } from "./graph";

export interface Tensor {
  /** Byte offset of its values: f16, or int8 when it has scales. */
  offset: number;
  count: number;
  /** Byte offset of its f16 scales, one per Q8_GROUP values. */
  scales?: number;
}
/** Tensors this large are int8: every matrix the GEMVs read and most convolutions. */
const Q8_LEAST = 16384;
export interface PhononModel {
  /** The weights, each tensor's values and scales 256-byte aligned. */
  blob: Uint8Array<ArrayBuffer>;
  tensors: Record<string, Tensor>;
  /** What setPace needs: the condition the LM adds for a pace. */
  pace: PaceWeights;
}
/** The LM's layers: 6 in Phonon 31M, 9 in the 40M models. */
export const lmLayers = (m: PhononModel) => Object.keys(m.tensors).filter((n) => /^l\d+\.qkv$/.test(n)).length;
const H = 512;

/** The safetensors tensors of `buffer`, read as f32. */
export function readSafetensors(
  buffer: ArrayBuffer,
): Map<string, { shape: number[]; data: Float32Array }> {
  const view = new DataView(buffer);
  const headerSize = Number(view.getBigUint64(0, true));
  const header = JSON.parse(
    new TextDecoder().decode(new Uint8Array(buffer, 8, headerSize)),
  ) as Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>;
  const base = 8 + headerSize;
  const out = new Map<string, { shape: number[]; data: Float32Array }>();
  for (const [name, t] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const [start, end] = t.data_offsets;
    const bytes = buffer.slice(base + start, base + end);
    let data: Float32Array;
    if (t.dtype === "F32") data = new Float32Array(bytes);
    else if (t.dtype === "BF16") {
      const src = new Uint16Array(bytes);
      const bits = new Uint32Array(src.length);
      for (let i = 0; i < src.length; i++) bits[i] = src[i] << 16;
      data = new Float32Array(bits.buffer);
    } else throw new Error(`Unexpected tensor type ${t.dtype} for ${name}`);
    out.set(name, { shape: t.shape, data });
  }
  return out;
}

/** f32 to f16 bits, rounding to nearest even. */
export function toF16(values: Float32Array): Uint16Array<ArrayBuffer> {
  const f = new Float32Array(1);
  const u = new Uint32Array(f.buffer);
  const out = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) {
    f[0] = values[i];
    const x = u[0];
    const sign = (x >>> 16) & 0x8000;
    const exp = (x >>> 23) & 0xff;
    let mant = x & 0x7fffff;
    if (exp === 0xff) {
      out[i] = sign | 0x7c00 | (mant ? 0x200 : 0);
      continue;
    }
    const e = exp - 127 + 15;
    if (e >= 31) {
      out[i] = sign | 0x7c00;
      continue;
    }
    if (e <= 0) {
      if (e < -10) {
        out[i] = sign;
        continue;
      }
      mant |= 0x800000;
      const shift = 14 - e;
      let half = mant >> shift;
      const rest = mant & ((1 << shift) - 1);
      const mid = 1 << (shift - 1);
      if (rest > mid || (rest === mid && half & 1)) half++;
      out[i] = sign | half;
      continue;
    }
    let half = (e << 10) | (mant >> 13);
    const rest = mant & 0x1fff;
    if (rest > 0x1000 || (rest === 0x1000 && half & 1)) half++;
    out[i] = sign | half;
  }
  return out;
}

/** Every f16 bit pattern as f32. */
let halfTable: Float32Array | null = null;
function halves(): Float32Array {
  if (halfTable) return halfTable;
  halfTable = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    const v =
      e === 0
        ? m * 2 ** -24
        : e === 31
          ? m
            ? Number.NaN
            : Infinity
          : (1 + m / 1024) * 2 ** (e - 15);
    halfTable[h] = h & 0x8000 ? -v : v;
  }
  return halfTable;
}

/** Symmetric int8 with one f16 scale per group, from the values rounded to f16 first:
 *  scale = max |v| / 127, q = v / scale rounded half to even. */
export function quantize(values: Float32Array): { q: Int8Array; scales: Uint16Array } {
  const table = halves();
  const bits = toF16(values);
  const q = new Int8Array(values.length);
  const scales = new Uint16Array(values.length / Q8_GROUP);
  const max = new Float32Array(scales.length);
  for (let g = 0; g < max.length; g++)
    for (let i = g * Q8_GROUP; i < (g + 1) * Q8_GROUP; i++)
      max[g] = Math.max(max[g], Math.abs(table[bits[i]]));
  for (let g = 0; g < max.length; g++) max[g] = Math.fround(max[g] / 127) || 1;
  const scaleBits = toF16(max);
  for (let g = 0; g < scales.length; g++) {
    const sb = scaleBits[g] & 0x7fff ? scaleBits[g] : 0x3c00;
    scales[g] = sb;
    const s = table[sb];
    for (let i = g * Q8_GROUP; i < (g + 1) * Q8_GROUP; i++) {
      const x = Math.fround(table[bits[i]] / s);
      const r = Math.round(x);
      q[i] = Math.max(-127, Math.min(127, r - x === 0.5 && r % 2 !== 0 ? r - 1 : r));
    }
  }
  return { q, scales };
}

/** y = W x (+ b) for W [rows][cols]. */
function matvec(w: Float32Array, x: Float32Array, rows: number, b?: Float32Array): Float32Array {
  const cols = x.length;
  const y = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = b ? b[r] : 0;
    for (let c = 0; c < cols; c++) s += w[r * cols + c] * x[c];
    y[r] = s;
  }
  return y;
}
const silu = (v: Float32Array) => v.map((x) => x / (1 + Math.exp(-x)));

export function convertCheckpoint(buffer: ArrayBuffer): PhononModel {
  const src = readSafetensors(buffer);
  const get = (name: string) => {
    const t = src.get(name);
    if (!t) throw new Error(`The checkpoint has no ${name}`);
    return t.data;
  };
  const tensors: Record<string, Float32Array> = {};
  const put = (name: string, data: Float32Array) => (tensors[name] = data);
  const cat = (...parts: Float32Array[]) => {
    const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
    let o = 0;
    for (const p of parts) out.set(p, (o += p.length) - p.length);
    return out;
  };
  // Interleaved rotary pairs (2j, 2j+1) become halves (j, j + 32), per head.
  const rotaryRows = (w: Float32Array, cols: number) => {
    const out = new Float32Array(w.length);
    for (let h = 0; h < 8; h++)
      for (let j = 0; j < 32; j++)
        for (const [from, to] of [
          [2 * j, j],
          [2 * j + 1, j + 32],
        ])
          out.set(
            w.subarray((h * 64 + from) * cols, (h * 64 + from + 1) * cols),
            (h * 64 + to) * cols,
          );
    return out;
  };
  const qkv = (w: Float32Array) =>
    cat(
      rotaryRows(w.subarray(0, H * H), H),
      rotaryRows(w.subarray(H * H, 2 * H * H), H),
      w.subarray(2 * H * H),
    );
  const scaleRows = (w: Float32Array, scale: Float32Array) => {
    const cols = w.length / scale.length;
    return w.map((v, i) => v * scale[Math.floor(i / cols)]);
  };

  const C = "flow_lm.condition_provider.conditioners.";
  const pace: PaceWeights = {
    speakers: get(`${C}num_speakers.embed.weight`).slice(16, 32),
    speakersProj: get(`${C}num_speakers.output_proj.weight`),
    durationProj: get(`${C}duration_delta.output_proj.weight`),
    latentProj: get("flow_lm.input_linear.weight"),
    bos: get("flow_lm.bos_emb"),
  };
  const start = paceVectors(pace, 0);
  put("embed", get(`${C}transcript_in_segment.embed.weight`));
  put("prompt_proj", get(`${C}speaker_wavs.output_proj.weight`));
  put("latent_proj", pace.latentProj);
  put("cond", start.cond);
  put("bos_embed", start.bos);
  for (let i = 0; src.has(`flow_lm.transformer.layers.${i}.norm1.weight`); i++) {
    const p = `flow_lm.transformer.layers.${i}.`;
    put(`l${i}.ln1.w`, get(p + "norm1.weight"));
    put(`l${i}.ln1.b`, get(p + "norm1.bias"));
    put(`l${i}.qkv`, qkv(get(p + "self_attn.in_proj.weight")));
    put(`l${i}.o`, get(p + "self_attn.out_proj.weight"));
    put(`l${i}.ln2.w`, get(p + "norm2.weight"));
    put(`l${i}.ln2.b`, get(p + "norm2.bias"));
    put(`l${i}.fc1`, get(p + "linear1.weight"));
    put(`l${i}.fc2`, get(p + "linear2.weight"));
  }
  put("norm.w", get("flow_lm.out_norm.weight"));
  put("norm.b", get("flow_lm.out_norm.bias"));
  put("eos.w", get("flow_lm.out_eos.weight"));
  put("eos.b", get("flow_lm.out_eos.bias"));
  put("lat_mean", get("flow_lm.emb_mean"));
  put("lat_std", get("flow_lm.emb_std"));

  // Flow head. One LSD step conditions on times 0 and 1: their embeddings'
  // mean is a constant, added to the condition projection's bias.
  const F = "flow_lm.flow_net.";
  const timeEmbedding = (k: number, t: number) => {
    const q = `${F}time_embed.${k}.`;
    const freqs = get(q + "freqs");
    const e = new Float32Array(2 * freqs.length);
    freqs.forEach((f, i) => {
      e[i] = Math.cos(t * f);
      e[freqs.length + i] = Math.sin(t * f);
    });
    const hdn = silu(matvec(get(q + "mlp.0.weight"), e, H, get(q + "mlp.0.bias")));
    const out = matvec(get(q + "mlp.2.weight"), hdn, H, get(q + "mlp.2.bias"));
    const ms = out.reduce((s, v) => s + v * v, 0) / H;
    const alpha = get(q + "mlp.3.alpha");
    return out.map((v, i) => (v / Math.sqrt(ms + 1e-5)) * alpha[i]);
  };
  const t0 = timeEmbedding(0, 0);
  const t1 = timeEmbedding(1, 1);
  const condBias = get(F + "cond_embed.bias").map((v, i) => v + (t0[i] + t1[i]) / 2);
  put("fh.in.w", get(F + "input_proj.weight"));
  put("fh.in.b", get(F + "input_proj.bias"));
  put("fh.cond.w", get(F + "cond_embed.weight"));
  put("fh.cond.b", condBias);
  const blocks = [0, 1, 2, 3, 4, 5].map((i) => `${F}res_blocks.${i}.`);
  put(
    "fh.ada.w",
    cat(
      ...blocks.map((b) => get(b + "adaLN_modulation.1.weight")),
      get(F + "final_layer.adaLN_modulation.1.weight"),
    ),
  );
  put(
    "fh.ada.b",
    cat(
      ...blocks.map((b) => get(b + "adaLN_modulation.1.bias")),
      get(F + "final_layer.adaLN_modulation.1.bias"),
    ),
  );
  blocks.forEach((b, i) => {
    put(`fh${i}.ln.w`, get(b + "in_ln.weight"));
    put(`fh${i}.ln.b`, get(b + "in_ln.bias"));
    put(`fh${i}.fc1.w`, get(b + "mlp.0.weight"));
    put(`fh${i}.fc1.b`, get(b + "mlp.0.bias"));
    put(`fh${i}.fc2.w`, get(b + "mlp.2.weight"));
    put(`fh${i}.fc2.b`, get(b + "mlp.2.bias"));
  });
  put("fh.out.w", get(F + "final_layer.linear.weight"));
  put("fh.out.b", get(F + "final_layer.linear.bias"));

  // Codec decoder. Its latent projection (1x1, no bias) merges into the
  // depthwise upsample: W[c][phase][tap][i] = up[c][tap * 16 + phase] * P[c][i].
  const P = get("mimi.quantizer.output_proj.weight");
  const up = get("mimi.upsample.convtr.convtr.weight");
  const W = new Float32Array(512 * 16 * 2 * 32);
  for (let c = 0; c < 512; c++)
    for (let ph = 0; ph < 16; ph++)
      for (let j = 0; j < 2; j++)
        for (let i = 0; i < 32; i++)
          W[((c * 16 + ph) * 2 + j) * 32 + i] = up[c * 32 + j * 16 + ph] * P[c * 32 + i];
  put("up", W);
  for (let i = 0; i < 2; i++) {
    const p = `mimi.decoder_transformer.transformer.layers.${i}.`;
    put(`d${i}.ln1.w`, get(p + "norm1.weight"));
    put(`d${i}.ln1.b`, get(p + "norm1.bias"));
    put(`d${i}.qkv`, qkv(get(p + "self_attn.in_proj.weight")));
    put(`d${i}.o`, scaleRows(get(p + "self_attn.out_proj.weight"), get(p + "layer_scale_1.scale")));
    put(`d${i}.ln2.w`, get(p + "norm2.weight"));
    put(`d${i}.ln2.b`, get(p + "norm2.bias"));
    put(`d${i}.fc1`, get(p + "linear1.weight"));
    put(`d${i}.fc2`, scaleRows(get(p + "linear2.weight"), get(p + "layer_scale_2.scale")));
  }
  const conv = (name: string, key: string) => {
    const w = get(key + ".conv.weight");
    const [cout, cin, k] = src.get(key + ".conv.weight")!.shape;
    const out = new Float32Array(w.length);
    for (let o = 0; o < cout; o++)
      for (let i = 0; i < cin; i++)
        for (let t = 0; t < k; t++) out[(o * k + t) * cin + i] = w[(o * cin + i) * k + t];
    put(name + ".w", out);
    put(name + ".b", get(key + ".conv.bias"));
  };
  const convT = (name: string, key: string, stride: number) => {
    const w = get(key + ".convtr.weight");
    const [cin, cout] = src.get(key + ".convtr.weight")!.shape;
    const out = new Float32Array(w.length);
    for (let o = 0; o < cout; o++)
      for (let ph = 0; ph < stride; ph++)
        for (let j = 0; j < 2; j++)
          for (let i = 0; i < cin; i++)
            out[((o * stride + ph) * 2 + j) * cin + i] =
              w[(i * cout + o) * 2 * stride + j * stride + ph];
    put(name + ".w", out);
    put(name + ".b", get(key + ".convtr.bias"));
  };
  const L = "mimi.decoder.model.";
  conv("s0", L + "0");
  convT("s1", L + "2", 6);
  conv("s1r1", L + "3.block.1");
  conv("s1r2", L + "3.block.3");
  convT("s2", L + "5", 5);
  conv("s2r1", L + "6.block.1");
  conv("s2r2", L + "6.block.3");
  convT("s3", L + "8", 4);
  conv("s3r1", L + "9.block.1");
  conv("s3r2", L + "9.block.3");
  conv("s4", L + "11");

  const parts: Uint8Array[] = [];
  let size = 0;
  const place = (bytes: Uint8Array) => {
    const at = size;
    parts.push(bytes);
    size += Math.ceil(bytes.byteLength / 256) * 256;
    return at;
  };
  const layout: Record<string, Tensor> = {};
  for (const [name, data] of Object.entries(tensors)) {
    if (data.length < Q8_LEAST) {
      layout[name] = { offset: place(new Uint8Array(toF16(data).buffer)), count: data.length };
      continue;
    }
    const { q, scales } = quantize(data);
    const offset = place(new Uint8Array(q.buffer));
    layout[name] = { offset, count: data.length, scales: place(new Uint8Array(scales.buffer)) };
  }
  const blob = new Uint8Array(size);
  let at = 0;
  for (const p of parts) {
    blob.set(p, at);
    at += Math.ceil(p.byteLength / 256) * 256;
  }
  return { blob, tensors: layout, pace };
}

/** A model as one buffer, for the cache: a JSON header (the tensors, the pace vectors'
 *  lengths), the pace vectors and the blob. */
export function packModel(model: PhononModel): Uint8Array<ArrayBuffer> {
  const pace = Object.entries(model.pace) as [keyof PaceWeights, Float32Array][];
  const header = new TextEncoder().encode(
    JSON.stringify({ tensors: model.tensors, pace: pace.map(([k, v]) => [k, v.length]) }),
  );
  const at = Math.ceil((4 + header.length) / 4) * 4;
  const floats = pace.reduce((n, [, v]) => n + v.length, 0);
  const out = new Uint8Array(at + floats * 4 + model.blob.byteLength);
  new DataView(out.buffer).setUint32(0, header.length, true);
  out.set(header, 4);
  let o = at;
  for (const [, v] of pace) {
    out.set(new Uint8Array(v.buffer, v.byteOffset, v.byteLength), o);
    o += v.byteLength;
  }
  out.set(model.blob, o);
  return out;
}
export function unpackModel(buffer: ArrayBuffer): PhononModel {
  const length = new DataView(buffer).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, length))) as {
    tensors: Record<string, Tensor>;
    pace: [keyof PaceWeights, number][];
  };
  let o = Math.ceil((4 + length) / 4) * 4;
  const pace = {} as PaceWeights;
  for (const [k, n] of header.pace) {
    pace[k] = new Float32Array(buffer, o, n);
    o += n * 4;
  }
  return { tensors: header.tensors, pace, blob: new Uint8Array(buffer, o) };
}

/** A model without its weights, once a backend holds them: the tensors' places and copies of the
 *  pace's vectors, which are otherwise views into the whole packed file. */
export function withoutWeights(m: PhononModel): PhononModel {
  const pace = Object.fromEntries(Object.entries(m.pace).map(([k, v]) => [k, v.slice()])) as PhononModel["pace"];
  return { tensors: m.tensors, pace, blob: new Uint8Array() };
}

/** A catalog voice file: its speaker latents as [125 frames][512]. */
export function readVoice(buffer: ArrayBuffer): Float32Array<ArrayBuffer> {
  const t = readSafetensors(buffer).get("speaker_wavs");
  if (!t) throw new Error("The voice file has no speaker_wavs");
  const [, channels, frames] = t.shape;
  const out = new Float32Array(frames * channels);
  for (let c = 0; c < channels; c++)
    for (let f = 0; f < frames; f++) out[f * channels + c] = t.data[c * frames + f];
  return out;
}
