// The WebGPU backend (graph.ts): each op is a pipeline, its bindings and a dispatch over one
// weight buffer. Everything an engine runs is built once, when it starts, so no request
// allocates. Pipelines compile in the background (`ready`), each with a layout read from its
// shader's own declarations, so bind groups exist before the compile ends.
import * as K from "./kernels";
import type { Tensor } from "./checkpoint";
import type * as G from "./graph";

// WebGPU's flag values: TypeScript's DOM types lack the constants' globals.
const STORAGE = 0x80;
const UNIFORM = 0x40;
const COPY_DST = 0x08;
const COPY_SRC = 0x04;
const MAP_READ = 0x01;
const COMPUTE = 0x4;
/** Largest split-K partials: 8-frame s2 and s3 each need 7.5 MiB. */
const SCRATCH = 7.5 * 2 ** 20;

interface Pipe {
  layout: GPUBindGroupLayout;
  pipeline: GPUComputePipeline | null;
}
export interface Op {
  pipe: Pipe;
  bindGroup: GPUBindGroup;
  groups: [number, number, number];
}
export type Binding = GPUBuffer | GPUBufferBinding;

export function encode(enc: GPUCommandEncoder, ops: Op[]) {
  const pass = enc.beginComputePass();
  for (const o of ops) {
    pass.setPipeline(o.pipe.pipeline!);
    pass.setBindGroup(0, o.bindGroup);
    pass.dispatchWorkgroups(...o.groups);
  }
  pass.end();
}

/** A shader's bind group layout, from its `@binding(i) var<storage|uniform, ...>` lines. */
function layoutOf(device: GPUDevice, code: string): GPUBindGroupLayout {
  const entries = [
    ...code.matchAll(/@binding\((\d+)\) var<(storage|uniform)(?:, (read_write|read))?>/g),
  ].map(([, b, kind, access]): GPUBindGroupLayoutEntry => ({
    binding: Number(b),
    visibility: COMPUTE,
    buffer: {
      type:
        kind === "uniform" ? "uniform" : access === "read_write" ? "storage" : "read-only-storage",
    },
  }));
  return device.createBindGroupLayout({ entries });
}

/** A residual into the output itself reads the output (one buffer cannot be bound twice). */
function residual(res: G.Residual, rbuf: GPUBuffer | undefined, ybuf: GPUBuffer) {
  return rbuf === ybuf ? { res: `${res}_y` as G.Residual, rbuf: undefined } : { res, rbuf };
}

class Staging implements G.Staging {
  readonly buf: GPUBuffer;
  private ready: Promise<void> | null = null;
  constructor(device: GPUDevice, bytes: number) {
    this.buf = device.createBuffer({ size: bytes, usage: COPY_DST | MAP_READ });
  }
  map() {
    return (this.ready ??= this.buf.mapAsync(MAP_READ));
  }
  range() {
    return this.buf.getMappedRange();
  }
  unmap() {
    this.buf.unmap();
    this.ready = null;
  }
}

class Encoder implements G.Encoder<GPUBuffer, Op> {
  private readonly device: GPUDevice;
  private readonly enc: GPUCommandEncoder;
  constructor(device: GPUDevice) {
    this.device = device;
    this.enc = device.createCommandEncoder();
  }
  run(ops: Op[]) {
    if (ops.length) encode(this.enc, ops);
  }
  copy(src: GPUBuffer, so: number, dst: GPUBuffer, dO: number, bytes: number) {
    this.enc.copyBufferToBuffer(src, so, dst, dO, bytes);
  }
  clear(buf: GPUBuffer, offset: number, bytes: number) {
    this.enc.clearBuffer(buf, offset, bytes);
  }
  read(src: GPUBuffer, so: number, dst: G.Staging, dO: number, bytes: number) {
    this.enc.copyBufferToBuffer(src, so, (dst as Staging).buf, dO, bytes);
  }
  submit() {
    this.device.queue.submit([this.enc.finish()]);
  }
}

export class GpuGraph implements G.Graph<GPUBuffer, Op> {
  readonly device: GPUDevice;
  readonly weights: GPUBuffer;
  readonly state: GPUBuffer;
  readonly depth = 3;
  readonly kvBytes = 2;
  /** Bound where a kernel's binding goes unused. */
  readonly zero: GPUBuffer;
  /** 32-lane subgroups for the GEMV sums. */
  readonly sg: boolean;
  private readonly tensors: Record<string, Tensor>;
  private readonly pipes = new Map<string, Pipe>();
  private readonly compiling: Promise<void>[] = [];
  private scratchBuf: GPUBuffer | null = null;

  constructor(
    device: GPUDevice,
    model: { blob: Uint8Array<ArrayBuffer>; tensors: Record<string, Tensor> },
    sg: boolean,
  ) {
    this.device = device;
    this.tensors = model.tensors;
    this.sg = sg;
    this.weights = this.buffer(model.blob.byteLength, model.blob);
    this.state = this.buffer(32);
    this.zero = this.buffer(256);
  }

  buffer(size: number, data?: ArrayBufferView<ArrayBuffer>) {
    const b = this.device.createBuffer({
      size: Math.max(256, Math.ceil(size / 256) * 256),
      usage: STORAGE | COPY_DST | COPY_SRC,
    });
    if (data) this.write(b, 0, data);
    return b;
  }
  write(buf: GPUBuffer, offset: number, data: ArrayBufferView<ArrayBuffer>) {
    this.device.queue.writeBuffer(buf, offset, data.buffer, data.byteOffset, data.byteLength);
  }
  encoder() {
    return new Encoder(this.device);
  }
  staging(bytes: number) {
    return new Staging(this.device, bytes);
  }
  /** A uniform of a kernel's `fields` (kernels.ts UNIFORMS), 0 where `values` has none. */
  u<F extends K.Fields>(fields: F, values: Partial<Record<F[number], number>>) {
    const given: Partial<Record<string, number>> = values;
    const words = new Uint32Array(Math.max(4, Math.ceil(fields.length / 4) * 4));
    fields.forEach((f, i) => (words[i] = given[f] ?? 0));
    const b = this.device.createBuffer({ size: words.byteLength, usage: UNIFORM | COPY_DST });
    this.device.queue.writeBuffer(b, 0, words);
    return b;
  }
  private tensor(name: string): Tensor {
    const t = this.tensors[name];
    if (!t) throw new Error(`No tensor ${name}`);
    return t;
  }
  /** An f16 tensor's offset in halves. */
  half(name: string) {
    const t = this.tensor(name);
    if (t.scales !== undefined) throw new Error(`${name} is int8`);
    return t.offset / 2;
  }
  /** A tensor's offset in u32 words, and its scales' offset in halves (int8 only). */
  words(name: string) {
    const t = this.tensor(name);
    return {
      wo: t.offset / 4,
      wso: t.scales === undefined ? 0 : t.scales / 2,
      q8: t.scales !== undefined,
    };
  }
  /** An int8 tensor's offsets, for the kernels that only read int8. */
  q8(name: string) {
    const w = this.words(name);
    if (!w.q8) throw new Error(`${name} is not int8`);
    return w;
  }
  /** Settles when every pipeline built so far has compiled. */
  async ready() {
    while (this.compiling.length) await Promise.all(this.compiling.splice(0));
  }
  op(code: string, bindings: Binding[], groups: [number, number, number]): Op {
    let pipe = this.pipes.get(code);
    if (!pipe) {
      const layout = layoutOf(this.device, code);
      const p: Pipe = { layout, pipeline: null };
      this.compiling.push(
        this.device
          .createComputePipelineAsync({
            layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
            compute: { module: this.device.createShaderModule({ code }), entryPoint: "main" },
          })
          .then((pipeline) => {
            p.pipeline = pipeline;
          }),
      );
      this.pipes.set(code, p);
      pipe = p;
    }
    const bindGroup = this.device.createBindGroup({
      layout: pipe.layout,
      entries: bindings.map((b, i) => ({
        binding: i,
        resource: b instanceof GPUBuffer ? { buffer: b } : b,
      })),
    });
    return { pipe, bindGroup, groups };
  }
  /** Split-K partials. Dispatches run in order, so every GEMM's partials share one buffer. */
  private scratch(bytes: number): GPUBufferBinding {
    if (bytes > SCRATCH) throw new Error(`Split-K partials of ${bytes} bytes`);
    this.scratchBuf ??= this.buffer(SCRATCH);
    return { buffer: this.scratchBuf, offset: 0, size: Math.ceil(bytes / 256) * 256 };
  }

  gemv(x: GPUBuffer | null, wname: string, ybuf: GPUBuffer, o: G.Gemv<GPUBuffer>): Op {
    const { ln, combine } = o;
    const { res, rbuf } = residual(o.res ?? "none", o.rbuf, ybuf);
    const w = this.q8(wname);
    const code = K.gemv({
      K: o.K,
      N: o.N,
      act: o.act,
      bias: !!o.bias,
      res,
      combine: combine?.splits,
      ln: ln && { affine: !!ln.w, modulate: !!ln.mod, eps: ln.eps },
      sg: this.sg,
    });
    const u = this.u(K.UNIFORMS.gemv, {
      xo: o.xo,
      go: o.go,
      so: ln?.so,
      ho: ln?.ho,
      wo: w.wo,
      bo: o.bias ? this.half(o.bias) : 0,
      vo: o.vname ? this.half(o.vname) : 0,
      lgo: ln?.w ? this.half(ln.w) : 0,
      lbo: ln?.b ? this.half(ln.b) : 0,
      wso: w.wso,
    });
    return this.op(
      code,
      [
        x ?? this.zero,
        this.weights,
        ybuf,
        combine?.parts ?? rbuf ?? this.zero,
        u,
        o.gbuf ?? this.zero,
        ln?.mod ?? this.zero,
        this.state,
      ],
      [Math.ceil(o.N / K.gemvShape(o.K).rowsPerGroup), 1, 1],
    );
  }

  gemm(xbuf: GPUBuffer, wname: string, ybuf: GPUBuffer, o: G.Gemm<GPUBuffer>): Op[] {
    const { M, K: k, N: n, load = "linear", conv, yo = 0, ro = 0, gate } = o;
    const { res, rbuf } = residual(o.res ?? "none", o.rbuf, ybuf);
    const w = this.words(wname);
    const phases = load === "convT" ? conv!.stride! : 1;
    // Row tiles are capped at 4, so long inputs share the split counts of shorter ones.
    const tiles = Math.ceil(n / 64) * Math.min(4, Math.ceil(M / 64)) * phases;
    const splits = Math.max(1, Math.min(Math.floor(160 / tiles), Math.floor(k / 128)));
    const groups: [number, number, number] = [
      Math.ceil(n / 64),
      Math.ceil(M / 64),
      phases * splits,
    ];
    const bo = o.bias ? this.half(o.bias) : 0;
    const shape = { K: k, N: n, load, conv, q8: w.q8, gate };
    const P = K.UNIFORMS.gemm;
    if (splits === 1) {
      const code = K.gemm({ ...shape, act: o.act, bias: !!o.bias, res });
      return [
        this.op(
          code,
          [
            xbuf,
            this.weights,
            ybuf,
            rbuf ?? this.zero,
            this.u(P, { M, yo, ro, wo: w.wo, bo, wso: w.wso }),
            this.state,
          ],
          groups,
        ),
      ];
    }
    const rowsOut = M * phases;
    const part = this.scratch(splits * rowsOut * n * 4);
    return [
      this.op(
        K.gemm({ ...shape, splits }),
        [
          xbuf,
          this.weights,
          part,
          this.zero,
          this.u(P, { M, rows: rowsOut, wo: w.wo, wso: w.wso }),
          this.state,
        ],
        groups,
      ),
      this.op(
        K.gemmReduce({ N: n, splits, act: o.act, bias: !!o.bias, res, gate }),
        [
          part,
          ybuf,
          this.weights,
          rbuf ?? this.zero,
          this.u(K.UNIFORMS.gemmReduce, { rows: rowsOut, yo, ro, stride: rowsOut, bo }),
          this.state,
        ],
        [Math.ceil((rowsOut * n) / 256), 1, 1],
      ),
    ];
  }

  layernorm(x: GPUBuffer, y: GPUBuffer, rows: number, w: string, b: string, eps: string, gate: G.Gate): Op {
    return this.op(
      K.layernorm({ eps, gate }),
      [x, y, this.weights, this.u(K.UNIFORMS.layernorm, { go: this.half(w), bo: this.half(b) }), this.state],
      [rows, 1, 1],
    );
  }

  ropeKV(qkv: GPUBuffer, q: GPUBuffer, kc: GPUBuffer, vc: GPUBuffer, at: G.Rows, o: { theta: string; cache: G.Cache }): Op {
    return this.op(K.ropeKV({ theta: o.theta, cache: o.cache }), [qkv, q, kc, vc, this.state, this.u(K.UNIFORMS.rows, at)], [at.rows, 1, 1]);
  }

  attention(q: GPUBuffer, kc: GPUBuffer, vc: GPUBuffer, out: GPUBuffer, at: G.Rows, o: { cache: G.Cache }): Op {
    return this.op(
      K.attention({ cache: o.cache }),
      [q, kc, vc, out, this.state, this.u(K.UNIFORMS.rows, at)],
      [Math.ceil(at.rows / 16), 8, 1],
    );
  }

  attnStep(
    qkv: GPUBuffer,
    kc: GPUBuffer,
    vc: GPUBuffer,
    parts: GPUBuffer,
    o: { splits: number; theta: string; align?: { at: G.AlignRows; buf: GPUBuffer } },
  ): Op {
    const code = K.attnSplit({ splits: o.splits, theta: o.theta, sg: this.sg, align: o.align?.at });
    const bindings = [qkv, kc, vc, parts, this.state];
    return this.op(code, o.align ? [...bindings, o.align.buf] : bindings, [o.splits, 8, 1]);
  }

  alignRow(q: GPUBuffer, kc: GPUBuffer, al: GPUBuffer, at: G.AlignRows): Op {
    return this.op(K.alignRow(at), [q, kc, al, this.state], [1, 1, 1]);
  }

  embedRows(ids: GPUBuffer, x: GPUBuffer, rows: number, o: { bos: number; pad: number }): Op {
    const em = this.q8("embed");
    return this.op(
      K.embedRows(o),
      [ids, this.weights, x, this.u(K.UNIFORMS.embedRows, { eo: em.wo, eso: em.wso, bo: this.half("bos_embed") })],
      [(rows * 512) / 256, 1, 1],
    );
  }

  noise(z: GPUBuffer): Op {
    return this.op(K.noise(), [z, this.state], [1, 1, 1]);
  }

  finishStep(x: GPUBuffer, lat: GPUBuffer, lats: GPUBuffer, o: { eps: string; threshold: string }): Op {
    return this.op(
      K.finishStep(o),
      [
        x,
        this.weights,
        lat,
        lats,
        this.state,
        this.u(K.UNIFORMS.finishStep, {
          ewo: this.half("eos.w"),
          ebo: this.half("eos.b"),
          ngo: this.half("norm.w"),
          nbo: this.half("norm.b"),
        }),
      ],
      [1, 1, 1],
    );
  }

  upsample(lats: GPUBuffer, y: GPUBuffer, T: number): Op {
    const up = this.q8("up");
    return this.op(
      K.upsample(),
      [
        lats,
        this.weights,
        y,
        this.u(K.UNIFORMS.upsample, { T, wo: up.wo, wso: up.wso, mo: this.half("lat_mean"), so: this.half("lat_std") }),
        this.state,
      ],
      [(T * 512) / 256, 1, 1],
    );
  }

  shiftRows(stages: G.Stage<GPUBuffer>[], T: number): Op {
    return this.op(
      K.shiftRows(stages),
      [...stages.map((st) => st.buf), this.u(K.UNIFORMS.shiftRows, { T })],
      [Math.ceil(stages.reduce((n, st) => n + st.ctx * st.ch, 0) / 256), 1, 1],
    );
  }
}
