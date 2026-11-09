// The page's side of the CPU backend: an engine (engine.ts Engine) whose work runs in a worker
// (cpu-worker.ts), so speech never blocks the page. The worker's code and module load only when
// a page uses the CPU.
import type { Engine, Synthesis, SynthesizeOptions } from "./engine";
import type { Align } from "./files";

export type ToWorker =
  | { type: "load"; packed: ArrayBuffer; align: Align; threads: number }
  | {
      type: "speak";
      id: number;
      ids: number[];
      voice: string;
      options: { seed?: number; pace?: number; words?: number };
      latents?: Float32Array<ArrayBuffer>;
    }
  | { type: "cancel"; id: number }
  | { type: "close" };

type FromWorker =
  | { type: "loaded"; threads: number; simd: "relaxed" | "simd"; memory: number }
  | { type: "closed" }
  | { type: "rows"; id: number; rows: Float32Array }
  | { type: "chunk"; id: number; samples: Float32Array }
  | { type: "done"; id: number; cancelled: boolean }
  | { type: "error"; id?: number; message: string };

interface Job {
  options: SynthesizeOptions;
  chunks: Float32Array[];
  done: (s: Synthesis | null) => void;
  fail: (e: Error) => void;
  cancelSent: boolean;
}

export class CpuEngine implements Engine {
  /** Threads that take part in each op, this worker's included. */
  readonly threads: number;
  /** The WebAssembly build: "relaxed" (relaxed SIMD) or "simd". */
  readonly simd: "relaxed" | "simd";
  private readonly worker: Worker;
  private readonly voices = new Set<string>();
  private readonly jobs = new Map<number, Job>();
  private next = 1;

  private constructor(worker: Worker, info: { threads: number; simd: "relaxed" | "simd" }) {
    this.worker = worker;
    this.threads = info.threads;
    this.simd = info.simd;
    worker.onmessage = (e: MessageEvent<FromWorker>) => this.hear(e.data);
    // A worker that dies (out of memory) fails every text in flight, and later ones at once.
    let lose: (e: Error) => void = () => {};
    this.lost = new Promise((settle) => (lose = settle));
    worker.onerror = (e) => {
      this.dead = new Error(e.message || "The CPU engine's worker stopped.");
      for (const job of this.jobs.values()) job.fail(this.dead);
      this.jobs.clear();
      lose(this.dead);
    };
  }
  private dead: Error | null = null;
  /** Settles when the worker stops: load again to go on. */
  readonly lost: Promise<Error>;

  /** Starts the worker with a model's packed weights (moved to it, not copied). */
  static async start(packed: ArrayBuffer, align: Align, threads: number): Promise<CpuEngine> {
    const worker = new Worker(new URL("./phonon-cpu.js", import.meta.url), { type: "module" });
    const info = await new Promise<{ threads: number; simd: "relaxed" | "simd" }>((resolve, reject) => {
      worker.onmessage = (e: MessageEvent<FromWorker>) => {
        if (e.data.type === "loaded") resolve(e.data);
        else if (e.data.type === "error") reject(new Error(e.data.message));
      };
      worker.onerror = (e) => reject(new Error(e.message || "The CPU engine's worker failed to start."));
      const m: ToWorker = { type: "load", packed, align, threads };
      worker.postMessage(m, [packed]);
    }).catch((error) => {
      // Its pool threads end with it.
      worker.terminate();
      throw error;
    });
    return new CpuEngine(worker, info);
  }

  /** Ends the worker and its pool threads. Texts in progress fail, and so do later ones. */
  async close() {
    if (this.dead) return;
    this.dead = new Error("This model is closed.");
    for (const job of this.jobs.values()) job.fail(this.dead);
    this.jobs.clear();
    // The worker ends its pool threads first, then this ends it (and them, if it did not answer).
    await new Promise<void>((done) => {
      const timer = setTimeout(done, 1000);
      this.worker.onmessage = (e: MessageEvent<FromWorker>) => {
        if (e.data.type !== "closed") return;
        clearTimeout(timer);
        done();
      };
      this.worker.postMessage({ type: "close" } satisfies ToWorker);
    });
    this.worker.terminate();
  }

  hasVoice(name: string) {
    return this.voices.has(name);
  }

  synthesize(ids: number[], voice: string, options: SynthesizeOptions = {}): Promise<Synthesis | null> {
    if (this.dead) return Promise.reject(this.dead);
    const id = this.next++;
    const { seed, pace, words, latents } = options;
    const m: ToWorker = { type: "speak", id, ids, voice, options: { seed, pace, words } };
    if (!this.voices.has(voice) && latents) {
      m.latents = latents.slice();
      this.voices.add(voice);
    }
    return new Promise((done, fail) => {
      this.jobs.set(id, { options, chunks: [], done, fail, cancelSent: false });
      this.worker.postMessage(m, m.latents ? [m.latents.buffer] : []);
    });
  }

  private hear(m: FromWorker) {
    if (m.type === "loaded" || m.type === "closed" || m.id === undefined) return;
    const job = this.jobs.get(m.id);
    if (!job) return;
    if (m.type === "error") {
      this.jobs.delete(m.id);
      job.fail(new Error(m.message));
      return;
    }
    if (m.type === "done") {
      this.jobs.delete(m.id);
      job.done(m.cancelled ? null : { chunks: job.chunks });
      return;
    }
    if (job.options.cancelled?.()) {
      if (!job.cancelSent) this.worker.postMessage({ type: "cancel", id: m.id } satisfies ToWorker);
      job.cancelSent = true;
      return;
    }
    if (m.type === "rows") job.options.onRows?.(m.rows);
    else {
      job.chunks.push(m.samples);
      job.options.onChunk?.(m.samples);
    }
  }
}
