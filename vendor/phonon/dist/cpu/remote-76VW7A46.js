// src/remote.ts
var CpuEngine = class _CpuEngine {
  /** Threads that take part in each op, this worker's included. */
  threads;
  /** The WebAssembly build: "relaxed" (relaxed SIMD) or "simd". */
  simd;
  worker;
  voices = /* @__PURE__ */ new Set();
  jobs = /* @__PURE__ */ new Map();
  next = 1;
  constructor(worker, info) {
    this.worker = worker;
    this.threads = info.threads;
    this.simd = info.simd;
    worker.onmessage = (e) => this.hear(e.data);
    let lose = () => {
    };
    this.lost = new Promise((settle) => lose = settle);
    worker.onerror = (e) => {
      this.dead = new Error(e.message || "The CPU engine's worker stopped.");
      for (const job of this.jobs.values()) job.fail(this.dead);
      this.jobs.clear();
      lose(this.dead);
    };
  }
  dead = null;
  /** Settles when the worker stops: load again to go on. */
  lost;
  /** Starts the worker with a model's packed weights (moved to it, not copied). */
  static async start(packed, align, threads) {
    const worker = new Worker(new URL("./phonon-cpu.js", import.meta.url), { type: "module" });
    const info = await new Promise((resolve, reject) => {
      worker.onmessage = (e) => {
        if (e.data.type === "loaded") resolve(e.data);
        else if (e.data.type === "error") reject(new Error(e.data.message));
      };
      worker.onerror = (e) => reject(new Error(e.message || "The CPU engine's worker failed to start."));
      const m = { type: "load", packed, align, threads };
      worker.postMessage(m, [packed]);
    }).catch((error) => {
      worker.terminate();
      throw error;
    });
    return new _CpuEngine(worker, info);
  }
  /** Ends the worker and its pool threads. Texts in progress fail, and so do later ones. */
  async close() {
    if (this.dead) return;
    this.dead = new Error("This model is closed.");
    for (const job of this.jobs.values()) job.fail(this.dead);
    this.jobs.clear();
    await new Promise((done) => {
      const timer = setTimeout(done, 1e3);
      this.worker.onmessage = (e) => {
        if (e.data.type !== "closed") return;
        clearTimeout(timer);
        done();
      };
      this.worker.postMessage({ type: "close" });
    });
    this.worker.terminate();
  }
  hasVoice(name) {
    return this.voices.has(name);
  }
  synthesize(ids, voice, options = {}) {
    if (this.dead) return Promise.reject(this.dead);
    const id = this.next++;
    const { seed, pace, words, latents } = options;
    const m = { type: "speak", id, ids, voice, options: { seed, pace, words } };
    if (!this.voices.has(voice) && latents) {
      m.latents = latents.slice();
      this.voices.add(voice);
    }
    return new Promise((done, fail) => {
      this.jobs.set(id, { options, chunks: [], done, fail, cancelSent: false });
      this.worker.postMessage(m, m.latents ? [m.latents.buffer] : []);
    });
  }
  hear(m) {
    if (m.type === "loaded" || m.type === "closed" || m.id === void 0) return;
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
      if (!job.cancelSent) this.worker.postMessage({ type: "cancel", id: m.id });
      job.cancelSent = true;
      return;
    }
    if (m.type === "rows") job.options.onRows?.(m.rows);
    else {
      job.chunks.push(m.samples);
      job.options.onChunk?.(m.samples);
    }
  }
};
export {
  CpuEngine
};
