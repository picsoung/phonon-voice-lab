// Phonon in the browser on WebGPU: load a model once, then speak text in any
// voice of Gradium's catalog in the model's languages, with the time of every
// word. The model's files (files.ts) stay in the browser's cache once downloaded.
import { readVoice, unpackModel, withoutWeights } from "./checkpoint";
import { cached } from "./cache";
import {
  CATALOG_URL,
  DEFAULT_VOICE,
  MODELS,
  voiceUrl,
  type Align,
  type Language,
  type ModelFiles,
  type ModelId,
} from "./files";
import { PhononEngine, TEXT_MAX, type Engine } from "./engine";
import { GpuGraph } from "./gpu";
import { paceForSpeed } from "./pace";
import { SAMPLE_RATE, WordTimes, wordRanges, type Span, type Word } from "./timing";
import { Tokenizer } from "./tokenizer";

/** Where a model runs: WebGPU, or the CPU with its threads (each op's work shared among them) and
 *  its WebAssembly build ("relaxed" for relaxed SIMD). */
export type Backend = { name: "webgpu" } | { name: "cpu"; threads: number; simd: "relaxed" | "simd" };

/** A started engine on one backend, and how to end it. */
interface Runtime {
  engine: Engine;
  backend: Backend;
  /** Settles when the GPU device is lost (a reset), or the CPU's worker stops. */
  lost: Promise<unknown>;
  close(): Promise<void>;
}

async function startGpu(adapter: GPUAdapter, packed: ArrayBuffer, align: Align): Promise<Runtime> {
  const subgroups = subgroups32(adapter);
  const device = await adapter.requestDevice({ requiredFeatures: subgroups ? ["subgroups"] : [] });
  try {
    const model = unpackModel(packed);
    const engine = new PhononEngine(new GpuGraph(device, model, subgroups), withoutWeights(model), align);
    await engine.ready();
    return { engine, backend: { name: "webgpu" }, lost: device.lost, close: async () => device.destroy() };
  } catch (error) {
    device.destroy();
    throw error;
  }
}

async function startCpu(packed: ArrayBuffer, align: Align, threads: number): Promise<Runtime> {
  const { CpuEngine } = await import("./remote");
  const engine = await CpuEngine.start(packed, align, threads);
  return {
    engine,
    backend: { name: "cpu", threads: engine.threads, simd: engine.simd },
    lost: engine.lost,
    close: () => engine.close(),
  };
}

export interface LoadOptions {
  /** files.ts MODELS: Phonon 31M by default. */
  model?: ModelId;
  /** Hears the download in percent. */
  progress?: (percent: number) => void;
  /** "auto" (the default) runs on WebGPU when the browser gives an adapter, else on the CPU. */
  backend?: "auto" | "webgpu" | "cpu";
  /** CPU threads, when the page is cross-origin isolated: by default one per core, at most 8. */
  threads?: number;
}

/** Text cut where it must be so each part fits the model's text window:
 *  between sentences, then between words, then inside a word. */
function parts(text: string, count: (s: string) => number): string[] {
  if (count(text) <= TEXT_MAX) return [text];
  const out: string[] = [];
  let part = "";
  const add = (piece: string) => {
    if (part && count(part + piece) > TEXT_MAX) {
      out.push(part);
      part = "";
    }
    if (count(piece) <= TEXT_MAX) part += piece;
    else if (/\s/.test(piece.trim())) for (const word of piece.split(/(?<=\s)/)) add(word);
    else for (let i = 0; i < piece.length; i += 32) add(piece.slice(i, i + 32));
  };
  for (const sentence of text.match(/[^.!?]+(?:[.!?]+|$)\s*/g) ?? [text]) add(sentence);
  if (part) out.push(part);
  return out.map((p) => p.trim()).filter(Boolean);
}

/** A voice of Gradium's catalog. */
export interface Voice {
  name: string;
  id: string;
  language: Language;
  locale: string;
  gender: string;
  description: string;
}

/** Speech for a text: its 24 kHz samples and when each word, and each token in it, is heard. */
export interface Speech {
  audio: Float32Array;
  sampleRate: number;
  words: Word[];
  tokens: Span[];
}

export interface SpeakOptions {
  /** A catalog voice's name ("Delia") or id. The model's `defaultVoice` when not given. */
  voice?: string;
  /** Speaking speed, from about 0.77 to 1.3. */
  speed?: number;
  /** The sampling seed: the same seed and text give the same audio. */
  seed?: number;
  /** Each chunk of audio as soon as it is made: the first comes within milliseconds. */
  onAudio?: (samples: Float32Array) => void;
  /** Each word with its times, in order, as soon as nothing still to come can change them: once
   *  the token after it has its start, or the text has ended. They are the same as the result's
   *  `words`. With audio made much faster than real time, that is before the word is heard. A
   *  long word, or a slow device, can make a word arrive after its sound has begun. */
  onWord?: (word: Word) => void;
  signal?: AbortSignal;
}

/** The model, ready to speak. */
export class Phonon {
  private readonly runtime: Runtime;
  private readonly engine: Engine;
  private readonly tokenizer: Tokenizer;
  private readonly catalog: Voice[];
  /** Frames the alignment head reads ahead of the audio (files.ts). */
  private readonly lag: number;
  private closed = false;
  private readonly localVoices = new Map<string, Float32Array<ArrayBuffer>>();
  /** Settles when the GPU device is lost (a reset), or the CPU's worker stops: load again to go on. */
  readonly lost: Promise<unknown>;
  readonly backend: Backend;
  /** The voice `speak` uses when none is given: the default of the model's first language. */
  readonly defaultVoice: string;

  private constructor(files: ModelFiles, tokenizer: Tokenizer, catalog: Voice[], runtime: Runtime) {
    this.runtime = runtime;
    this.engine = runtime.engine;
    this.lost = runtime.lost;
    this.backend = runtime.backend;
    this.tokenizer = tokenizer;
    this.catalog = catalog;
    this.lag = files.align.lag;
    this.defaultVoice = DEFAULT_VOICE[files.languages[0]];
  }

  /** Loads a model: a download the first time, then the browser's cache. It runs on WebGPU
   *  when the browser has an adapter, else on the CPU, in a worker. */
  static async load({ model = "phonon-31m-en", progress, backend = "auto", threads }: LoadOptions = {}): Promise<Phonon> {
    const files: ModelFiles = MODELS[model];
    const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
    const adapter =
      backend === "cpu" ? null : await gpu?.requestAdapter({ powerPreference: "high-performance" }).catch(() => null);
    if (!adapter && backend === "webgpu") throw new Error("This browser has no WebGPU.");
    const [packed, tokenizer, catalog] = await Promise.all([
      cached(files.weights, progress),
      cached(files.tokenizer).then((b) => new Tokenizer(JSON.parse(new TextDecoder().decode(b)))),
      cached(CATALOG_URL).then((b) =>
        new TextDecoder()
          .decode(b)
          .split("\n")
          .filter(Boolean)
          .map((line) => JSON.parse(line) as Voice & { voice_id: string })
          .filter((v) => files.languages.includes(v.language))
          .map((v) => ({
            name: v.name,
            id: v.voice_id,
            language: v.language,
            locale: v.locale,
            gender: v.gender,
            description: v.description,
          })),
      ),
    ]);
    const runtime = adapter
      ? await startGpu(adapter, packed, files.align)
      : await startCpu(packed, files.align, threads ?? Math.min(8, navigator.hardwareConcurrency || 1));
    return new Phonon(files, tokenizer, catalog, runtime);
  }

  /** Releases the model: its GPU device, or its CPU worker and threads. A text in progress
   *  fails, and so does every later `speak`. */
  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    await this.runtime.close();
  }

  /** Local demo extension: registers already-encoded voice latents, without uploading. */
  addVoice(id: string, buffer: ArrayBuffer): void {
    if (this.closed) throw new Error("This model is closed.");
    if (!id.startsWith("local:")) throw new Error("Local voice IDs must start with local:");
    this.localVoices.set(id, readVoice(buffer));
  }

  /** The catalog's voices in the model's languages. */
  get voices(): readonly Voice[] {
    return this.catalog;
  }

  /** The words `speak` gives for a text, in order, each with its tokens' text, as the tokenizer
   *  writes them: a page can lay them out before the times arrive. */
  words(text: string): { text: string; tokens: string[] }[] {
    return this.parts(text).flatMap((part) => {
      const pieces = this.tokenizer.encode(part).map((id) => this.tokenizer.pieces[id]);
      return wordRanges(pieces).map((w) => ({
        text: w.text,
        tokens: pieces.slice(w.from, w.to).map((p) => p.replaceAll("▁", "")),
      }));
    });
  }

  private parts(text: string) {
    return parts(text, (s) => this.tokenizer.encode(s).length);
  }

  /** Text to speech, in parts that fit the model, joined. */
  async speak(text: string, options: SpeakOptions = {}): Promise<Speech> {
    if (this.closed) throw new Error("This model is closed.");
    const { voice = this.defaultVoice, speed = 1, seed = 1, onAudio, onWord, signal } = options;
    const v = this.catalog.find((c) => c.name === voice || c.id === voice);
    const local = this.localVoices.get(voice);
    if (!v && !local) throw new Error(`No voice ${voice}`);
    const voiceId = v?.id ?? voice;
    const latents = this.engine.hasVoice(voiceId) ? undefined : local ?? readVoice(await cached(voiceUrl(voiceId)));
    const audio: Float32Array[] = [];
    const words: Word[] = [];
    const tokens: Span[] = [];
    let at = 0;
    // Each part has its own seed: the noise depends on the seed and the frame, which starts at 0.
    for (const [i, part] of this.parts(text).entries()) {
      signal?.throwIfAborted();
      const ids = this.tokenizer.encode(part);
      const shift = <S extends Span>(s: S): S => ({ ...s, start: s.start + at, end: s.end + at });
      const shiftWord = (w: Word): Word => ({ ...shift(w), tokens: w.tokens.map(shift) });
      const times = new WordTimes(
        ids.map((id) => this.tokenizer.pieces[id]),
        this.lag,
        onWord && ((w) => onWord(shiftWord(w))),
      );
      const out = await this.engine.synthesize(ids, voiceId, {
        seed: seed + i,
        words: part.split(/\s+/).filter(Boolean).length,
        pace: paceForSpeed(speed),
        latents,
        onRows: (rows) => times.addRows(rows),
        onChunk: (chunk) => {
          times.addAudio(chunk);
          onAudio?.(chunk);
        },
        cancelled: () => !!signal?.aborted,
      });
      signal?.throwIfAborted();
      const pcm = join(out!.chunks);
      const timed = times.finish();
      words.push(...timed.words.map(shiftWord));
      tokens.push(...timed.tokens.map(shift));
      audio.push(pcm);
      at += pcm.length / SAMPLE_RATE;
    }
    return { audio: join(audio), sampleRate: SAMPLE_RATE, words, tokens };
  }
}

function join(parts: Float32Array[]): Float32Array {
  const out = new Float32Array(parts.reduce((n, c) => n + c.length, 0));
  let at = 0;
  for (const c of parts) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

/** 32-lane subgroups: the GEMVs sum with them where every subgroup has exactly 32 lanes. */
function subgroups32(adapter: GPUAdapter): boolean {
  const info = adapter.info as GPUAdapterInfo & { subgroupMinSize?: number; subgroupMaxSize?: number };
  return adapter.features.has("subgroups") && info.subgroupMinSize === 32 && info.subgroupMaxSize === 32;
}

export { TEXT_MAX } from "./engine";
export { DEFAULT_VOICE, MODELS, type Language, type ModelId } from "./files";
export type { Span, Word } from "./timing";
