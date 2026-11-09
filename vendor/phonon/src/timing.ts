// Word and token times from an alignment head of Phonon's LM (files.ts names each model's). Each
// 80 ms frame's scores over the text give a position in it. The head reads a few frames ahead of
// what is heard (its lag, also in files.ts), and a start that falls in silence moves to where the
// sound begins. WordTimes takes the frames and the audio as they are made and reports each word as soon
// as nothing still to come can change its times. It is the only path: a whole text is the same
// calls at once.
export const SAMPLE_RATE = 24000;
const FRAME_SECS = 0.08;
/** Quieter than this (dB) is silence. */
const SILENT_DB = -45;
/** Samples per loudness window (10 ms). */
const HOP = SAMPLE_RATE / 100;
/** Steps between the times a token can start at. */
const STEP_SECS = 0.005;
/** Pieces that join the word before them. */
const PUNCTUATION = new Set([".", ",", "!", "?", ";", ":", ")", '"', "'"]);

export interface Span {
  text: string;
  /** Seconds from the start of the audio. */
  start: number;
  end: number;
}
/** A word, with the tokens it is made of. */
export interface Word extends Span {
  tokens: Span[];
}

/** The words of a text's tokens: each word's text and its tokens [from, to). A token that starts
 *  with ▁ starts a word, unless it is only punctuation. */
export function wordRanges(pieces: string[]): { text: string; from: number; to: number }[] {
  const words: { text: string; from: number; to: number }[] = [];
  pieces.forEach((p, j) => {
    const text = p.replaceAll("▁", "");
    const joins = !p.startsWith("▁") || (!!text && [...text].every((c) => PUNCTUATION.has(c)));
    if (words.length && joins) {
      words.at(-1)!.text += text;
      words.at(-1)!.to = j + 1;
    } else words.push({ text, from: j, to: j + 1 });
  });
  return words.filter((w) => w.text);
}

/** One text's word and token times, from its alignment rows and audio as they arrive. */
export class WordTimes {
  private readonly pieces: string[];
  private readonly ranges: { text: string; from: number; to: number }[];
  private readonly onWord?: (word: Word) => void;
  /** Frames the head reads ahead of the audio. */
  private readonly lag: number;
  /** Each frame's position before the never-back rule, and after it (pos[0] waits for frame 1). */
  private readonly raw: number[] = [];
  private readonly pos: number[] = [];
  /** Loudness per complete 10 ms window, and the samples not yet in a window. */
  private readonly loud: number[] = [];
  private samples = 0;
  private partial = new Float32Array(HOP);
  private partialAt = 0;
  /** The next time a token can start at, and each token's start before and after the silence step. */
  private t = 0;
  private readonly starts: (number | null)[];
  private readonly at: number[] = [];
  private emitted = 0;
  private ended = false;

  constructor(pieces: string[], lag: number, onWord?: (word: Word) => void) {
    this.pieces = pieces;
    this.lag = lag;
    this.ranges = wordRanges(pieces);
    this.onWord = onWord;
    this.starts = Array.from({ length: pieces.length }, () => null);
  }

  /** Alignment rows of the next frames, `pieces.length` scores each. */
  addRows(rows: Float32Array) {
    const n = this.pieces.length;
    for (let r = 0; r + n <= rows.length; r += n) {
      const row = rows.subarray(r, r + n);
      let k = 0;
      for (let i = 1; i < n; i++) if (row[i] > row[k]) k = i;
      let num = 0;
      let den = 0;
      for (let i = Math.max(0, k - 2); i <= Math.min(n - 1, k + 2); i++) {
        const p = Math.exp(row[i] - row[k]);
        num += p * i;
        den += p;
      }
      this.raw.push(num / den);
      this.settle();
    }
    this.advance();
  }

  /** The next samples of the text's audio. */
  addAudio(pcm: Float32Array) {
    for (let i = 0; i < pcm.length; i++) {
      this.partial[this.partialAt++] = pcm[i];
      if (this.partialAt === HOP) {
        let s = 0;
        for (let j = 0; j < HOP; j++) s += this.partial[j] * this.partial[j];
        this.loud.push(20 * Math.log10(Math.sqrt(s / HOP) + 1e-9));
        this.partialAt = 0;
      }
    }
    this.samples += pcm.length;
    this.advance();
  }

  /** The text has ended: every remaining word, and all the times. */
  finish(): { tokens: Span[]; words: Word[] } {
    this.ended = true;
    if (!this.raw.length) return { tokens: [], words: [] };
    this.settle();
    this.advance();
    const tokens = this.pieces.map((_, j) => this.token(j));
    return { tokens, words: this.ranges.map((w) => this.word(w)) };
  }

  private get end() {
    return this.samples / SAMPLE_RATE;
  }

  /** Frame 0's row comes from the end-of-text query, which in some models reads the last token:
   *  it counts no further than frame 1's. Positions never move back. */
  private settle() {
    const { raw, pos } = this;
    if (!pos.length && (raw.length >= 2 || (this.ended && raw.length)))
      pos.push(raw.length > 1 ? Math.min(raw[0], raw[1]) : raw[0]);
    while (pos.length && pos.length < raw.length) pos.push(Math.max(raw[pos.length], pos[pos.length - 1]));
  }

  /** The head's position heard at time t, or null while frames still to come could change it. */
  private posAt(t: number): number | null {
    const { pos } = this;
    const g = (t - FRAME_SECS / 2) / FRAME_SECS - this.lag;
    if (!pos.length) return null;
    if (g <= 0) return pos[0];
    if (this.ended && g >= pos.length - 1) return pos[pos.length - 1];
    const i = Math.floor(g);
    if (i + 1 >= pos.length) return null;
    return pos[i] * (1 - (g - i)) + pos[i + 1] * (g - i);
  }

  /** Takes every time step and token start that can no longer change, then reports words. */
  private advance() {
    const n = this.pieces.length;
    if (!n || !this.raw.length) return;
    // A token starts at the first step whose position reaches it. Steps stop at the audio's end.
    for (; this.t <= this.end; this.t += STEP_SECS) {
      const p = this.posAt(this.t);
      if (p === null) break;
      const k = Math.round(p);
      for (let j = 0; j <= k && j < n; j++) this.starts[j] ??= this.t;
    }
    // A start that falls in silence moves to where the sound begins, and never before the last.
    while (this.at.length < n) {
      const j = this.at.length;
      const start = this.starts[j];
      if (start === null && !this.ended) break;
      const from = start ?? this.end;
      let i = Math.floor(from * 100);
      while (i < this.loud.length && this.loud[i] < SILENT_DB) i++;
      if (i >= this.loud.length && !this.ended) break;
      this.at.push(Math.max(from, i / 100, j ? this.at[j - 1] : 0));
    }
    // A word is known once the token after it has started (or the text has ended).
    while (this.emitted < this.ranges.length) {
      const w = this.ranges[this.emitted];
      if (!(this.ended || w.to < this.at.length)) break;
      this.onWord?.(this.word(w));
      this.emitted++;
    }
  }

  private token(j: number): Span {
    return {
      text: this.pieces[j].replaceAll("▁", ""),
      start: this.at[j],
      end: j + 1 < this.pieces.length ? this.at[j + 1] : this.end,
    };
  }
  private word(w: { text: string; from: number; to: number }): Word {
    const tokens = Array.from({ length: w.to - w.from }, (_, k) => this.token(w.from + k));
    return { text: w.text, start: tokens[0].start, end: tokens.at(-1)!.end, tokens };
  }
}
