import { test } from "node:test";
import assert from "node:assert/strict";
import { SAMPLE_RATE, WordTimes, type Word } from "./timing";

/** A head that moves through n tokens over the frames, its first row on the last token (as some
 *  40M models' end-of-text query reads), and audio with a pause in it. */
function take(n: number, frames: number) {
  const rows = new Float32Array(frames * n);
  for (let f = 0; f < frames; f++) {
    const at = f === 0 ? n - 1 : Math.min(n - 1, Math.floor(((f - 1) / (frames - 4)) * n));
    for (let i = 0; i < n; i++) rows[f * n + i] = -Math.abs(i - at) * 3;
  }
  const pcm = new Float32Array(frames * 1920);
  for (let i = 0; i < pcm.length; i++) {
    const t = i / SAMPLE_RATE;
    pcm[i] = t < 0.1 || (t > 0.9 && t < 1.2) ? 0 : 0.2 * Math.sin(i / 7);
  }
  return { rows, pcm };
}

test("words reported while frames and audio arrive equal the words of the whole text", () => {
  const pieces = ["▁Die", "▁Donau", "dampf", "schiff", "fahrt", "▁fährt", "▁sch", "nell", "."];
  const n = pieces.length;
  const { rows, pcm } = take(n, 40);
  const whole = new WordTimes(pieces, 3);
  whole.addRows(rows);
  whole.addAudio(pcm);
  const all = whole.finish();

  const streamed: Word[] = [];
  const live = new WordTimes(pieces, 3, (w) => streamed.push(w));
  // Rows ahead of audio in uneven pieces, as the engine's readbacks give them.
  let f = 0, a = 0;
  for (const step of [1, 1, 2, 4, 8, 8, 8, 8]) {
    live.addRows(rows.subarray(f * n, Math.min(40, f + step) * n));
    f = Math.min(40, f + step);
    const upto = Math.max(0, f - 2) * 1920;
    live.addAudio(pcm.subarray(a, upto));
    a = upto;
  }
  live.addAudio(pcm.subarray(a));
  const early = streamed.length;
  const done = live.finish();

  assert.deepEqual(done, all);
  assert.deepEqual(streamed, all.words);
  assert.deepEqual(all.words.map((w) => w.text), ["Die", "Donaudampfschifffahrt", "fährt", "schnell."]);
  assert.ok(early >= 3, "words arrive before the text ends");
  // Frame 0's row on the last token does not pin every word to the start.
  assert.ok(all.words[3].start > all.words[0].start + 0.5);
});
