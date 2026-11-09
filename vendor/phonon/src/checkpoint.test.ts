import { test } from "node:test";
import assert from "node:assert/strict";
import { packModel, quantize, unpackModel, type PhononModel } from "./checkpoint";
import { Q8_GROUP } from "./graph";

test("quantizes each group to its largest value over 127, rounding halves to even", () => {
  const values = new Float32Array(2 * Q8_GROUP);
  values[0] = 127;
  values[1] = 2.5;
  values[2] = -3.5;
  values[3] = 1.5;
  const { q, scales } = quantize(values);
  assert.deepEqual([...q.slice(0, 4)], [127, 2, -4, 2]);
  // f16 bits of 1 for the first group, and of 1 again for the all-zero group.
  assert.deepEqual([...scales], [0x3c00, 0x3c00]);
  assert.ok([...q.slice(Q8_GROUP)].every((v) => v === 0));
});

test("keeps a model through the cache's form", () => {
  const model: PhononModel = {
    blob: Uint8Array.from({ length: 512 }, (_, i) => i % 251),
    tensors: { up: { offset: 0, count: 256, scales: 256 }, cond: { offset: 256, count: 3 } },
    pace: {
      speakers: Float32Array.of(1, 2),
      speakersProj: Float32Array.of(3),
      durationProj: Float32Array.of(4, 5, 6),
      latentProj: Float32Array.of(7),
      bos: Float32Array.of(8, 9),
    },
  };
  const back = unpackModel(packModel(model).buffer);
  assert.deepEqual(back.tensors, model.tensors);
  assert.deepEqual([...back.blob], [...model.blob]);
  for (const [k, v] of Object.entries(model.pace)) assert.deepEqual([...back.pace[k as keyof typeof model.pace]], [...v]);
});
