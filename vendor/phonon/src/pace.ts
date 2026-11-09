// Phonon's pace: a duration_delta condition the LM adds to every audio
// position, from its speaker-count and duration embeddings. Negative is
// faster, positive slower.

export interface PaceWeights {
  speakers: Float32Array;
  speakersProj: Float32Array;
  durationProj: Float32Array;
  latentProj: Float32Array;
  bos: Float32Array;
}

const H = 512;

/** y = W x for W [rows][cols]. */
function matvec(w: Float32Array, x: Float32Array, rows: number): Float32Array {
  const cols = x.length;
  const y = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = 0;
    for (let c = 0; c < cols; c++) s += w[r * cols + c] * x[c];
    y[r] = s;
  }
  return y;
}

/** The condition at a pace, and the BOS row's input, which carries it too. */
export function paceVectors(
  p: PaceWeights,
  delta: number,
): { cond: Float32Array; bos: Float32Array } {
  const half = 64;
  const phases = new Float32Array(2 * half);
  for (let i = 0; i < half; i++) {
    const a = (1000 * delta) / 10000 ** (i / (half - 1));
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

/** Speaking speed to pace, from measured lengths of one passage: about 0.77x
 *  at +0.6 and 1.3x at -0.45, and no faster beyond. Interpolated between. */
const PACE: [speed: number, delta: number][] = [
  [0.77, 0.6],
  [0.8, 0.45],
  [0.84, 0.3],
  [0.93, 0.15],
  [1, 0],
  [1.07, -0.15],
  [1.22, -0.3],
  [1.3, -0.45],
];
export function paceForSpeed(speed: number): number {
  const at = PACE.findIndex(([s]) => speed <= s);
  if (at === 0) return PACE[0][1];
  if (at < 0) return PACE.at(-1)![1];
  const [s0, d0] = PACE[at - 1];
  const [s1, d1] = PACE[at];
  return d0 + ((speed - s0) / (s1 - s0)) * (d1 - d0);
}
