// Phonon's files in the browser's Cache Storage, keyed by their pinned URLs (files.ts). The cache
// is optional: an insecure page has none, and a full one still leaves the bytes in use.
import { MODEL_CACHE } from "./files";

async function open(): Promise<Cache | null> {
  try {
    return await caches.open(MODEL_CACHE);
  } catch {
    return null;
  }
}

/** A cached response, if there is one. */
export async function lookup(key: string): Promise<Response | undefined> {
  try {
    return await (await open())?.match(key);
  } catch {
    return undefined;
  }
}

/** Keeps `body` under `key` when the cache has room. */
export async function store(key: string, body: BodyInit): Promise<void> {
  try {
    await (await open())?.put(key, new Response(body));
  } catch {
    /* Downloaded again next time. */
  }
}

async function download(url: string, progress?: (percent: number) => void): Promise<Uint8Array<ArrayBuffer>> {
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`The voice model could not download (${response.status}).`);
  const total = Number(response.headers.get("content-length")) || 0;
  const parts: Uint8Array[] = [];
  let loaded = 0;
  const reader = response.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.byteLength;
    if (total && progress) progress(Math.min(100, Math.round((loaded / total) * 100)));
  }
  const bytes = new Uint8Array(loaded);
  let o = 0;
  for (const p of parts) {
    bytes.set(p, o);
    o += p.byteLength;
  }
  return bytes;
}

/** A file from the cache, or downloaded (reporting progress in percent) and then kept. */
export async function cached(url: string, progress?: (percent: number) => void): Promise<ArrayBuffer> {
  const hit = await lookup(url);
  if (hit) return hit.arrayBuffer();
  const bytes = await download(url, progress);
  await store(url, bytes);
  return bytes.buffer;
}
