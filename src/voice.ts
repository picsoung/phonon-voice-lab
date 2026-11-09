/** Validate the public ba7a5a45@300 voice layout before giving it to the engine. */
export function validateVoice(buffer: ArrayBuffer): void {
  const fail = (message: string): never => { throw new Error(message); };
  if (buffer.byteLength < 10 || buffer.byteLength > 4 * 1024 * 1024) fail('Choose a Phonon voice file smaller than 4 MB.');
  const size = Number(new DataView(buffer).getBigUint64(0, true));
  if (!Number.isSafeInteger(size) || size < 2 || size > 65536 || size + 8 > buffer.byteLength) fail('This is not a valid safetensors header.');
  let header;
  try { header = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(new Uint8Array(buffer, 8, size))); }
  catch { fail('This file has an invalid safetensors header.'); }
  if (!header || typeof header !== 'object') fail('This file has an invalid safetensors header.');
  const tensor = header.speaker_wavs;
  if (!tensor || Object.keys(header).some(k => k !== '__metadata__' && k !== 'speaker_wavs')) fail('Expected a voice file containing only speaker_wavs, not model weights.');
  if (tensor.dtype !== 'F32' || JSON.stringify(tensor.shape) !== '[1,512,125]') fail('Expected speaker_wavs with shape [1, 512, 125] and dtype F32.');
  const offsets = tensor.data_offsets;
  if (!Array.isArray(offsets) || offsets.length !== 2 || offsets[0] !== 0 || offsets[1] !== 512 * 125 * 4 || 8 + size + offsets[1] !== buffer.byteLength) fail('The voice data is truncated or has an invalid size.');
  const view = new DataView(buffer, 8 + size);
  for (let i = 0; i < 512 * 125; i++) if (!Number.isFinite(view.getFloat32(i * 4, true))) fail('The voice contains invalid numeric values.');
}
