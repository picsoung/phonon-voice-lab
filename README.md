# Phonon Voice Lab

A feasibility demo for **Phonon `.safetensors` voice → browser speech**, using Blixt's WebGPU runtime with its CPU/WASM fallback. It does not clone WAV recordings.

## Run

```sh
npm install
npm run build
npm run dev
```

Open http://localhost:5173. Load the engine, generate a catalog baseline, then choose a Phonon voice file and generate the same text. You can replay and download 24 kHz mono PCM WAV audio. `npm test` checks voice validation. Serve over HTTPS or localhost; the dev server supplies COOP/COEP headers for threaded CPU inference.

The initial model download is about 44 MB, from Hugging Face. Browser Cache Storage retains model assets. Uploaded voice files are read in memory, never uploaded or persisted. A reload requires choosing your local voice again. Load and generate once online before testing cached generation without network. A new catalog voice may still need downloading.

## Obtain a custom voice

With uv installed and `GRADIUM_API_KEY` configured in your terminal:

```sh
uv run https://raw.githubusercontent.com/gradium-ai/xn-ptts/main/scripts/clone-voice.py \
  speaker.wav --lang en -o voice.safetensors
```

Gradium performs cloning remotely and saves the voice in your account. Use a consenting speaker. This demo needs no API key. Drop the exported file into the demo; do not drop the source WAV.

## Integration and local change

Vendor source: https://github.com/blixt/phonon at `36e7e5eff820733be38e9df19ccf58ae891bd45c`, Apache-2.0. Its LICENSE and NOTICE are preserved in `vendor/phonon`.

Upstream parses voice safetensors internally, but its public `speak` API only resolves catalog IDs/names. The local patch in `vendor/phonon/src/index.ts` adds `addVoice(id, arrayBuffer)` and uses registered latents in `speak`. It supports both the existing WebGPU and CPU paths. No upstream contribution has been published.

```js
// Local extension, not an upstream API:
validateVoice(buffer);
tts.addVoice('local:my-voice', buffer);
const speech = await tts.speak('Hello.', { voice: 'local:my-voice' });
```

`src/voice.ts` validates the released voice format: a single F32 `speaker_wavs` tensor shaped `[1, 512, 125]`, with finite values and exact data length. Other layouts are deliberately rejected. A valid shape does not prove voice quality; listen to generated samples.

This demo uses Phonon 31M English only (Apache-2.0). Catalog voices have their own published terms. Downloading a non-English voice does not add a language to this model.

## Measurement and scope

First audio measures the time until the runtime emits its first chunk; playback starts after the full result is ready. Generation time includes voice preparation/loading. Backend is reported explicitly; CPU success is not evidence of WebGPU success. The demo includes cancellation, WAV export, a real waveform and malformed-file validation. It has no cloning backend, billing, authentication or YouTube import.

## Reproduce the browser check

With Python 3 and Chromium at `/usr/bin/chromium`, and the demo server already running:

```sh
python3 scripts/prepare-fixtures.py
node scripts/browser-test.mjs
# Optional, with a working WebGPU adapter:
TEST_BACKEND=webgpu node scripts/browser-test.mjs
```

The harness routes the downloaded public files to their original Hugging Face URLs to work around this managed Chromium profile's proxy CA issue, without disabling TLS checks. It verifies baseline generation, local voice ingestion, non-silent WAV output, no requests during local synthesis, malformed-file errors and mobile overflow. See VERIFICATION.md for results and limits.
