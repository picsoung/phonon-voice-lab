# Phonon on WebGPU

Fast text-to-speech in the browser, with a start and end time for every word.

[Try it in your browser](https://blixt.xyz/phonon/) (downloads 44.3 MB the first time) · [Listen to the benchmark phrases](https://blixt.xyz/phonon/listen.html)

## What it does

blixt/phonon is a WebGPU runtime for [Phonon 31M](https://huggingface.co/gradium/phonon-31m-en), an Apache-2.0 text-to-speech model by [Gradium](https://gradium.ai). Gradium did not make blixt/phonon. It also runs Gradium's larger 40M models in English, French, German, Spanish and Portuguese, which are not for commercial use ([Other models](#other-models)).

- It makes 24 kHz English speech in any of Gradium's English voices, and streams the audio as it goes.
- It gives the start and end time of every word and of each token in it, while it speaks.
- The speaking speed goes from about 0.77× to 1.3×.
- It downloads Gradium's weights with the large matrices converted to 8-bit integers (43.7 MB, [on Hugging Face](https://huggingface.co/blixt/phonon-31m-en-webgpu)) and keeps them in the browser's cache.
- It runs in Chrome on macOS and Windows and in Safari on iOS, and a user ran it in Firefox on a Pixel 10 Pro. Other browsers are not tested.
- Without WebGPU it runs on the CPU, in WebAssembly ([The CPU tier](#the-cpu-tier)). On the benchmark's sentences it makes the same take as WebGPU, with the same word times. On an M3 Max it makes speech 37 times faster than real time on 8 threads and 7 times on one, against 11 and 6 times for phonon-tts on the CPU on 3 threads and one.

Against Gradium's own browser package, [phonon-tts](https://www.npmjs.com/package/phonon-tts), on WebGPU, with Phonon 31M:

- Speed: 115× real time against 31× on an M3 Max, 73× against 21× on an RTX 2080 Ti, and 26× against 13× on an iPhone 17 Pro.
- First audio of a sentence: 3 ms against 15 ms on the M3 Max, and 11 ms against 44 ms on the iPhone.
- A fresh page with the model cached: first audio after 46 ms against 373 ms on the M3 Max.
- First visit: 44.3 MB to download against 66.8 MB. With the download at 50 Mbit/s, first audio after 7.2 s against 10.9 s on the M3 Max, and 7.3 s against 11.3 s on the iPhone.
- Word times: phonon-tts gives none.
- Speech quality: Whisper heard no word errors in 110 clips, against 0.22% for phonon-tts. Speaker similarity is the same, and UTMOS predicts 4.40 for both.

## Results

![Speed of each runtime on each device](docs/figures/speed.svg)

![Time to first audio](docs/figures/first-audio.svg)

![Cold start](docs/figures/cold-start.svg)

<!-- results:start -->
| Device | Runtime | Speed (× real time) | First audio, warm (ms) | Cold start (ms) | First visit at 50 Mbit/s (s) | Download (MB) | All 11 phrases, warm (s) |
|---|---|--:|--:|--:|--:|--:|--:|
| MacBook Pro, M3 Max | **blixt/phonon** | 115 | 3.2 | 46 | 7.2 | 44.3 | 0.4 |
| MacBook Pro, M3 Max | phonon-tts, WebGPU | 31 | 14.9 | 373 | 10.9 | 66.8 | 1.8 |
| MacBook Pro, M3 Max | phonon-tts, CPU | 15 | 9.8 | 324 | 10.9 | 66.8 | 3.9 |
| MacBook Pro, M3 Max | Kokoro 82M, WebGPU | 14 | 207.4 | 847 | 54.1 | 331.1 | 4.1 |
| PC, RTX 2080 Ti | **blixt/phonon** | 73 | 6.1 | 177 | 7.3 | 44.3 | 0.7 |
| PC, RTX 2080 Ti | phonon-tts, WebGPU | 21 | 26.5 | 702 | 11.3 | 66.8 | 2.7 |
| PC, RTX 2080 Ti | phonon-tts, CPU | 6 | 25.5 | 569 | 11.1 | 66.8 | 10.1 |
| PC, RTX 2080 Ti | Kokoro 82M, WebGPU | 11 | 271.7 | 2022 | 55.5 | 330.0 | 5.2 |
| iPhone 17 Pro, Safari 27.2 | **blixt/phonon** | 26 | 10.7 | 147 | 7.3 | 44.3 | 2.1 |
| iPhone 17 Pro, Safari 27.2 | phonon-tts, WebGPU | 13 | 43.8 | 550 | 11.3 | 66.8 | 5.1 |
| iPhone 17 Pro, Safari 27.2 | phonon-tts, CPU | 6 | 31.2 | 568 | 12.0 | 66.8 | 10.5 |
| iPhone 17 Pro, Safari 27.2 | Kokoro 82M, WebGPU | did not finish | did not finish | did not finish | did not finish | – | did not finish |
<!-- results:end -->

Each value is a median, and the bars show the 25th and 75th percentiles. Speed is seconds of speech per second of compute, over every phrase of every warm run: 30× means that 10 seconds of speech take about a third of a second to make. Cold start is a fresh page that loads the runtime and speaks one sentence, with the files already in the browser's cache. First visit is the same on a page that first deletes everything its site stores, so it downloads every file, and it is given at 50 Mbit/s: the time any download was running is replaced by the bytes the visit downloaded at that speed. A first visit is otherwise mostly download time, which depends on the connection, not on the runtime. Its bytes are what came over the network (Download in the table) and each runtime's own code at its gzip size, as a static host serves it. Kokoro's first audio is its whole first sentence, because the benchmark gives it one sentence per call.

blixt/phonon gets faster as texts get longer, because each text has a fixed cost before its first frame. The other runtimes stay near one speed:

![Speed against text length](docs/figures/speed-by-length.svg)

The word times come from the model's attention, corrected with the audio's loudness. This figure compares each word start with [wav2vec2](https://huggingface.co/facebook/wav2vec2-base-960h) forced alignment of the same audio:

![Word timing error](docs/figures/word-timing.svg)

All four runtimes speak clearly. [bench/quality.py](bench/quality.py) scored each runtime's 110 warm clips on the M3 Max with [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) and [UTMOS](https://github.com/tarepan/SpeechMOS):

![Quality](docs/figures/quality.svg)

## Listen

The [listening page](https://blixt.xyz/phonon/listen.html) plays every benchmark phrase from each runtime, side by side. The three Phonon runtimes use the same model and the same voice (Delia). They sample with different random numbers, so each clip is a different take of the same text. Kokoro 82M is a different model, there for reference. The clips are also in [docs/audio](docs/audio).

## Use it

```sh
npm install github:blixt/phonon   # the package is named phonon-webgpu
```

```js
import { Phonon } from "phonon-webgpu";

const tts = await Phonon.load({ progress: (percent) => console.log(percent) });
const speech = await tts.speak("The birch canoe slid on the smooth planks.", {
  voice: "Delia", // any voice of Gradium's catalog in the model's languages: tts.voices
  speed: 1,
  onAudio: (samples) => {}, // each chunk, as soon as it exists
  onWord: (word) => {}, // each word with its times, in order, as soon as they are known
});
// speech.audio: Float32Array at speech.sampleRate (24000)
// speech.words: [{ text: "The", start: 0.13, end: 0.40, tokens: [{ text: "The", start: 0.13, end: 0.40 }] }, …], in seconds
// speech.tokens: every token's time
```

`onWord` gives the same words as `speech.words`, as soon as nothing still to come can change them. `tts.words(text)` lists the words and their tokens before speaking, so a page can lay them out first. The demo does that, and lights each word token by token while it is heard, so a long compound like "Donaudampfschifffahrtsgesellschaft" lights in pieces.

`Phonon.load` runs on WebGPU when the browser gives it an adapter, and otherwise on the CPU, in a worker. `backend: "webgpu"` or `"cpu"` picks one, and `tts.backend` says which runs (on the CPU, with its threads and its WebAssembly build). The CPU uses threads only on a cross-origin isolated page, one served with `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. Elsewhere it runs on one thread. `threads` sets how many, and by default it is one per core, at most 8. The first load downloads 44.3 MB from Hugging Face (the weights, the tokenizer and a voice), and later loads read them from the browser's cache. `tts.close()` releases a model: its GPU device, or its worker and threads.

`dist/phonon.js` is 25,116 bytes with gzip. The CPU tier's files are in `dist/cpu`, and a page fetches them only when it runs on the CPU: the page's side of the engine (1,343 bytes with gzip), its worker (14,077 bytes) and one WebAssembly module (16,802 bytes, or 17,016 for the build without relaxed SIMD). Minified, as the demo serves them, the first two are 980 and 9,832 bytes.

The code is in [src](src): [lm.ts](src/lm.ts) has the language model and [codec.ts](src/codec.ts) the audio decoder, as ops of a backend ([graph.ts](src/graph.ts)). [gpu.ts](src/gpu.ts) runs those ops on WebGPU with the WGSL shaders of [kernels.ts](src/kernels.ts), and [cpu.ts](src/cpu.ts) runs them in WebAssembly, with the Rust kernels of [cpu](cpu). [engine.ts](src/engine.ts) is the loop that drives either, and [timing.ts](src/timing.ts) makes the word times.

## Other models

Gradium also publishes six Phonon 40M models. They have 9 language-model layers where Phonon 31M has 6, and they are under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), which does not allow commercial use. blixt/phonon runs all of them, from conversions in [blixt/phonon-40m-webgpu](https://huggingface.co/blixt/phonon-40m-webgpu) (53.7 MB each), and the [demo](https://blixt.xyz/phonon/) has them in its model menu.

| Model | Languages | License | First download |
|---|---|---|--:|
| `phonon-31m-en` (the default) | English | Apache 2.0 | 44.3 MB |
| `phonon-40m-en` | English | CC BY-NC 4.0 | 54.3 MB |
| `phonon-40m-fr`, `phonon-40m-de`, `phonon-40m-es`, `phonon-40m-pt` | French, German, Spanish, Portuguese | CC BY-NC 4.0 | 54.3 MB |
| `phonon-40m-multilingual` | all five | CC BY-NC 4.0 | 54.3 MB |

Gradium's listeners preferred each one-language model to the multilingual one in its language, and Phonon 31M about as much as the multilingual model in English ([model card](https://huggingface.co/gradium/phonon-40m-multilingual#benchmarks)). A model's voices are the catalog's voices in its languages:

```js
const tts = await Phonon.load({ model: "phonon-40m-fr" });
const speech = await tts.speak("Le petit chat dort au soleil près de la fenêtre.", { voice: "Coralie" });
```

A shorter run of the benchmark on the M3 Max checked Phonon 40M multilingual against phonon-tts with the same model, with the 31M runtimes in the same rounds: 3 rounds of first visits, then 5 rounds of cold runs and 5 rounds of warm runs, on the same English texts in the voice Delia ([bench/results/40m](bench/results/40m)). The three extra layers make each frame cost more, and the other 40M models do the same work:

| Runtime | Model | Speed (× real time) | First audio, warm (ms) | Cold start (ms) | First visit at 50 Mbit/s (s) | Download (MB) |
|---|---|--:|--:|--:|--:|--:|
| **blixt/phonon** | 31M | 116 | 2.6 | 43 | 7.2 | 44.3 |
| **blixt/phonon** | 40M multilingual | 96 | 3.2 | 50 | 8.7 | 54.3 |
| phonon-tts, WebGPU | 31M | 29 | 19.5 | 345 | 10.9 | 66.8 |
| phonon-tts, WebGPU | 40M multilingual | 25 | 28.9 | 211 | 12.5 | 76.6 |

## How it works

Everything here goes beyond what the model card and the phonon-tts source say. "The reference" is the PyTorch model in Hugging Face Transformers (`GradiumPhononTTSForConditionalGeneration`), and the baseline is phonon-tts 0.6.0. Numbers marked (A/B) are the effect of one technique, measured with and without it on the M3 Max in headless Chrome. They are not from the benchmark above.

### Starting point

The engine is new code: WGSL compute shaders, TypeScript and, for the CPU, Rust compiled to WebAssembly, with no phonon-tts or ONNX Runtime inside. [bench/parity.py](bench/parity.py) checks the engine against the reference. It runs the reference in f32 with the engine's own noise in place of `torch.randn`, so both should make the same take of each text. On the 11 benchmark texts in the voice Delia, all 11 came out the same length to the frame (the Rainbow passage is 349 frames). The log-mel distance from the reference was 2.6 dB on average, and the waveforms correlated at 0.51. For scale, the reference with a different seed was 9.5 dB away, correlated at 0.21, and had a different length in 6 of the 11 texts. The waveforms are not identical, because int8 weights and an f16 key/value cache change the fine detail of each frame.

The tokenizer must be `tokenizer.5.0.0.json`, not `tokenizer.json`. Only the first has the 191 normalizer rules the model expects, which, for example, turn brackets into commas and add a final period. [tokenizer.ts](src/tokenizer.ts) runs those rules, Metaspace splitting and a Unigram Viterbi search with byte fallback, and its ids match the Python tokenizer.

### Weights

[scripts/build-weights.ts](scripts/build-weights.ts) converts Gradium's `model.safetensors` (the language model in bf16, the codec decoder in f32) once, with [checkpoint.ts](src/checkpoint.ts), into the file the library downloads from [blixt/phonon-31m-en-webgpu](https://huggingface.co/blixt/phonon-31m-en-webgpu). It does not use the GGUF file.

- The checkpoint stores rotary pairs interleaved (dims 2j and 2j+1). The converter moves the rows of q and k to halves (j and j+32) in each head, so a kernel rotates with two contiguous reads.
- The flow head does one LSD step, from time 0 to time 1. Its time embedding is the same at every step, and the converter adds it once to the bias of `cond_embed`.
- The codec's latent projection (1×1, no bias) is merged into the depthwise ×16 upsampling weights, and one kernel denormalizes, projects and upsamples the latents.
- The codec transformer's layer scales are multiplied into its output projection and second MLP weights.
- Convolution weights become [out][tap][in]. Transposed convolutions have kernel = 2 × stride, and each becomes one GEMM per output phase, [out][phase][tap][in], in which a kernel gathers its inputs and never scatters.
- Every tensor with 16,384 values or more becomes int8, with one f16 scale per 32 values along its rows. The scale is max |value| / 127 after rounding the values to f16, and quantized values round half to even. Other tensors stay f16. Shaders read the whole blob as `u32` words and unpack with `unpack4xI8` and `unpack2x16float`, and all arithmetic is f32, which means no shader needs the `shader-f16` feature.
- On the 110 warm clips per runtime of the benchmark, Whisper large-v3-turbo found no word errors in blixt/phonon's speech (95% interval 0 to 0%), against 0.22% (0 to 0.54%) for phonon-tts on WebGPU. The paired difference is −0.22 points (−0.53 to 0). WavLM speaker embeddings of blixt/phonon's clips are as close to phonon-tts's clips of the same text (cosine 0.988) as phonon-tts's clips are to each other across seeds (0.989). UTMOS predicts 4.40 for both. Nobody has run a listening test.
- The file is 43.7 MB, against 103 MB for the checkpoint and 64.5 MB for Gradium's GGUF file. `npm run build-weights` rebuilds it byte for byte, and its model card gives the SHA-256.
- The 40M checkpoints have the same tensors, with 9 language-model layers in place of 6. The converter reads the layer count from the checkpoint. Each file is 53.7 MB, against 122 MB for a checkpoint and 74.6 MB for Gradium's GGUF file. Each model has its own tokenizer, and blixt/phonon's ids match the Python tokenizer's on text in all 5 languages. Whisper large-v3-turbo heard test sentences from each model in its languages as written.

### Speed

In order of their effect:

1. The whole autoregressive loop stays on the GPU. A 32-byte state buffer holds the position the next step reads, the frame it makes, the end of speech and the codec's position (`STATE` in [kernels.ts](src/kernels.ts)), and each step's last kernel moves it on. The page encodes many steps into one submit without reading anything back. Results come back through mapped staging buffers, up to 3 submits behind the GPU.
2. One GEMV kernel does every single-row matrix product. 32 lanes work on each output row, each lane loads 16 bytes of weights at a time, and the lanes sum with subgroup operations where every subgroup has exactly 32 lanes (Apple and NVIDIA). Other GPUs use a shared-memory reduction, which is also tested. LayerNorm, with its affine weights and the flow head's adaLN modulation, runs as the GEMV's prologue in one pass (sum and sum of squares). Together with the int8 weights, this GEMV cuts the language model's step from 0.65 ms to 0.30 ms (A/B).
3. Each voice is prefilled once. A voice is 125 rows of speaker latents. The engine runs them through the language model's layers once (the last layer only as far as its keys and values) and keeps their keys and values (1.5 MB in f16 for the 6 layers of Phonon 31M), and each text copies them into the cache and prefills only its own tokens. This cuts first audio from 8.8 ms to 4.9 ms (A/B).
4. Nothing is allocated per request. All buffers and bind groups exist from the start: about 108 MB with Phonon 31M, plus 1.5 MB for each voice used. The key/value cache is f16.
5. A step's attention is split over the keys (flash-decoding). 8 workgroups per head each take a share of the keys and leave a partial sum with its maximum, and the output projection's GEMV merges the splits in its prologue, at no extra dispatch. Inside a split, 8 lanes share each key for the scores, and 16 lanes × 8 key groups make the values. This halves a layer's attention, from 30 µs to 15 µs (A/B).
6. The GPU decides the end of speech. The step's last kernel computes the end-of-speech logit, and when it passes 4.0 the kernel writes how many frames to keep. Every heavy kernel reads that number and returns at once for later frames, which makes the steps a submit runs past the end cost about 2 µs each. On short texts, this raises the speed from 115× to 130× (A/B).
7. Audio comes out in chunks of 1, 2, 4 and then 8 frames. All chunk sizes share one decoder state: the codec transformer's ring of keys and values (window 251, ring 512 positions at 200 Hz) and each convolution stage's left context. After each chunk, one small kernel moves every stage's last rows to its front, and the convolutions never need a position. The engine clears that context before each text. Chunk size does not change the audio in my tests, though the repository has no test for it.
8. Pipelines compile in the background. Each bind group layout is read from its shader's `@binding` declarations, which lets bind groups exist before `createComputePipelineAsync` finishes. This cuts the first request after a load from about 53 ms to 13-19 ms (A/B).
9. The prefill has 4 fixed sizes. The text and BOS rows are padded to 64, 128, 192 or 256 rows, each size has prebuilt operations, and a text only writes its ids. Padding rows cannot change the real rows, because attention is causal. Each padding row writes keys and values at a position after BOS, and the step at that position writes its own key and value before it reads the cache. After the prefill, the BOS row is copied to row 0, where the step's flow head reads.
10. The codec's convolutions are split-K GEMMs with 64×64 tiles, A and B in shared memory as vec4 along K (16 KB, the portable limit). When a GEMM has few output tiles, K is split to reach about 160 workgroups and a second kernel sums the parts. Row tiles are capped at 4 when choosing the splits, so all text lengths share a few shader variants.
11. The prefill and the codec transformer use a flash-style attention kernel: 16 query rows share each tile of 16 keys in shared memory, with an online softmax. It gives about 2% (A/B).

An empty dependent dispatch costs about 1.6 µs in Chrome and WebKit on the M3 Max. A Phonon 31M step has 49 dispatches (5 for each of the 6 layers, 19 for the flow head and the end of the step), and their barriers and ramps take about a third of its 300 µs.

### The CPU tier

A browser without WebGPU runs the same model on the CPU. [lm.ts](src/lm.ts) and [codec.ts](src/codec.ts) build their ops through an interface ([graph.ts](src/graph.ts)) that both backends implement, so both run one graph, and [engine.ts](src/engine.ts) drives both with the same loop. The CPU's ops are about a dozen kernels in Rust with WebAssembly SIMD ([cpu](cpu)), in one module of 77 KB (17 KB with gzip), and a worker of 10 KB with gzip. Each op is a descriptor of a few words in the module's memory, written once when the graph is built, and a submit runs its ops in one call. The engine runs in a worker, so speaking never blocks the page. The weights are the same file. The key and value cache is f32, where WebGPU's is f16.

On the M3 Max, under load from other work (load average about 20), in Chrome, interleaved over 5 rounds of the 11 benchmark texts:

| Runtime | Threads | Speed (× real time) | First audio, warm (ms) | Cold start (ms) |
|---|--:|--:|--:|--:|
| **blixt/phonon, CPU** | 8 | 37.5 | 4.0 | 86 |
| **blixt/phonon, CPU** | 3 | 17.8 | 8.5 | 99 |
| **blixt/phonon, CPU** | 1 | 7.0 | 20.1 | 138 |
| phonon-tts, CPU | 3 | 11.2 | 12.2 | 310 |
| phonon-tts, CPU | 1 | 5.7 | 23.6 | 451 |

In order of their effect:

1. Relaxed SIMD. A second build of the module uses `f32x4.relaxed_madd`, a fused multiply-add, and the page loads it where the browser validates it (Chrome 114, Firefox 146). It makes speech 1.47 times faster on one thread (7.1× against 4.8×, A/B). Safari has no relaxed SIMD and runs the other build.
2. Threads. With shared memory, pool workers take part in every op. Each op's work is in units that threads claim from a counter in chunks, so a thread on an efficiency core, or one the system preempts, holds back no one. An op waits only for the threads that joined it, and a thread that wakes after the engine's thread has finished the op skips it. On 12 threads this gives 38.5× against 28.5× for fixed shares, and on 16 threads of a busy machine 22.7× against 10.5× (A/B). Speech is the same for any number of threads: a unit's sums never depend on which thread runs it.
3. Dot-product GEMM tiles. A tile of rows by columns keeps its sums along K in vector lanes, from contiguous loads of the activations and of the weights dequantized into a panel, and adds the lanes once at its end. On the codec's first convolution it ran at 31 GMAC/s against 23 for the best tile that broadcasts one activation at a time (A/B). The best shape depends on the CPU: 4 × 4 or 2 × 8 on the M3 Max, and 2 × 4 on a Ryzen 7 9700X, which has 16 SIMD registers to the M3's 32 (35 against 25 GMAC/s). The engine times each shape on two of the codec's products when it first loads (70 ms on one thread of the M3 Max) and keeps the fastest in the browser's cache.
4. No work on padding. A text's prefill marks its padding rows (`Gate` "text" in [graph.ts](src/graph.ts)), and the CPU skips them. WebGPU still computes them, as its prefill sizes are fixed, and its output is unchanged. This cut first audio of a sentence on one thread from 48 ms to 20 ms in the benchmark.
5. int16 GEMV. An LM step's input is rounded to int16 with one scale per 32 values, and each row of int8 weights meets it in `i32x4.dot_i16x8_s`, 8 products per instruction. It made the GEMVs 1.7 to 2.3 times faster on the Ryzen, and speech 3% faster on the M3 Max (A/B).

The threads need a cross-origin isolated page, and GitHub Pages cannot set headers. The demo, when it finds no WebGPU adapter, registers a small service worker ([coi.js](site/coi.js)) that adds them, and moves to its own URL with `?cpu`. The service worker's scopes are that URL and the folder `cpu/` of the CPU tier's files, as a worker's script is matched by its own URL. It never controls the demo's plain URL, so a visitor with WebGPU is never controlled, isolated or reloaded, in any tab. A page that finds an adapter unregisters it, and a `?cpu` page with an adapter moves back to the plain URL. This was tested in headless Chrome in both orders, a visit without WebGPU then one with it and the reverse, and the CPU side in Firefox and WebKit. On the M3 Max the reload makes the CPU tier about 5 times faster (37.5× on 8 threads against 7.0× on one), and the reload took 80 to 100 ms from a local server, on a first visit that downloads 44 MB anyway. To compare the two tiers on one device, open the demo with `?cpu=force`: it then runs on the CPU, with threads, even where WebGPU works. A library user can set the headers on their own server, or run on one thread.

The CPU tier is checked against WebGPU and the reference:

- [bench/ops.mjs](bench/ops.mjs) runs every op of the graph on WebGPU and on the CPU with the model's weights and fixed inputs: each GEMM tile shape, both GEMVs and both builds, 268 outputs. Each output must be within 1e-4 of its largest value, 1e-3 with the int16 GEMV, and 2e-3 where WebGPU keeps the cache in f16 or takes the sine of a large angle. A kernel that changes on one side fails it.
- [bench/golden.mjs](bench/golden.mjs) speaks 5 clips on each backend and checks each against the reference's take ([bench/golden.json](bench/golden.json), from [bench/golden.py](bench/golden.py)): the same length in frames, within 4 dB in log-mel distance, and word starts within one frame of the other backend. A change to the shared graph fails it on both backends.
- Parity ([bench/parity.py](bench/parity.py)) over the 11 benchmark texts with seeds 1, 2 and 3 in the voice Delia: WebGPU made 31 of the 33 clips the same length as the reference, the CPU 30. On the 30 Harvard clips both were 2.68 dB from the reference on average, correlated at 0.53. The Rainbow passage drifted from the reference on both, at a different point in each seed.
- Word starts against wav2vec2 alignment, on the 11 texts in 4 voices (712 words): a mean of 66.8 ms and a median of 65 ms on the CPU, against 66.7 ms and 65 ms on WebGPU. On the 10 Harvard sentences in Delia the CPU's word times equal WebGPU's to the bit.
- All five 40M models gave the same lengths as on WebGPU, and the same word times on 8 of 10 test sentences.
- Chrome 153, Firefox 155 and WebKit (headless, on macOS) all ran it, threaded and not, and made the same audio to the bit.

The CPU tier's memory is 115 MB for Phonon 31M and 139 MB for a 40M model, in the module's memory: the weights (43.3 or 53.4 MB), the f32 key and value cache (25 or 38 MB), the codec's stages and the activations. The worker keeps no other copy of the weights: Chrome measured its whole worker at 116 MB with Phonon 31M on one thread.

### Word timings

1. Find a head that follows the text. I ran the reference on four texts and recorded each head's attention over the text tokens at every 80 ms frame. Each head got a score: the correlation between the frame number and the token it attends most, times the share of frames that do not move back, times its mean peak attention. Layer 3, head 0 scored 0.78. The next best (layer 3, head 5) scored 0.45, and all others 0.15 or less. Layer 3, head 0 has a correlation of 0.99 or more, never moves back and puts about 80% of its attention on one token.
2. Record it on the GPU. In every step, the layer 3 attention kernel writes head 0's scores over the text tokens into an alignment buffer, one row per frame. A small extra kernel in the prefill writes frame 0's row from the BOS row's query. The rows of each submit's frames come back with its audio, in the readback the engine makes anyway, about 1 KB per frame.
3. Turn the scores into a position. For each frame, take the token with the highest score and average the token index over the 2 tokens on each side of it, with softmax weights. The position never moves back. Frame 0's position counts no further than frame 1's: frame 0's row comes from the end-of-text query, and in some 40M models that query reads the last token, which would pin every word to the start.
4. Correct the lead. The head reaches a word a few frames before the audio does, and the times shift back by the model's lag: 3 frames (240 ms), or 2 frames for the German 40M model. After a pause, the head reaches the next word 400 to 570 ms early, because it moves on when the previous word ends. A word start that falls where the audio is quieter than −45 dB (in 10 ms steps) moves to where the sound begins.
5. Group the tokens into words. A token that starts with ▁ starts a word, unless it is only punctuation.
6. Report each word once nothing still to come can change it. Every step looks only at earlier frames and audio, except three short looks ahead: interpolation needs the next frame, the silence step needs the audio up to where sound begins, and a word's end is the next word's start. [timing.ts](src/timing.ts) takes frames and audio as they arrive and reports a word as soon as the token after it has its start. A whole text goes through the same code, and on 47 texts in all seven models (814 words) the words reported while speaking equalled the words of the finished text. A word's end waits for the next word's start, and the last word waits for the end of the text. With audio made much faster than real time, a word arrives before it is heard. A long word, or a device that makes audio only a little faster than real time, can make it arrive after its sound has begun.

I chose the 3-frame shift and the −45 dB threshold on 110 words of four other texts. On those words, a 4-frame shift without the silence step was a mean 110 ms from a wav2vec2 forced aligner, and the chosen settings were about 55 ms. The figure above is a held-out check on the 11 benchmark texts in 4 voices (712 words). Word starts are a mean 67 ms and a median 65 ms from the aligner, 76% are within 100 ms, and the 90th percentile is 125 ms. The median signed difference is +40 ms, with the engine later. Only word starts were scored here, and the aligner has its own error of a few tens of milliseconds. The "never move back" rule could also hide a repeated or skipped word.

Each 40M model has its own head and lag, and [files.ts](src/files.ts) names them. [bench/heads.mjs](bench/heads.mjs) speaks 4 other texts in the model's language once with each of the 72 heads (9 layers of 8) and replays the word times at several lags. [bench/heads.py](bench/heads.py) scores them against [MMS](https://pytorch.org/audio/stable/generated/torchaudio.pipelines.MMS_FA.html) forced alignment of romanized text, which works in all five languages and scored the multilingual model's English within 2 ms of [wav2vec2](https://huggingface.co/facebook/wav2vec2-base-960h). The head is the one with the lowest mean word-start error at lag 3. Another lag replaces 3 only when it is no worse on any measure (mean, median, 90th percentile and share within 80 ms, of word starts and of tokens inside words) and better on the mean. That is German's lag of 2, which also held in each voice and on held-out text. For the other models, every other lag made some measure worse. In each model one head was best by far:

| Model | Head (layer, head), lag | Search: next best | Held out: mean, median | Within 100 ms |
|---|---|---|--:|--:|
| `phonon-40m-en` | 6, 0, 3 | 3, 4 | 65 ms, 66 ms | 79% |
| `phonon-40m-fr` | 4, 1, 3 | 4, 4 | 42 ms, 32 ms | 93% |
| `phonon-40m-de` | 3, 5, 2 | 3, 3 | 39 ms, 31 ms | 93% |
| `phonon-40m-es` | 4, 6, 3 | 4, 4 | 60 ms, 58 ms | 85% |
| `phonon-40m-pt` | 4, 2, 3 | 4, 5 | 58 ms, 55 ms | 86% |
| `phonon-40m-multilingual` | 4, 1, 3 | 4, 4 | 35 to 57 ms by language | 84 to 97% |

The held-out check is 6 other texts in 2 voices for each language (106 to 124 words), scored with MMS. The multilingual model's head was chosen on English and held up in all five languages.

Token starts inside words come from the same head and are about as close as word starts. MMS aligns each character, which gives the true start of a token in the middle of a word:

| Model, texts | Word starts: mean, median | Tokens inside words: mean, median | Tokens inside words: median signed |
|---|--:|--:|--:|
| `phonon-31m-en`, 11 benchmark texts, 2 voices | 69 ms, 69 ms | 66 ms, 57 ms | +47 ms |
| `phonon-40m-fr`, 6 texts, 2 voices | 39 ms, 31 ms | 48 ms, 34 ms | +4 ms |
| `phonon-40m-de` | 39 ms, 31 ms | 45 ms, 35 ms | +20 ms |
| `phonon-40m-es` | 60 ms, 56 ms | 87 ms, 91 ms | +90 ms |
| `phonon-40m-pt` | 58 ms, 55 ms | 87 ms, 57 ms | +52 ms |

Much of the rest is a steady late offset in a model, not scatter: Spanish tokens inside words are a median 90 ms late. For those models, a lag that would remove it makes another measure worse.

### Sampling

The flow head's noise comes from a PCG hash of the seed, the frame and the channel, through Box-Muller, at temperature 0.3. phonon-tts and the reference use other random number generators, and the same seed gives a different take of the same text.

A text ends when the end-of-speech logit passes 4.0, plus 1 frame, or 3 frames for a text of at most 4 words, as the checkpoint's `processor_config.json` says. A text gets at most (tokens / 3 + 2) seconds of frames, as in the reference, and never more than 768 frames (61 s). One text can have up to 240 tokens. [index.ts](src/index.ts) cuts longer texts at sentence ends, then at word ends, and joins the parts and their word times. Each part gets the next seed, because the noise depends on the seed and the frame, and each part starts again at frame 0.

Speed uses the model's `duration_delta` condition. The engine computes the condition vector and the BOS row for a value in the browser and writes them into the weights. I measured clip lengths at several values to make a table from speed to value: 0.77× at +0.6 and 1.3× at −0.45. Values below −0.45 made speech no faster. [pace.ts](src/pace.ts) interpolates the table.

### Failed experiments

- Allocating uniforms and split-K buffers for each text added up to 17.3 GB of GPU buffers over 412 requests, all left to the garbage collector.
- Averaging the token index over all tokens, not just around the peak, made the word highlight jump ahead: a little attention on the last token pulled frame 0 to about the 4th token.
- More attention splits (16 or 32 per head) changed nothing.
- Deeper submit queues, or batches of 8 frames before the GPU decided the end, were slower or delayed the first audio.
- Indirect dispatch to skip work past the end costs 5-6 µs per dispatch, against 1.6 µs for a direct one. Kernels that return at once are cheaper.
- vec4 shared-memory tiles in the codec GEMM gave no measurable gain over scalar tiles. The kernel keeps them, since they cost nothing. My first try had a data race: WGSL can store a whole vector when a thread writes one of its components, and threads must not share a vec4 that each writes in part. A padding column against bank conflicts did not fit in 16 KB.
- int4 weights for the language model and the flow head cut 43 MB to 29.5 MB with the same recognizer errors, but gave no speed. They moved clip lengths by up to ±10%, against ±3% for int8. Nobody listened to them, and int8 stayed.
- Chunks of 16 frames made a test hang. I did not find the cause, and chunks stop at 8.
- New split-K variants per text length made the first text of each new length compile new shaders (332 ms). Capping the row tiles at 4 fixed it.
- `layout: "auto"` drops bindings that a shader does not use, and a bind group with the full set then fails. Every kernel now touches all its bindings, and the layouts come from the declarations.
- More than 8 storage buffers in one shader breaks the portable limit. The weights are one buffer, bound once, and kernels get each tensor's offset in a uniform.
- An `uncapturederror` handler in a worker crashes the page after a synthesis in WebKit. The engine has none.
- On the CPU, a GEMM tile that broadcasts one activation at a time to 8 or 16 columns lost to the best dot-product tile on every product measured, on the M3 Max and on the Ryzen. A dot-product tile of 6 × 4 needs more vector registers than the M3 Max has, and ran at 15 GMAC/s against 24 for 4 × 4.
- A SIMD epilogue for 4 rows of a GEMV at a time, in place of a scalar one for each row, did not change the f32 GEMV's speed on the Ryzen (4 GMAC/s for `fc1`).

### Benchmark notes

- `kokoro-js` `stream()` never returned on a plain string here. The benchmark calls `generate` once per sentence.
- Headless WebKit with an in-memory profile keeps no Cache Storage between workers. Use a profile on disk.
- Playwright drops the page when a route fulfils a 100 MB body in Chromium. The benchmark redirects to a local server instead.
- The PC could not reach the Mac over Tailscale (a shared node is one-way). An SSH tunnel carried the page and Chrome's debugging port, and the page loaded as `localhost`, which WebGPU accepts as a secure origin.
- phonon-tts loads only public Hugging Face repositories in a browser, and Gradium's 40M models are gated. For the 40M run, `HF_LOCAL` in [bench/drive.mjs](bench/drive.mjs) served the model's files from a local download and answered phonon-tts's lookup of the repository as for a public one.
- Kokoro 82M did not finish on the iPhone. A first visit and a cold run each finished once. After that, Safari either closed the tab on load ("A problem repeatedly occurred") or the page stopped before its first audio. Its fp32 model is 326 MB. The iPhone ran it on its own, after the three Phonon runtimes, because a crashed tab stops the page.
- On the iPhone, phonon-tts on the CPU ran on 1 thread in 5 of its 21 runs: its worker threads failed to start with "Out of memory", and it went on without them.

### Open work

- Fewer dispatches: folding the noise into the flow head's input GEMV, and the end-of-speech, output and latent projections into one workgroup, should give 3-6%.
- Faster codec convolutions: wider tiles per thread, and direct kernels for the last convolution (1 output channel) and `s3r1` (32 channels).
- Text normalization of numbers and symbols, as phonon-tts does.
- A CPU build for Safari with int16 GEMMs (`i32x4.dot_i16x8_s`), as Safari has no fused multiply-add in WebAssembly. It should make the codec about 1.5 times faster there.
- CPU attention that shares each key and value between 4 query rows, and ELU in the convolutions' loads (together about 10% of the CPU's time).
- An f16 key and value cache on the CPU, which would save 13 MB with Phonon 31M.

## Method

| Device | Browser | Note |
|---|---|---|
| MacBook Pro, Apple M3 Max | Chrome 153, headless | Metal 3, 32-lane subgroups |
| PC, NVIDIA RTX 2080 Ti, AMD Ryzen 7 3700X, Windows 11 | Chrome 156, headless | Turing, 32-lane subgroups |
| iPhone 17 Pro (A19 Pro) | Safari 27.2 | Safari reports 4 CPU cores |

- Texts: the 10 Harvard sentences of list 1 (IEEE, 1969) and the first paragraph of the Rainbow Passage (Fairbanks, 1960), in [bench/phrases.json](bench/phrases.json).
- Runtimes: blixt/phonon, phonon-tts 0.6.0 on WebGPU and on the CPU (3 threads, `q8` weights, no text normalization so that both Phonon runtimes do the same work), and kokoro-js 1.2.1 with Kokoro 82M in fp32 on WebGPU. Both Phonon runtimes use the same model revision and the voice Delia. Kokoro uses `af_heart`.
- Runs: 10 rounds of first visits, then 10 rounds of cold runs and 10 rounds of warm runs. Each round runs every runtime once, in an order that rotates each round. One untimed cold run per runtime fills the caches again before the cold runs.
- First visits: before each one, the page deletes its site's Cache Storage, IndexedDB and origin private file system. Every fetch to another origin, in the page and in its workers, goes past the HTTP cache, and [bench/netcount.js](bench/netcount.js) records its bytes and when its first and last byte came. The runtimes' code comes from the benchmark server, and the analysis counts it at its gzip size. After each first visit, a 16 MB download from Hugging Face measures the connection. A cold run is a fresh page that loads the runtime and speaks the first Harvard sentence. A warm run is a fresh page that loads the runtime, speaks one untimed sentence, then speaks all 11 texts with that round's seed. The Mac and the PC each waited until their CPU was under 15% and their GPU under 10% for a minute before they started ([bench/wait-quiet.sh](bench/wait-quiet.sh) on the Mac). The iPhone ran with Safari in the foreground.
- First audio is the time from the call to the first chunk of audio.

To run it again:

```sh
npm install
npm run build
node bench/run.mjs                                  # serves the page and plans the runs
node bench/drive.mjs "http://127.0.0.1:7893/bench/?device=mac"   # or open that page on a device
python bench/plots.py && python bench/table.py      # figures and table
```

The 40M run, with Gradium's gated files downloaded to a local folder:

```sh
node bench/run.mjs --first-runs 3 --runs 5 --runtimes ours,oursMultilingual,gradiumWebgpu,gradiumWebgpuMultilingual --audio none --out bench/results/40m
HF_LOCAL="gradium/phonon-40m-multilingual=path/to/download" node bench/drive.mjs "http://127.0.0.1:7893/bench/?device=mac"
```

The backend checks (bench/ops.mjs and bench/golden.mjs) run in headless Chrome with WebGPU, through the `playwright-core` dev dependency. Playwright's Chromium is a separate download, once:

```sh
npm install
node node_modules/playwright-core/cli.js install chromium
npm run build                               # dist/cpu holds the WebAssembly; npm run build-cpu rebuilds it with Rust
npm run test-backends
```

`PLAYWRIGHT=/path/to/a/package.json` makes the bench scripts take `playwright-core` from another install. The CPU tier's run, and its parity with the reference:

```sh
node bench/run.mjs --runs 5 --kinds cold,warm --runtimes oursCpu,oursCpu3,oursCpu1,gradiumCpu,gradiumCpu1 --audio none --out bench/results/cpu
node bench/drive.mjs "http://127.0.0.1:7893/bench/?device=mac"
node bench/parity.mjs out/webgpu webgpu 1,2,3 && node bench/parity.mjs out/cpu cpu 1,2,3
python bench/parity.py out/webgpu out/cpu
```

`npm run build-cpu` rebuilds the WebAssembly modules, and needs Rust with the wasm32-unknown-unknown target and rust-src.

## Limits

- Without WebGPU, the browser must have WebAssembly SIMD (Chrome 91, Firefox 89, Safari 16.4). On Android, Chrome turns WebGPU off under Advanced Protection and on some GPUs, and the demo then speaks on the CPU. "Copy details" copies the browser, the GPU, the backend and the last error for a bug report.
- Safari has no relaxed SIMD, and its CPU tier runs the build without fused multiply-adds, about 1.5 times slower.
- Only Phonon 31M, in English, is Apache-2.0. The 40M models are under CC BY-NC 4.0: no commercial use.
- phonon-tts can write out numbers, amounts, dates and symbols as words before it speaks, in each language's own way. blixt/phonon cannot, and numbers read best written as words.
- The word times follow the model's attention. They are not a forced alignment.

## Credits and license

The models, their tokenizers and the voices are by [Gradium](https://gradium.ai). Phonon 31M is under the Apache License 2.0, and the Phonon 40M models under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Phonon builds on [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) and the Mimi codec by [Kyutai](https://kyutai.org). This repository does not include a model. The code downloads Gradium's voices from Hugging Face at a pinned revision, and the converted weights from [blixt/phonon-31m-en-webgpu](https://huggingface.co/blixt/phonon-31m-en-webgpu) and [blixt/phonon-40m-webgpu](https://huggingface.co/blixt/phonon-40m-webgpu), which credit Gradium and keep each model's license.

The code in this repository is under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE).
