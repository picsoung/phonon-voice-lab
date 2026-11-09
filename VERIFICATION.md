# Verification

## Passed

- `npm run build`: builds the extended Blixt runtime, CPU worker and app.
- `npm test`: voice layout validation, malformed/truncated data, wrong dimensions and NaN rejection.
- Chromium 1440px desktop and 390px mobile: no horizontal overflow; captured the generated-audio state.
- Real Delia catalog baseline generated successfully on CPU/WASM.
- The same real embedding selected as a local `.safetensors` file generated successfully through the new registration API.
- Uploaded-file synthesis made zero network requests; the output was a non-silent 24 kHz PCM WAV, 115244 bytes, 2.40 seconds. One run measured 93 ms to first generated chunk and 0.47 seconds total generation. These environment-specific numbers are not performance promises.
- Invalid file upload produces an error while retaining the last valid voice.

## Test environment caveat

The managed proxy certificate is trusted by Python but not the temporary Chromium profile. The first browser attempt could not fetch Hugging Face files due to `ERR_CERT_AUTHORITY_INVALID`. For the successful test, assets were downloaded with TLS verification using Python and delivered to their original URLs with Playwright request fixtures. Production code still fetches the normal pinned Hugging Face URLs; TLS verification was not disabled. Thus the test proves model inference and local-file integration, but does not prove direct model download works in this managed Chromium profile.

The browser test script expects these TLS-verified fixtures in `/tmp/phonon-fixtures/{weights,tokenizer,catalog}` and `/tmp/phonon-test-voice.safetensors`. It is a development harness, not a self-contained CI test.

No new custom clone was created: the test uses a real published catalog embedding through the uploaded-file path. No claim of speaker similarity, subjective voice quality or new cloning functionality is made. Exported custom voices with the same published layout should use the same path, but a user-provided custom voice still needs a listening test.

## WebGPU result

The separate WebGPU run also passed baseline and uploaded-file generation, produced a non-silent 115244-byte, 2.40-second WAV (peak PCM amplitude 16907), and made zero network requests during local-file synthesis. Chromium used a software GPU (SwiftShader), with approximately 9.46 seconds to first chunk and 79.79 seconds total generation. These timings do not represent hardware GPU performance. The desktop/mobile screenshots retain the CPU run's measurements.
