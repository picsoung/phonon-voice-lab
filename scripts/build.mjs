import { build } from 'esbuild';
import { mkdir, cp } from 'node:fs/promises';
await mkdir('public/runtime/cpu', {recursive:true});
await build({entryPoints:['vendor/phonon/src/index.ts'],bundle:true,splitting:true,format:'esm',target:'es2022',outdir:'public/runtime',entryNames:'phonon',chunkNames:'cpu/[name]-[hash]'});
await build({entryPoints:['vendor/phonon/src/cpu-worker.ts'],bundle:true,format:'esm',target:'es2022',define:{__PROFILE__:'false'},outfile:'public/runtime/cpu/phonon-cpu.js'});
for (const name of ['phonon-cpu-relaxed.wasm','phonon-cpu-simd.wasm']) await cp(`vendor/phonon/dist/cpu/${name}`,`public/runtime/cpu/${name}`);
await build({entryPoints:['src/app.ts'],bundle:true,external:['/runtime/phonon.js'],format:'esm',target:'es2022',outfile:'public/app.js'});
