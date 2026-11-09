import {build} from 'esbuild';
import {test} from 'node:test';
import assert from 'node:assert/strict';
const built=await build({entryPoints:['src/voice.ts'],bundle:true,format:'esm',platform:'node',write:false});
const {validateVoice}=await import('data:text/javascript;base64,'+Buffer.from(built.outputFiles[0].text).toString('base64'));
function fixture(tensor={dtype:'F32',shape:[1,512,125],data_offsets:[0,256000]}){const h=Buffer.from(JSON.stringify({speaker_wavs:tensor}));const b=Buffer.alloc(8+h.length+256000);b.writeBigUInt64LE(BigInt(h.length));h.copy(b,8);return b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);}
test('accepts the released Phonon voice layout',()=>assert.doesNotThrow(()=>validateVoice(fixture())));
test('rejects WAV and truncated data',()=>{assert.throws(()=>validateVoice(new ArrayBuffer(44)));assert.throws(()=>validateVoice(fixture().slice(0,-4)));});
test('rejects incompatible tensor dimensions',()=>assert.throws(()=>validateVoice(fixture({dtype:'F32',shape:[1,256,250],data_offsets:[0,256000]})),/shape/));
test('rejects non-finite speaker latents',()=>{const b=fixture();new DataView(b).setFloat32(b.byteLength-4,NaN,true);assert.throws(()=>validateVoice(b),/numeric/);});
