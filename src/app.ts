import { validateVoice } from './voice';
// Built from the pinned, locally extended Blixt source. See README.md.
// @ts-ignore runtime JS is generated during build
import { Phonon } from '/runtime/phonon.js';
const $ = <T extends HTMLElement>(id:string) => document.getElementById(id) as T;
const load=$<HTMLButtonElement>('load'), speak=$<HTMLButtonElement>('speak'), stop=$<HTMLButtonElement>('stop');
const catalog=$<HTMLSelectElement>('catalog'), backend=$<HTMLSelectElement>('backend'), file=$<HTMLInputElement>('file');
const text=$<HTMLTextAreaElement>('text'), speed=$<HTMLInputElement>('speed'), audio=$<HTMLAudioElement>('audio');
const remove=$<HTMLButtonElement>('remove'), download=$<HTMLAnchorElement>('download');
let tts:any, local:{id:string;name:string;buffer:ArrayBuffer}|undefined, busy=false, loading=false, selecting=false, serial=0;
let url:string|undefined, controller:AbortController|undefined;
function status(message:string,error=false){$('status').textContent=message;$('status').dataset.error=String(error);}
function controls(){speak.disabled=!tts||busy||loading||selecting||!text.value.trim();load.disabled=!!tts||busy||loading;backend.disabled=!!tts||loading;file.disabled=busy||loading;catalog.disabled=!!local||busy||loading;remove.disabled=busy||loading;speed.disabled=busy;stop.hidden=!busy;}
text.addEventListener('input',()=>{$('count').textContent=`${text.value.length} / 2000`;controls();});text.dispatchEvent(new Event('input'));
speed.addEventListener('input',()=>{$('speed-value').textContent=`${Number(speed.value).toFixed(2)}×`;});
file.addEventListener('change',async()=>{
 const selected=file.files?.[0];if(!selected)return;selecting=true;controls();
 try{
  if(selected.size>4*1024*1024)throw new Error('Choose a Phonon voice file smaller than 4 MB.');
  const buffer=await selected.arrayBuffer();validateVoice(buffer);
  const candidate={id:`local:${++serial}`,name:selected.name,buffer};
  tts?.addVoice(candidate.id,candidate.buffer);local=candidate;
  $('file-status').textContent=`${selected.name} · valid Phonon voice · ${(selected.size/1024).toFixed(0)} KB`;
  remove.hidden=false;status('Voice file ready. It has not been uploaded.');
 }catch(error){status((error as Error).message,true);}finally{file.value='';selecting=false;controls();}
});
remove.addEventListener('click',()=>{local=undefined;remove.hidden=true;$('file-status').textContent='Your file stays on this device.';status('Catalog voice selected.');controls();});
load.addEventListener('click',async()=>{
 loading=true;controls();const progress=$<HTMLProgressElement>('progress');progress.hidden=false;status('Downloading and preparing the model…');
 try{
  tts=await Phonon.load({backend:backend.value,progress:(p:number)=>{progress.value=p;status(`Loading model · ${Math.round(p)}%`);}});
  if(local)tts.addVoice(local.id,local.buffer);
  catalog.replaceChildren(...tts.voices.map((v:any)=>{const o=document.createElement('option');o.value=v.id;o.textContent=`${v.name} · ${v.locale}`;if(v.name==='Delia')o.selected=true;return o;}));
  $('metric-backend').textContent=tts.backend.name==='webgpu'?'WebGPU':`CPU · ${tts.backend.threads} thread${tts.backend.threads===1?'':'s'}`;
  load.textContent='Engine ready';status('Ready. Choose a voice and generate speech.');
  tts.lost.then(()=>{tts=undefined;load.textContent='Reload engine';status('The engine stopped. Reload it to continue.',true);controls();});
 }catch(error){if(tts){await tts.close();tts=undefined;}status(`Could not load the engine: ${(error as Error).message}. Try CPU mode or a WebGPU-capable browser.`,true);}
 finally{loading=false;progress.hidden=true;controls();}
});
stop.addEventListener('click',()=>{controller?.abort();audio.pause();});
speak.addEventListener('click',async()=>{
 if(!tts||busy)return;busy=true;controller=new AbortController();controls();audio.pause();
 if(url){URL.revokeObjectURL(url);url=undefined;}audio.removeAttribute('src');audio.load();download.hidden=true;
 const started=performance.now();let first:number|undefined;status('Generating speech on your device…');
 for(const id of ['metric-first','metric-time','metric-duration'])$(id).textContent='—';
 try{
  const result=await tts.speak(text.value.trim(),{voice:local?.id??catalog.value,speed:Number(speed.value),seed:1,signal:controller.signal,onAudio:()=>{if(first===undefined)first=performance.now()-started;}});
  const elapsed=performance.now()-started;
  if(!result.audio.length)throw new Error('The engine returned no audio.');
  url=URL.createObjectURL(wav(result.audio,result.sampleRate));audio.src=url;download.href=url;download.hidden=false;
  $('metric-first').textContent=first===undefined?'—':`${first.toFixed(0)} ms`;$('metric-time').textContent=`${(elapsed/1000).toFixed(2)} s`;$('metric-duration').textContent=`${(result.audio.length/result.sampleRate).toFixed(2)} s`;
  draw(result.audio);status(`Generated with ${local?.name??catalog.selectedOptions[0].textContent}. All synthesis ran locally.`);
  try{await audio.play();}catch{status('Audio ready. Press play below to listen.');}
 }catch(error){status(controller.signal.aborted?'Generation cancelled.':`Generation failed: ${(error as Error).message}`,!controller.signal.aborted);}
 finally{busy=false;controller=undefined;controls();}
});
function wav(pcm:Float32Array,rate:number){const out=new ArrayBuffer(44+pcm.length*2),v=new DataView(out);const str=(at:number,s:string)=>{for(let i=0;i<s.length;i++)v.setUint8(at+i,s.charCodeAt(i));};str(0,'RIFF');v.setUint32(4,36+pcm.length*2,true);str(8,'WAVE');str(12,'fmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,rate,true);v.setUint32(28,rate*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);str(36,'data');v.setUint32(40,pcm.length*2,true);pcm.forEach((x,i)=>v.setInt16(44+i*2,Math.max(-1,Math.min(1,x))*(x<0?32768:32767),true));return new Blob([out],{type:'audio/wav'});}
let waveform:Float32Array|undefined;
function draw(pcm?:Float32Array){if(pcm)waveform=pcm;const canvas=$<HTMLCanvasElement>('wave'),rect=canvas.getBoundingClientRect();canvas.width=Math.round(rect.width*devicePixelRatio);canvas.height=90*devicePixelRatio;const ctx=canvas.getContext('2d')!;ctx.scale(devicePixelRatio,devicePixelRatio);ctx.strokeStyle='#36694f';ctx.lineWidth=1.5;ctx.beginPath();if(!waveform){ctx.moveTo(16,45);ctx.lineTo(rect.width-16,45);}else{const bars=Math.floor((rect.width-32)/3),step=Math.max(1,Math.floor(waveform.length/bars));for(let i=0;i<bars;i++){let peak=0;for(let j=i*step;j<Math.min(waveform.length,(i+1)*step);j++)peak=Math.max(peak,Math.abs(waveform[j]));const h=Math.max(1,peak*38);ctx.moveTo(16+i*3,45-h);ctx.lineTo(16+i*3,45+h);}}ctx.stroke();}
new ResizeObserver(()=>draw()).observe($('wave'));
window.addEventListener('pagehide',()=>{controller?.abort();if(url)URL.revokeObjectURL(url);void tts?.close();});
