import {chromium} from 'playwright-core';
import {writeFile,mkdir} from 'node:fs/promises';
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true,args:['--no-sandbox','--enable-unsafe-webgpu','--use-angle=swiftshader','--enable-features=Vulkan']});
const page=await browser.newPage({viewport:{width:1440,height:1100}});
page.on('pageerror',e=>console.log('PAGE ERROR',e.message));
page.on('requestfailed',r=>console.log('REQUEST FAILED',r.url(),r.failure()?.errorText));
// The managed proxy CA is trusted by Python, but not this Chromium profile.
// Serve TLS-verified, downloaded fixtures through Playwright without disabling TLS validation.
await page.route('https://huggingface.co/**', async route => {
 const u=route.request().url();
 const path=u.endsWith('.int8.bin')?'/tmp/phonon-fixtures/weights':u.endsWith('tokenizer.5.0.0.json')?'/tmp/phonon-fixtures/tokenizer':u.endsWith('voices.jsonl')?'/tmp/phonon-fixtures/catalog':u.includes('/voices/')?'/tmp/phonon-test-voice.safetensors':null;
 if(!path)throw new Error('Unexpected fixture request '+u);
 await route.fulfill({path,headers:{'access-control-allow-origin':'*','cross-origin-resource-policy':'cross-origin'}});
});
await page.goto('http://localhost:5173');
await mkdir('.impeccable/review',{recursive:true});
await page.setViewportSize({width:390,height:844});
console.log('responsive overflow',await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth));
await page.setViewportSize({width:1440,height:1100});
await page.locator('#backend').selectOption(process.env.TEST_BACKEND || 'cpu');
await page.locator('#load').click();
await page.waitForFunction(()=>!document.querySelector('#speak').disabled || document.querySelector('#status').dataset.error==='true',{},{timeout:120000});
console.log('load',await page.locator('#status').innerText());
if(await page.locator('#speak').isDisabled())throw new Error('Model did not load');
await page.locator('#text').fill('Hello from a local voice file.');
await page.locator('#speak').click();
await page.waitForFunction(()=>!document.querySelector('#speak').disabled,{},{timeout:120000});
console.log('baseline',await page.locator('#status').innerText());
if(await page.locator('#download').isHidden())throw new Error('Baseline did not generate');
await page.locator('#file').setInputFiles('/tmp/phonon-test-voice.safetensors');
await page.waitForFunction(()=>document.querySelector('#file-status').textContent.includes('valid Phonon'));
let requests=[];page.on('request',r=>{if(!r.url().startsWith('blob:'))requests.push(r.url());});
await page.locator('#speak').click();
await page.waitForFunction(()=>!document.querySelector('#speak').disabled,{},{timeout:120000});
console.log('local voice',await page.locator('#status').innerText());
if(await page.locator('#download').isHidden())throw new Error('Local voice did not generate');
console.log('local generation requests',requests);
console.log('metrics',await page.locator('dl').innerText());
const samples=await page.evaluate(async()=>{const b=await (await fetch(document.querySelector('#audio').src)).arrayBuffer();const v=new DataView(b);let peak=0;for(let i=44;i<b.byteLength;i+=2)peak=Math.max(peak,Math.abs(v.getInt16(i,true)));return {bytes:b.byteLength,peak};});
console.log('WAV evidence',samples);if(samples.peak<100)throw new Error('Generated audio appears silent');
if (!process.env.TEST_BACKEND) {await page.screenshot({path:'.impeccable/review/desktop.png',fullPage:true});await page.setViewportSize({width:390,height:844});await page.screenshot({path:'.impeccable/review/mobile.png',fullPage:true});}
await page.locator('#file').setInputFiles({name:'bad.safetensors',mimeType:'application/octet-stream',buffer:Buffer.from('invalid')});await page.waitForFunction(()=>document.querySelector('#status').dataset.error==='true');console.log('invalid file',await page.locator('#status').innerText());
await browser.close();
