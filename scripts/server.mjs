import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
const root = resolve('public');
const mime={'.html':'text/html; charset=utf-8','.js':'text/javascript','.css':'text/css','.wasm':'application/wasm','.json':'application/json'};
createServer(async (req,res)=>{
  try {
    const path=resolve(root, '.'+decodeURIComponent(new URL(req.url,'http://localhost').pathname));
    if(path!==root && !path.startsWith(root+sep)) {res.writeHead(403).end();return;}
    const file=(await stat(path)).isDirectory()?resolve(path,'index.html'):path;
    res.writeHead(200,{'Content-Type':mime[extname(file)]||'application/octet-stream','Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp'});
    res.end(await readFile(file));
  } catch {res.writeHead(404).end('Not found');}
}).listen(Number(process.env.PORT||5173),'0.0.0.0',()=>console.log('Voice Lab: http://localhost:'+(process.env.PORT||5173)));
