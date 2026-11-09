"""Download public test assets with TLS verification; no Gradium key needed."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

root = Path('/tmp/phonon-fixtures')
root.mkdir(exist_ok=True)
base = 'https://huggingface.co/'
assets = {
 root/'weights': base+'blixt/phonon-31m-en-webgpu/resolve/f4635a6dddaa6072451e0155c03c97fbe889b6ec/phonon-31m-en.int8.bin',
 root/'tokenizer': base+'gradium/phonon-31m-en/resolve/92741c5b07d085ff840e26fea8ce43110b87852f/tokenizer.5.0.0.json',
 root/'catalog': base+'gradium/phonon-voices/resolve/0ee17205f254b5abc43b266c8a0f215565c352f7/voices.jsonl',
 Path('/tmp/phonon-test-voice.safetensors'): base+'gradium/phonon-voices/resolve/0ee17205f254b5abc43b266c8a0f215565c352f7/voices/nGzMHMPAWGaJFHOo.safetensors',
}
def download(item):
 path, url = item
 with urlopen(url, timeout=60) as response:
  data = response.read()
 path.write_bytes(data)
 print(f'{path.name}: {len(data)} bytes', flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:
 list(pool.map(download, assets.items()))
