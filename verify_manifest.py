from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent
d=json.loads((r/'MANIFEST.json').read_text(encoding='utf-8'))
for f in d['files']:
 p=r/f['path']
 assert p.is_file() and p.stat().st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'],f['path']
print('MANIFEST_VERIFIED',len(d['files']))
