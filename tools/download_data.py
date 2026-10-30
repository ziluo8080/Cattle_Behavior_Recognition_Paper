"""Anonymous public dataset download, safe extraction and50-file SHA256 checks."""
from pathlib import Path
import argparse,hashlib,json,tarfile,shutil,urllib.request
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path);p.add_argument('--verify-only',action='store_true');p.add_argument('--output',type=Path,default=ROOT/'data/extracted');a=p.parse_args(argv)
 manifest=json.loads((ROOT/'data/DATASET_MANIFEST.json').read_text('utf8'));asset=json.loads((ROOT/'data/DATASET_ASSET.json').read_text('utf8'))
 files={r['path']:r for r in manifest['files']};assert len(files)==50
 if not a.verify_only:
  archive=a.archive or ROOT/'data'/asset['name']
  if not archive.exists():
   temp=archive.with_suffix(archive.suffix+'.part');archive.parent.mkdir(parents=True,exist_ok=True)
   with urllib.request.urlopen(asset['url'],timeout=60) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024*1024)
   assert temp.stat().st_size==asset['bytes'] and sha(temp)==asset['sha256'];temp.replace(archive)
  assert archive.stat().st_size==asset['bytes'] and sha(archive)==asset['sha256']
  a.output.mkdir(parents=True,exist_ok=True)
  with tarfile.open(archive,'r:gz') as tar:
   members=[m for m in tar.getmembers() if m.isfile()];assert {m.name for m in members}==set(files)
   for m in members:
    target=(a.output/m.name).resolve();assert target.is_relative_to(a.output.resolve())
    if target.exists():assert sha(target)==files[m.name]['sha256'];continue
    target.parent.mkdir(parents=True,exist_ok=True)
    with tar.extractfile(m) as s,target.open('wb') as d:shutil.copyfileobj(s,d,4*1024*1024)
 for name,r in files.items():
  target=a.output/name;assert target.is_file() and target.stat().st_size==r['bytes'] and sha(target)==r['sha256'],name
 print(json.dumps(dict(status='PASS_50_ORIGINAL_CSV_SHA256',files=50,root=str(a.output)),ensure_ascii=False))
if __name__=='__main__':main()
