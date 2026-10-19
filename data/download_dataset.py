"""Download and verify the exact 40 CSV files used in this study (standard library)."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import tarfile
import urllib.request

REPO = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', type=Path, help='Use an already downloaded archive')
    ap.add_argument('--dest', type=Path, default=REPO / 'data/extracted')
    ap.add_argument('--verify-only', action='store_true', help='Verify archive and CSVs without extraction')
    args = ap.parse_args()
    asset = json.loads((REPO / 'data/DATASET_ASSET.json').read_text(encoding='utf-8'))
    source = json.loads((REPO / 'evidence/RAW_SOURCE_MEMBER_INDEX.json').read_text(encoding='utf-8'))
    expected = {r['member']: r for r in source['members']}
    archive = args.archive or REPO / 'data/downloads' / asset['name']
    if not archive.exists():
        if args.archive:
            raise FileNotFoundError(archive)
        archive.parent.mkdir(parents=True, exist_ok=True)
        part = archive.with_suffix(archive.suffix + '.partial')
        with urllib.request.urlopen(asset['url'], timeout=90) as response, part.open('wb') as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        part.replace(archive)
    h = hashlib.sha256()
    with archive.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    if archive.stat().st_size != asset['bytes'] or h.hexdigest() != asset['sha256']:
        raise ValueError('Archive size or SHA256 differs from the published dataset manifest')
    seen = set()
    destination = args.dest.resolve()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if member.name not in expected or member.name in seen or not member.isfile():
                raise ValueError('Unexpected, duplicate, or non-file archive member: ' + member.name)
            rel = PurePosixPath(member.name)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe archive member')
            target = destination.joinpath(*rel.parts).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Extraction would leave the destination directory')
            record = expected[member.name]
            if member.size != record['bytes']:
                raise ValueError('CSV size differs: ' + member.name)
            if not args.verify_only and target.exists():
                if target.stat().st_size != record['bytes'] or hashlib.sha256(target.read_bytes()).hexdigest() != record['sha256']:
                    raise FileExistsError('Refusing to overwrite a changed CSV: ' + str(target))
            part = None
            output = None
            if not args.verify_only and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                part = target.with_name(target.name + '.partial-' + str(os.getpid()))
                output = part.open('xb')
            digest = hashlib.sha256()
            try:
                payload = tar.extractfile(member)
                while chunk := payload.read(1024 * 1024):
                    digest.update(chunk)
                    if output:
                        output.write(chunk)
            finally:
                if output:
                    output.close()
            if digest.hexdigest() != record['sha256']:
                if part:
                    part.unlink(missing_ok=True)
                raise ValueError('CSV SHA256 differs: ' + member.name)
            if part:
                part.replace(target)
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError('Archive is missing expected CSV files')
    print(json.dumps({'status': 'PASS', 'CSV_files': len(seen), 'archive_sha256': h.hexdigest(),
                      'extracted': not args.verify_only, 'sensor_data_root': str(destination / 'sensor_data')}, ensure_ascii=False))

if __name__ == '__main__':
    main()
