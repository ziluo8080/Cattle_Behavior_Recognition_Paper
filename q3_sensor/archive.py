"""Stored-ZIP timestamp indexing; decode only authorized TRAIN/VAL CSV ranges.

Index probes receive opaque neighboring payload bytes, but interpret timestamp
prefixes only. Partial extraction does not verify full-member CRC or archive SHA.
"""
import hashlib
import io
import math
import struct
import zipfile

import pandas as pd

from q3_rgb.remote_zip import SplitRemote

REVISION = '532a193d5fd90ed6cfb2d640923c4358ceca00b4'
URL = f'https://huggingface.co/datasets/neis-lab/mmcows/resolve/{REVISION}/sensor_data.zip'
SIZE = 18130099946


class SensorRemote(SplitRemote):
    def __init__(self):
        super().__init__([dict(url=URL, size=SIZE)])
        self.receipts = []

    def _fetch(self, part, start, length):
        data = super()._fetch(part, start, length)
        self.receipts.append(dict(start=start, bytes=length, sha256=hashlib.sha256(data).hexdigest()))
        return data


def member_name(cow, kind):
    tag, name = f'T{cow:02d}', f'C{cow:02d}'
    suffix = {'uwb': f'main_data/uwb/{tag}/{tag}_0725.csv',
              'hd': f'sub_data/head_direction/{tag}/{tag}_0725.csv',
              'akl': f'main_data/ankle/{name}/{name}_0725.csv',
              'labels': f'behavior_labels/individual/{name}_0725.csv'}[kind]
    return 'sensor_data/' + suffix


class StoredCSV:
    BLOCK = 4096

    def __init__(self, remote, info):
        self.remote, self.info, self.cache = remote, info, {}
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError('Scoped random access requires stored CSV')
        local = remote.read_at(info.header_offset, 30)
        if local[:4] != b'PK\x03\x04':
            raise ValueError('Invalid ZIP local header')
        name_len, extra_len = struct.unpack_from('<HH', local, 26)
        self.base = info.header_offset + 30 + name_len + extra_len
        self.size = info.file_size
        header = self.block(0).split(b'\n', 1)[0]
        self.header = header.decode('utf-8-sig').strip()
        if self.header.split(',')[0] != 'timestamp':
            raise ValueError('Timestamp must be first CSV field')

    def block(self, k):
        if k not in self.cache:
            p = k * self.BLOCK
            self.cache[k] = self.remote.read_at(self.base + p, min(self.BLOCK, self.size - p))
        return self.cache[k]

    @staticmethod
    def records(data, origin, skip_partial):
        cursor = data.find(b'\n') + 1 if skip_partial else 0
        while cursor < len(data):
            end = data.find(b'\n', cursor)
            if end < 0:
                # EOF row without newline is handled by lower_bound's EOF join.
                break
            prefix = data[cursor:end].split(b',', 1)[0].strip()
            if prefix != b'timestamp' and prefix:
                yield origin + cursor, float(prefix), origin + end + 1
            cursor = end + 1

    def lower_bound(self, second):
        count = math.ceil(self.size / self.BLOCK)
        # Interpolate on timestamp-only probes; every final boundary is still
        # found by exact CSV line timestamps, not by assumed sampling rate.
        lo, hi = 0, count - 1
        first_rows = list(self.records(self.block(0), 0, False))
        last_rows = list(self.records(self.block(hi), hi * self.BLOCK, hi != 0))
        if not last_rows and hi:
            last_rows = list(self.records(self.block(hi - 1), (hi - 1) * self.BLOCK, hi - 1 != 0))
        tlo, thi = first_rows[0][1], last_rows[-1][1]
        while lo <= hi:
            fraction = (second - tlo) / (thi - tlo) if thi > tlo else .5
            mid = max(lo, min(hi, lo + int((hi - lo) * fraction)))
            rows = list(self.records(self.block(mid), mid * self.BLOCK, mid != 0))
            if not rows:
                hi = mid - 1
            elif second < rows[0][1]:
                hi, thi = mid - 1, rows[0][1]
            elif second > rows[-1][1]:
                lo, tlo = mid + 1, rows[-1][1]
            else:
                lo = mid
                break
        start, end = max(0, lo - 1), min(count, lo + 2)
        joined = b''.join(self.block(k) for k in range(start, end))
        if end == count and not joined.endswith(b'\n'):
            joined += b'\n'
        rows = list(self.records(joined, start * self.BLOCK, start != 0))
        hits = [(p, t) for p, t, e in rows if t >= second]
        if hits:
            pos, following = hits[0]
            previous = [t for p, t, e in rows if p < pos]
            return pos, (previous[-1] if previous else None), following
        if end != count:
            raise ValueError('Timestamp boundary not found')
        return self.size, (rows[-1][1] if rows else None), None

    def extract(self, segments):
        frames, boundaries = [], []
        for s in segments:
            p, before, first = self.lower_bound(s['start'])
            q, last, after = self.lower_bound(s['end'])
            chunks = [self.remote.read_at(self.base + at, min(q - at, 4 * 1024 * 1024))
                      for at in range(p, q, 4 * 1024 * 1024)]
            payload = b''.join(chunks)
            # Validate timestamp scope before pandas decodes any other field.
            stamps = [float(line.split(b',', 1)[0]) for line in payload.splitlines() if line]
            if any(not s['start'] <= t < s['end'] for t in stamps):
                raise ValueError('Out-of-scope row before scientific field decoding')
            if any(a > b for a, b in zip(stamps, stamps[1:])):
                raise ValueError('Non-monotonic source')
            frame = pd.read_csv(io.BytesIO(self.header.encode() + b'\n' + payload))
            frames.append(frame)
            boundaries.append(dict(**s, csv_start=p, csv_end=q,
                                   outside_before=before, outside_after=after, rows=len(frame)))
        return pd.concat(frames, ignore_index=True), boundaries


def zip_catalog(remote):
    with zipfile.ZipFile(remote) as archive:
        return {i.filename: i for i in archive.infolist()}
