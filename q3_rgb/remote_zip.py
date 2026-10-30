"""Read a byte-split ZIP through validated HTTP ranges, without full download."""
import io
import time
import urllib.request
import zipfile
import struct
import zlib


class SplitRemote(io.RawIOBase):
    def __init__(self, parts):
        super().__init__()
        self.parts = parts
        self.size = sum(p['size'] for p in parts)
        self.pos = 0

    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset
        if self.pos < 0: raise ValueError('Negative seek')
        return self.pos

    def _fetch(self, part, start, length):
        for attempt in range(5):
            try:
                url = part['url']+'?q3_range=%d_%d'%(start,length)
                req = urllib.request.Request(url, headers={'Range':'bytes=%d-%d'%(start,start+length-1)})
                with urllib.request.urlopen(req, timeout=45) as r:
                    expected = 'bytes %d-%d/%d'%(start,start+length-1,part['size'])
                    if r.status != 206 or r.headers.get('Content-Range') != expected:
                        raise RuntimeError('Wrong HTTP range: '+str(r.headers.get('Content-Range')))
                    data = r.read(length+1)
                    if len(data) != length: raise RuntimeError('Incomplete HTTP range')
                    return data
            except Exception:
                if attempt == 4: raise
                time.sleep(min(attempt+1,4))

    def read_at(self, start, length):
        length = min(length, self.size-start)
        if length <= 0: return b''
        chunks = []
        offset = 0
        for part in self.parts:
            if start < offset+part['size'] and length:
                local = max(0,start-offset)
                take = min(length,part['size']-local)
                while take:
                    n = min(take,4*1024*1024)
                    chunks.append(self._fetch(part,local,n))
                    local += n; start += n; take -= n; length -= n
            offset += part['size']
        if length: raise ValueError('Read past archive boundary')
        return b''.join(chunks)

    def read(self, n=-1):
        n = self.size-self.pos if n < 0 else n
        data = self.read_at(self.pos,n)
        self.pos += len(data)
        return data

    def extract_member(self, info):
        header = self.read_at(info.header_offset,30)
        fields = struct.unpack('<IHHHHHIIIHH',header)
        if fields[0] != 0x04034b50 or fields[2]&1:
            raise ValueError('Invalid or encrypted ZIP local header')
        offset = info.header_offset+30+fields[-2]+fields[-1]
        data = self.read_at(offset,info.compress_size)
        if info.compress_type == zipfile.ZIP_DEFLATED: data = zlib.decompress(data,-15)
        elif info.compress_type != zipfile.ZIP_STORED: raise ValueError('Unsupported compression')
        if len(data) != info.file_size or zlib.crc32(data)&0xffffffff != info.CRC:
            raise ValueError('ZIP member CRC mismatch: '+info.filename)
        return data
