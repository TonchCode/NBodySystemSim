#!/usr/bin/env python3
"""Convert the viewer's 24-bit bottom-up BMP screenshots to PNG (stdlib only: zlib + struct).

    python tools/bmp2png.py shot.bmp [more.bmp ...] [--delete]
Writes shot.png next to each input; --delete removes the BMP after a successful conversion.
"""
import struct, sys, zlib, os

def convert(path):
    d = open(path, 'rb').read()
    if d[:2] != b'BM':
        raise ValueError(path + ': not a BMP')
    off, = struct.unpack_from('<I', d, 10)
    w, h = struct.unpack_from('<ii', d, 18)
    bpp, = struct.unpack_from('<H', d, 28)
    if bpp != 24:
        raise ValueError(path + ': only 24-bit BMPs')
    row = (w * 3 + 3) & ~3
    raw = bytearray()
    for y in range(abs(h)):
        src = (abs(h) - 1 - y) if h > 0 else y            # bottom-up -> top-down
        r = d[off + src * row: off + src * row + w * 3]
        raw.append(0)
        px = bytearray(r)
        px[0::3], px[2::3] = r[2::3], r[0::3]               # BGR -> RGB
        raw += px
    def chunk(t, b):
        return struct.pack('>I', len(b)) + t + b + struct.pack('>I', zlib.crc32(t + b) & 0xffffffff)
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, abs(h), 8, 2, 0, 0, 0)) + \
          chunk(b'IDAT', zlib.compress(bytes(raw), 9)) + chunk(b'IEND', b'')
    out = os.path.splitext(path)[0] + '.png'
    open(out, 'wb').write(png)
    return out

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--delete']
    for p in args:
        o = convert(p)
        print(p, '->', o)
        if '--delete' in sys.argv:
            os.remove(p)
