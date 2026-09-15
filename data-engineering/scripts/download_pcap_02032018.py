"""
Download and Extract Friday-02-03-2018 PCAPs from S3
"""
import os
import sys
import io
import time
import zlib
import struct
import zipfile
import urllib.request

S3_PCAP_ZIP_URL = (
    "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/"
    "Original%20Network%20Traffic%20and%20Log%20data/Friday-02-03-2018/pcap.zip"
)
DEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw_pcap", "02032018"))

class RemoteCentralDirectory(io.RawIOBase):
    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            self.size = int(resp.headers["Content-Length"])
        self.pos = 0

    def readable(self) -> bool: return True
    def seekable(self) -> bool: return True
    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET: self.pos = offset
        elif whence == io.SEEK_CUR: self.pos += offset
        elif whence == io.SEEK_END: self.pos = self.size + offset
        return self.pos
    def tell(self) -> int: return self.pos
    def readinto(self, b) -> int:
        sz = len(b)
        if self.pos >= self.size or sz == 0: return 0
        end = min(self.pos + sz - 1, self.size - 1)
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=60) as resp: data = resp.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)

def download_member(zf: zipfile.ZipFile, member_name: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[+] Already exists ({os.path.getsize(dest_path)/(1024*1024):.2f} MB): {dest_path}")
        return dest_path

    info = zf.getinfo(member_name)
    header_offset = info.header_offset
    comp_size = info.compress_size
    
    # Read local header
    local_hdr_req = urllib.request.Request(
        S3_PCAP_ZIP_URL,
        headers={"Range": f"bytes={header_offset}-{header_offset + 1024}"}
    )
    with urllib.request.urlopen(local_hdr_req, timeout=30) as resp:
        local_hdr_data = resp.read()

    sig, _, _, _, _, _, _, _, _, fn_len, ef_len = struct.unpack("<IHHHHHIIIHH", local_hdr_data[:30])
    payload_offset = header_offset + 30 + fn_len + ef_len
    payload_end = payload_offset + comp_size - 1

    stream_req = urllib.request.Request(
        S3_PCAP_ZIP_URL,
        headers={"Range": f"bytes={payload_offset}-{payload_end}"}
    )
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    with urllib.request.urlopen(stream_req, timeout=120) as resp, open(dest_path, "wb") as out_fp:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk: break
            uncomp = decompressor.decompress(chunk)
            if uncomp: out_fp.write(uncomp)
        tail = decompressor.flush()
        if tail: out_fp.write(tail)
    print(f"[+] Downloaded {member_name} -> {dest_path} ({os.path.getsize(dest_path)/(1024*1024):.2f} MB)")
    return dest_path

def main():
    print(f"[*] Reading central directory from {S3_PCAP_ZIP_URL}...")
    rz = io.BufferedReader(RemoteCentralDirectory(S3_PCAP_ZIP_URL))
    zf = zipfile.ZipFile(rz)
    
    ucap_members = [m.filename for m in zf.infolist() if m.filename.startswith("pcap/UCAP")]
    print(f"[*] Found {len(ucap_members)} UCAP files for 02-03-2018:")
    for m in ucap_members:
        print(f"    - {m}")
        fname = os.path.basename(m) + ".pcap"
        download_member(zf, m, os.path.join(DEST_DIR, fname))

if __name__ == "__main__":
    main()
