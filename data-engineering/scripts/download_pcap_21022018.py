"""
Download and Extract Wednesday-21-02-2018 PCAPs from S3
Downloads all UCAP captures for victim 172.31.69.28 (DDoS target) and auxiliary endpoints.
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
    "Original%20Network%20Traffic%20and%20Log%20data/Wednesday-21-02-2018/pcap.zip"
)
DEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw_pcap", "21022018"))

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
    info = zf.getinfo(member_name)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) == info.file_size:
        print(f"[+] Already downloaded ({os.path.getsize(dest_path)/(1024*1024):.2f} MB): {dest_path}")
        return dest_path

    header_offset = info.header_offset
    comp_size = info.compress_size
    uncomp_size = info.file_size
    
    print(f"[*] Downloading {member_name} ({comp_size/(1024*1024):.2f} MB compressed -> {uncomp_size/(1024*1024):.2f} MB)...", flush=True)

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
    downloaded_bytes = 0
    written_bytes = 0
    t0 = time.time()
    last_log = t0

    temp_path = dest_path + ".tmp"
    with urllib.request.urlopen(stream_req, timeout=120) as resp, open(temp_path, "wb") as out_fp:
        while True:
            chunk = resp.read(2 * 1024 * 1024)
            if not chunk: break
            downloaded_bytes += len(chunk)
            uncomp = decompressor.decompress(chunk)
            if uncomp:
                out_fp.write(uncomp)
                written_bytes += len(uncomp)
            
            now = time.time()
            if now - last_log >= 5.0 or downloaded_bytes == comp_size:
                speed = (downloaded_bytes / (1024*1024)) / max(1e-5, now - t0)
                pct = (downloaded_bytes / comp_size) * 100
                print(f"    {os.path.basename(dest_path)}: {pct:.1f}% ({downloaded_bytes/(1024*1024):.1f}/{comp_size/(1024*1024):.1f} MB) @ {speed:.2f} MB/s | Written: {written_bytes/(1024*1024):.1f} MB", flush=True)
                last_log = now

        tail = decompressor.flush()
        if tail:
            out_fp.write(tail)
            written_bytes += len(tail)

    if os.path.exists(dest_path):
        os.remove(dest_path)
    os.rename(temp_path, dest_path)
    elapsed = time.time() - t0
    print(f"[+] Complete {os.path.basename(dest_path)} in {elapsed:.1f}s ({os.path.getsize(dest_path)/(1024*1024):.2f} MB)", flush=True)
    return dest_path

def main():
    print(f"[*] Reading central directory from {S3_PCAP_ZIP_URL}...")
    rz = io.BufferedReader(RemoteCentralDirectory(S3_PCAP_ZIP_URL))
    zf = zipfile.ZipFile(rz)
    
    ucap_members = sorted([m.filename for m in zf.infolist() if m.filename.startswith("pcap/UCAP")])
    print(f"[*] Found {len(ucap_members)} UCAP files for 21-02-2018:")
    for m in ucap_members:
        info = zf.getinfo(m)
        print(f"    - {m} (comp={info.compress_size/(1024*1024):.2f} MB, uncomp={info.file_size/(1024*1024):.2f} MB)")

    total_start = time.time()
    for m in ucap_members:
        clean_name = os.path.basename(m).replace(" ", "_") + ".pcap"
        dest_file = os.path.join(DEST_DIR, clean_name)
        download_member(zf, m, dest_file)

    total_elapsed = time.time() - total_start
    print(f"\n[+] All 21-02-2018 PCAPs successfully downloaded and decompressed in {total_elapsed:.1f}s to: {DEST_DIR}")

if __name__ == "__main__":
    main()
