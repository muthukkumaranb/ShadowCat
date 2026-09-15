"""
High-Speed Direct Range Download and Extraction for 14-02-2018 PCAP
SIH26153 - Cyber World Model Architecture (Data Engineer Track)

Downloads the exact contiguous compressed byte range for 'pcap/UCAP172.31.69.25'
from the remote S3 zip archive in a single streaming HTTP connection and
decompresses directly to disk via zlib.
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
    "Original%20Network%20Traffic%20and%20Log%20data/Wednesday-14-02-2018/pcap.zip"
)
TARGET_MEMBER = "pcap/UCAP172.31.69.25"
DEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw_pcap"))
DEST_FILE = os.path.join(DEST_DIR, "UCAP172.31.69.25.pcap")


class RemoteCentralDirectory(io.RawIOBase):
    def __init__(self, url: str):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            self.size = int(resp.headers["Content-Length"])
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b) -> int:
        sz = len(b)
        if self.pos >= self.size or sz == 0:
            return 0
        end = min(self.pos + sz - 1, self.size - 1)
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def download_14_02_2018_pcap(dest_file: str = DEST_FILE) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(dest_file)), exist_ok=True)

    if os.path.exists(dest_file) and os.path.getsize(dest_file) > 100 * 1024 * 1024:
        print(f"[+] PCAP file already exists ({os.path.getsize(dest_file) / (1024*1024):.2f} MB): {dest_file}", flush=True)
        return dest_file

    print(f"[*] Reading central directory from remote zip: {S3_PCAP_ZIP_URL}", flush=True)
    rz = io.BufferedReader(RemoteCentralDirectory(S3_PCAP_ZIP_URL))
    zf = zipfile.ZipFile(rz)
    info = zf.getinfo(TARGET_MEMBER)
    
    header_offset = info.header_offset
    comp_size = info.compress_size
    uncomp_size = info.file_size
    print(f"[*] Target member found:", flush=True)
    print(f"    Name: {info.filename}", flush=True)
    print(f"    Header offset: {header_offset}", flush=True)
    print(f"    Compressed size: {comp_size / (1024*1024):.2f} MB", flush=True)
    print(f"    Uncompressed size: {uncomp_size / (1024*1024):.2f} MB", flush=True)

    # 1. Read the 30-byte local file header to find exact payload offset
    local_hdr_req = urllib.request.Request(
        S3_PCAP_ZIP_URL,
        headers={"Range": f"bytes={header_offset}-{header_offset + 1024}"}
    )
    with urllib.request.urlopen(local_hdr_req, timeout=30) as resp:
        local_hdr_data = resp.read()

    # Local file header structure:
    # 0..4: signature (0x04034b50)
    # 26..28: filename length
    # 28..30: extra field length
    sig, _, _, _, _, _, _, _, _, fn_len, ef_len = struct.unpack("<IHHHHHIIIHH", local_hdr_data[:30])
    payload_offset = header_offset + 30 + fn_len + ef_len
    payload_end = payload_offset + comp_size - 1

    print(f"[*] Exact compressed payload byte range: bytes={payload_offset}-{payload_end}", flush=True)
    print(f"[*] Starting single streaming HTTP Range download & on-the-fly decompression...", flush=True)

    start_time = time.time()
    stream_req = urllib.request.Request(
        S3_PCAP_ZIP_URL,
        headers={"Range": f"bytes={payload_offset}-{payload_end}"}
    )

    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    downloaded_bytes = 0
    written_uncompressed = 0
    last_log = time.time()

    with urllib.request.urlopen(stream_req, timeout=120) as resp, open(dest_file, "wb") as out_fp:
        while True:
            chunk = resp.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            downloaded_bytes += len(chunk)
            uncomp_chunk = decompressor.decompress(chunk)
            if uncomp_chunk:
                out_fp.write(uncomp_chunk)
                written_uncompressed += len(uncomp_chunk)

            if time.time() - last_log >= 3.0 or downloaded_bytes == comp_size:
                elapsed = time.time() - start_time
                speed = (downloaded_bytes / (1024 * 1024)) / max(1e-5, elapsed)
                pct = (downloaded_bytes / comp_size) * 100
                print(f"    Downloaded: {downloaded_bytes / (1024*1024):.1f}/{comp_size / (1024*1024):.1f} MB ({pct:.1f}%) | Written: {written_uncompressed / (1024*1024):.1f} MB @ {speed:.2f} MB/s", flush=True)
                last_log = time.time()

        tail = decompressor.flush()
        if tail:
            out_fp.write(tail)
            written_uncompressed += len(tail)

    elapsed = time.time() - start_time
    print(f"[+] Extraction complete in {elapsed:.1f}s!", flush=True)
    print(f"[+] Output PCAP saved to: {dest_file} ({os.path.getsize(dest_file) / (1024*1024):.2f} MB)", flush=True)
    return dest_file


if __name__ == "__main__":
    download_14_02_2018_pcap()
