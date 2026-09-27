"""
Resumable Chunked Downloader for 21-02-2018 DDOS-LOIC-UDP PCAP (part 2)
SIH26153 - Cyber World Model Architecture

Downloads the raw compressed deflate payload in short 4MB HTTP range requests with automatic retries.
Decompresses locally in seconds upon download completion.
Enforces a hard 20-minute (1200s) wall-clock timebox.
"""
import os
import sys
import io
import time
import zlib
import struct
import zipfile
import urllib.request
from typing import Optional

S3_PCAP_ZIP_URL = (
    "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/"
    "Original%20Network%20Traffic%20and%20Log%20data/Wednesday-21-02-2018/pcap.zip"
)
DEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw_pcap", "21022018"))
TIME_BOX_SECONDS = 1200  # 20 minutes hard limit

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

def get_member_payload_range(zf: zipfile.ZipFile, member_name: str):
    info = zf.getinfo(member_name)
    header_offset = info.header_offset
    comp_size = info.compress_size
    uncomp_size = info.file_size

    local_hdr_req = urllib.request.Request(
        S3_PCAP_ZIP_URL,
        headers={"Range": f"bytes={header_offset}-{header_offset + 1024}"}
    )
    with urllib.request.urlopen(local_hdr_req, timeout=30) as resp:
        local_hdr_data = resp.read()

    sig, _, _, _, _, _, _, _, _, fn_len, ef_len = struct.unpack("<IHHHHHIIIHH", local_hdr_data[:30])
    payload_offset = header_offset + 30 + fn_len + ef_len
    payload_end = payload_offset + comp_size - 1
    return payload_offset, payload_end, comp_size, uncomp_size

def download_chunk_with_retry(start_byte: int, end_byte: int, max_retries: int = 3) -> bytes:
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                S3_PCAP_ZIP_URL,
                headers={"Range": f"bytes={start_byte}-{end_byte}"}
            )
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = resp.read()
                expected = end_byte - start_byte + 1
                if len(data) == expected:
                    return data
                print(f"    [!] Short read: got {len(data)}, expected {expected}. Retrying...")
        except Exception as e:
            print(f"    [!] Attempt {attempt+1}/{max_retries} failed for bytes {start_byte}-{end_byte}: {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed to download byte range {start_byte}-{end_byte} after {max_retries} retries.")

def download_member_resumable(
    zf: zipfile.ZipFile,
    member_name: str,
    dest_pcap_path: str,
    start_wall_clock: float,
    chunk_size: int = 4 * 1024 * 1024  # 4MB chunks
) -> bool:
    os.makedirs(os.path.dirname(dest_pcap_path), exist_ok=True)
    pay_start, pay_end, comp_size, uncomp_size = get_member_payload_range(zf, member_name)
    
    if os.path.exists(dest_pcap_path) and os.path.getsize(dest_pcap_path) == uncomp_size:
        print(f"[+] Already downloaded and decompressed: {dest_pcap_path} ({uncomp_size / (1024*1024):.2f} MB)")
        return True

    deflate_path = dest_pcap_path + ".deflate"
    current_size = os.path.getsize(deflate_path) if os.path.exists(deflate_path) else 0

    print(f"[*] Member: {member_name}")
    print(f"    Compressed: {comp_size / (1024*1024):.2f} MB | Uncompressed: {uncomp_size / (1024*1024):.2f} MB")
    print(f"    Resume offset: {current_size / (1024*1024):.2f} MB ({current_size/comp_size*100:.1f}%)")

    with open(deflate_path, "ab" if current_size > 0 else "wb") as f_out:
        pos = current_size
        while pos < comp_size:
            # Check hard timebox
            elapsed = time.time() - start_wall_clock
            if elapsed >= TIME_BOX_SECONDS:
                print(f"\n[!] TIMEBOX EXCEEDED: Elapsed {elapsed:.1f}s >= {TIME_BOX_SECONDS}s. Aborting download.")
                return False

            chunk_end = min(pos + chunk_size - 1, comp_size - 1)
            raw_start = pay_start + pos
            raw_end = pay_start + chunk_end

            chunk_bytes = download_chunk_with_retry(raw_start, raw_end)
            f_out.write(chunk_bytes)
            f_out.flush()
            pos += len(chunk_bytes)

            pct = (pos / comp_size) * 100
            time_left = max(0, TIME_BOX_SECONDS - elapsed)
            speed = (pos - current_size) / (1024*1024) / max(0.1, time.time() - start_wall_clock)
            print(f"    Progress: {pct:5.1f}% ({pos/(1024*1024):.1f}/{comp_size/(1024*1024):.1f} MB) @ {speed:.2f} MB/s | Time left: {time_left:.0f}s", flush=True)

    print(f"\n[+] Compressed payload downloaded successfully. Decompressing to {dest_pcap_path}...")
    t_decomp = time.time()
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    with open(deflate_path, "rb") as f_in, open(dest_pcap_path, "wb") as f_pcap:
        while True:
            chunk = f_in.read(4 * 1024 * 1024)
            if not chunk: break
            uncomp = decompressor.decompress(chunk)
            if uncomp: f_pcap.write(uncomp)
        tail = decompressor.flush()
        if tail: f_pcap.write(tail)

    print(f"[+] Decompressed in {time.time() - t_decomp:.2f}s! Final PCAP size: {os.path.getsize(dest_pcap_path)/(1024*1024):.2f} MB")
    if os.path.exists(deflate_path):
        os.remove(deflate_path)
    return True

def main():
    start_time = time.time()
    print("=" * 70)
    print("Bounded Attempt: 21-02-2018 DDOS-LOIC-UDP PCAP (part 2)")
    print(f"Hard Wall-Clock Timebox: {TIME_BOX_SECONDS} seconds (20 minutes)")
    print("=" * 70)

    rz = io.BufferedReader(RemoteCentralDirectory(S3_PCAP_ZIP_URL))
    zf = zipfile.ZipFile(rz)

    # 1. Download part 2 (the headline DDOS-LOIC-UDP capture, 252 MB)
    p2_dest = os.path.join(DEST_DIR, "UCAP172.31.69.28_part2.pcap")
    success_p2 = download_member_resumable(
        zf=zf,
        member_name="pcap/UCAP172.31.69.28 part 2",
        dest_pcap_path=p2_dest,
        start_wall_clock=start_time,
        chunk_size=4 * 1024 * 1024
    )

    if not success_p2:
        print("[!] Step 1 failed to complete within the 20-minute timebox.")
        sys.exit(1)

    # 2. Also fetch the small UCAP172.31.69.7 (0.29 MB) to complete auxiliary servers
    p7_dest = os.path.join(DEST_DIR, "UCAP172.31.69.7.pcap")
    download_member_resumable(
        zf=zf,
        member_name="pcap/UCAP172.31.69.7",
        dest_pcap_path=p7_dest,
        start_wall_clock=start_time,
        chunk_size=1 * 1024 * 1024
    )

    total_time = time.time() - start_time
    print(f"\n[SUCCESS] Step 1 PCAP download completed in {total_time:.1f}s (< {TIME_BOX_SECONDS}s)!")
    sys.exit(0)

if __name__ == "__main__":
    main()
