"""
Cut a time slice out of a real CSE-CIC-IDS2018 capture, for uploading to the dashboard.

The raw captures are downloaded with data-engineering/scripts/download_pcap_<day>.py (several GB each).
The dashboard takes the slice as an upload: PCAP -> flows + packet features -> predict().

Times are on the CAPTURE clock (UTC epoch in the PCAP). For 14-02-2018 the capture clock is the
CIC-IDS2018 CSV time + 16 h before 10:00 CSV time (see src/pcap_extractor.py). In the CSV labels, SSH-Bruteforce
starts at 02:00 (capture clock 18:00) and 04:00-05:30 has no labelled attack (capture clock 20:00-21:30).

    # SSH-Bruteforce demo: 30 min of benign history, then the attack onset (14-02-2018 victim capture)
    python data-engineering/scripts/make_pcap_demo_slice.py --preset ssh
    # benign demo: 45 min with no labelled attack
    python data-engineering/scripts/make_pcap_demo_slice.py --preset benign
    # any window
    python data-engineering/scripts/make_pcap_demo_slice.py --pcap path/to/capture.pcap \\
        --start "2018-02-14 18:10" --end "2018-02-14 19:20" --out slice.pcap
"""
from __future__ import annotations

import argparse
import struct
from datetime import datetime, timezone
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw_pcap"
PRESETS = {
    # name: (capture file, start, end) on the capture clock, UTC
    "ssh": (RAW / "UCAP172.31.69.25.pcap", "2018-02-14 17:30", "2018-02-14 18:45"),
    "benign": (RAW / "UCAP172.31.69.25.pcap", "2018-02-14 20:00", "2018-02-14 20:45"),
}


def _epoch(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp()


def slice_pcap(src: Path, dst: Path, start: float, end: float) -> int:
    """Stream-copy packets with start <= ts < end (classic libpcap format, any endianness)."""
    n = 0
    with open(src, "rb") as f, open(dst, "wb") as out:
        header = f.read(24)
        magic = header[:4]
        if magic in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"):
            e = "<"
        elif magic in (b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"):
            e = ">"
        else:
            raise SystemExit(f"{src} is not a classic libpcap file")
        nano = magic in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d")
        out.write(header)
        while True:
            rec = f.read(16)
            if len(rec) < 16:
                break
            sec, frac, incl, _ = struct.unpack(f"{e}IIII", rec)
            data = f.read(incl)
            ts = sec + frac / (1e9 if nano else 1e6)
            if ts >= end:
                break
            if ts >= start:
                out.write(rec)
                out.write(data)
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--pcap")
    ap.add_argument("--start", help='capture-clock UTC, "YYYY-MM-DD HH:MM"')
    ap.add_argument("--end", help='capture-clock UTC, "YYYY-MM-DD HH:MM"')
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.preset:
        src, start, end = PRESETS[a.preset]
        out = Path(a.out or RAW / f"demo_{a.preset}_slice.pcap")
    else:
        if not (a.pcap and a.start and a.end):
            ap.error("give --preset, or --pcap --start --end")
        src, start, end, out = Path(a.pcap), a.start, a.end, Path(a.out or "slice.pcap")
    if not Path(src).exists():
        raise SystemExit(f"{src} not found. Download it first with data-engineering/scripts/download_pcap_14022018.py")
    n = slice_pcap(Path(src), out, _epoch(start), _epoch(end))
    print(f"wrote {out} ({n:,} packets, {out.stat().st_size / 1e6:.1f} MB, {start} -> {end} capture-clock UTC)")


if __name__ == "__main__":
    main()
