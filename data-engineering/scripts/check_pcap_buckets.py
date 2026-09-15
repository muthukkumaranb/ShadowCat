import io
import struct
from datetime import datetime, timezone

pcap_path = 'data-engineering/data/raw_pcap/UCAP172.31.69.25.pcap'
buckets = {}

with open(pcap_path, 'rb') as raw_fp:
    fp = io.BufferedReader(raw_fp, buffer_size=32*1024*1024)
    ghdr = fp.read(24)
    magic = struct.unpack('<I', ghdr[:4])[0]
    endian = '<' if magic == 0xa1b2c3d4 else '>'
    while True:
        phdr = fp.read(16)
        if len(phdr) < 16:
            break
        ts_sec, ts_usec, incl_len, _ = struct.unpack(f'{endian}IIII', phdr)
        pkt_data = fp.read(incl_len)
        if len(pkt_data) < 34 or struct.unpack('>H', pkt_data[12:14])[0] != 0x0800:
            continue
        ip_data = pkt_data[14:]
        ihl = (ip_data[0] & 0x0F) * 4
        proto = ip_data[9]
        if proto == 6 and len(ip_data) >= ihl + 4:
            sport, dport = struct.unpack('>HH', ip_data[ihl:ihl+4])
            b_ts = (ts_sec // 600) * 600  # 10-minute integer bucket
            if b_ts not in buckets:
                buckets[b_ts] = [0, 0, 0]  # ssh, ftp, other
            if dport == 22 or sport == 22:
                buckets[b_ts][0] += 1
            elif dport == 21 or sport == 21:
                buckets[b_ts][1] += 1
            else:
                buckets[b_ts][2] += 1

print("10-Minute PCAP Buckets:")
for b_ts in sorted(buckets):
    dt_str = datetime.fromtimestamp(b_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    ssh_c, ftp_c, oth_c = buckets[b_ts]
    print(f"{dt_str} -> SSH: {ssh_c:>8,d} | FTP: {ftp_c:>8,d} | Other: {oth_c:>6,d}")
