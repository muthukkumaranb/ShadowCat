import io
import struct
import pandas as pd

pcap_path = 'data-engineering/data/raw_pcap/UCAP172.31.69.25.pcap'
ssh_min, ssh_max, ftp_min, ftp_max = float('inf'), float('-inf'), float('inf'), float('-inf')
ssh_cnt, ftp_cnt, tot_cnt = 0, 0, 0
overall_min, overall_max = float('inf'), float('-inf')

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
        pkt_ts = ts_sec + (ts_usec / 1e6)
        pkt_data = fp.read(incl_len)
        tot_cnt += 1
        if pkt_ts < overall_min: overall_min = pkt_ts
        if pkt_ts > overall_max: overall_max = pkt_ts
        
        if len(pkt_data) < 34 or struct.unpack('>H', pkt_data[12:14])[0] != 0x0800:
            continue
        ip_data = pkt_data[14:]
        ihl = (ip_data[0] & 0x0F) * 4
        proto = ip_data[9]
        if proto == 6 and len(ip_data) >= ihl + 4:
            sport, dport = struct.unpack('>HH', ip_data[ihl:ihl+4])
            if dport == 22 or sport == 22:
                ssh_cnt += 1
                if pkt_ts < ssh_min: ssh_min = pkt_ts
                if pkt_ts > ssh_max: ssh_max = pkt_ts
            elif dport == 21 or sport == 21:
                ftp_cnt += 1
                if pkt_ts < ftp_min: ftp_min = pkt_ts
                if pkt_ts > ftp_max: ftp_max = pkt_ts

print(f"Total packets: {tot_cnt:,}", flush=True)
print(f"Overall Time Range: {pd.to_datetime(overall_min, unit='s', utc=True)} to {pd.to_datetime(overall_max, unit='s', utc=True)}", flush=True)
print(f"SSH Packets (Port 22): {ssh_cnt:,} from {pd.to_datetime(ssh_min, unit='s', utc=True)} to {pd.to_datetime(ssh_max, unit='s', utc=True)}", flush=True)
print(f"FTP Packets (Port 21): {ftp_cnt:,} from {pd.to_datetime(ftp_min, unit='s', utc=True)} to {pd.to_datetime(ftp_max, unit='s', utc=True)}", flush=True)
