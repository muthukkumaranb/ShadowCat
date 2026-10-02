import os
import urllib.request
import urllib.error
import time

base_url = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV/MachineLearningCVE/"

files = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]

out_dir = r"d:\sih2026\data-engineering\data\raw\CIC-IDS2017"
os.makedirs(out_dir, exist_ok=True)

for fname in files:
    url = base_url + fname
    out_file = os.path.join(out_dir, fname)
    if os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
        print(f"{fname} already exists.")
        continue

    print(f"Downloading {fname}...")
    try:
        urllib.request.urlretrieve(url, out_file)
        print(f"Downloaded {fname}")
    except Exception as e:
        print(f"Failed to download {fname}: {e}")
