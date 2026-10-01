import os
import urllib.request
import urllib.error
import time

days = [
    'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv',
    'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv',
    'Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv',
    'Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv',
    'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv',
    'Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv',
    'Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv',
    'Friday-23-02-2018_TrafficForML_CICFlowMeter.csv',
    'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv'
]

out_dir = r'd:\sih2026\data-engineering\data\raw\CSE-CIC-IDS2018-csv'
os.makedirs(out_dir, exist_ok=True)

for fname in days:
    url = f'https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/{fname}'
    out_file = os.path.join(out_dir, fname)
    
    if os.path.exists(out_file) and os.path.getsize(out_file) > 1000000:
        print(f'{fname} already downloaded.')
        continue

    print(f'Downloading {fname}...')
    retries = 0
    success = False
    while retries < 10 and not success:
        try:
            req = urllib.request.Request(url)
            if os.path.exists(out_file):
                current_size = os.path.getsize(out_file)
                req.add_header("Range", f"bytes={current_size}-")
                print(f"Resuming {fname} from {current_size} bytes...")
            else:
                current_size = 0
                
            with urllib.request.urlopen(req, timeout=10) as response:
                mode = 'ab' if current_size > 0 else 'wb'
                with open(out_file, mode) as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            success = True
                            break
                        f.write(chunk)
            print(f'Successfully downloaded {fname}.')
        except urllib.error.HTTPError as e:
            if e.code == 416: # Range not satisfiable (file already fully downloaded)
                success = True
                print(f'{fname} already fully downloaded.')
            else:
                print(f"HTTP Error: {e.code} - {e.reason}")
                retries += 1
                time.sleep(2)
        except Exception as e:
            print(f"Connection error: {e}. Retrying {retries+1}/10 in 5s...")
            retries += 1
            time.sleep(5)
    
    if not success:
        print(f"FAILED to download {fname} after 10 retries.")
