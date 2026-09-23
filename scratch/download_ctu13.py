import urllib.request
import os

url = "https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/9/capture20110817.binetflow"
output_path = "d:/sih2026/data/ctu13_scenario9_slice.csv"
print(f"Downloading slice from {url}...")
try:
    with urllib.request.urlopen(url) as response, open(output_path, 'wb') as out_file:
        for i in range(5000):
            line = response.readline()
            if not line:
                break
            out_file.write(line)
    print(f"Downloaded 5000 lines. Size: {os.path.getsize(output_path)} bytes.")
except Exception as e:
    print(f"Failed: {e}")
