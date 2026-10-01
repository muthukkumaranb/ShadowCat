import time
import sys
import os

LOG_FILE = r"C:\Users\MUTHUKUMARAN\.gemini\antigravity-ide\brain\6bf0770c-fc59-49ec-9fa1-be17dfac5f89\.system_generated\tasks\task-534.log"

def draw_progress_bar(percent, bar_len=50, prefix="", suffix=""):
    filled_len = int(round(bar_len * percent / 100))
    bar = '█' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}')
    sys.stdout.flush()

def tail_log():
    print("Monitoring pipeline progress...\n")
    if not os.path.exists(LOG_FILE):
        print("Log file not found yet. Waiting...")
        while not os.path.exists(LOG_FILE):
            time.sleep(1)
            
    with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        f.seek(0, 2)
        
        current_file = 0
        total_files = 10
        current_stage = ""
        overall_phase = "Initializing..."
        
        # Determine current state by reading the whole file first
        f.seek(0, 0)
        lines = f.readlines()
        for line in lines:
            if "[*" in line and "Processing" in line and "dataset files" in line:
                pass
            elif line.startswith("[") and "/10] Processing:" in line:
                current_file = int(line.split("/")[0].replace("[", ""))
                overall_phase = f"Processing CSV {current_file}/{total_files}"
            elif "-> Stage" in line:
                current_stage = line.strip().split(":")[0].replace("-> ", "")
            elif "[*] Assembling complete multi-day" in line:
                overall_phase = "Global Assembly (Pandas Concat)"
            elif "[*] Generating" in line:
                overall_phase = "Generating Forecast Labels"
            elif "[*] Assigning strict chronological" in line:
                overall_phase = "Splitting Train/Val/Test"
            elif "[*] Stage 6b:" in line:
                overall_phase = "Purge + Embargo (Leakage Protection)"
            elif "[*] Merging PCAP" in line:
                overall_phase = "PCAP Packet Merge"
            elif "[*] Assigning contiguous episode IDs" in line:
                overall_phase = "Episode Segmentation"
            elif "[*] Stage 7:" in line:
                overall_phase = "Fitting Robust Scaler"
            elif "[*] Stage 8:" in line:
                overall_phase = "Building LSTM Sequences"
            elif "[SUCCESS]" in line:
                overall_phase = "COMPLETED"
        
        while True:
            where = f.tell()
            line = f.readline()
            if not line:
                time.sleep(0.5)
                f.seek(where)
                
                # Calculate progress
                if overall_phase == "COMPLETED":
                    progress = 100.0
                elif "Processing CSV" in overall_phase:
                    progress = (current_file - 1) / total_files * 60.0
                    if current_stage == "Stage 1": progress += 1
                    elif current_stage == "Stage 2": progress += 2
                    elif current_stage == "Stage 3": progress += 3
                    elif current_stage == "Stage 4": progress += 4
                    elif current_stage == "Stage 5": progress += 5
                    elif current_stage == "Stage 6": progress += 6
                else:
                    # After the 10 files (which is 60% of time), the remaining 40% is the global phases
                    base = 60.0
                    if "Global Assembly" in overall_phase: base += 5
                    elif "Forecast Labels" in overall_phase: base += 10
                    elif "Splitting" in overall_phase: base += 15
                    elif "Purge" in overall_phase: base += 20
                    elif "PCAP" in overall_phase: base += 25
                    elif "Episode" in overall_phase: base += 30
                    elif "Robust Scaler" in overall_phase: base += 35
                    elif "LSTM" in overall_phase: base += 38
                    progress = base
                
                draw_progress_bar(progress, prefix=f"{overall_phase:<40}")
                if overall_phase == "COMPLETED":
                    print("\n\nPipeline finished completely!")
                    break
            else:
                # Update state
                if line.startswith("[") and "/10] Processing:" in line:
                    current_file = int(line.split("/")[0].replace("[", ""))
                    overall_phase = f"Processing CSV {current_file}/{total_files}"
                elif "-> Stage" in line:
                    current_stage = line.strip().split(":")[0].replace("-> ", "")
                elif "[*] Assembling complete multi-day" in line:
                    overall_phase = "Global Assembly (Pandas Concat)"
                elif "[*] Generating" in line:
                    overall_phase = "Generating Forecast Labels"
                elif "[*] Assigning strict chronological" in line:
                    overall_phase = "Splitting Train/Val/Test"
                elif "[*] Stage 6b:" in line:
                    overall_phase = "Purge + Embargo (Leakage Protection)"
                elif "[*] Merging PCAP" in line:
                    overall_phase = "PCAP Packet Merge"
                elif "[*] Assigning contiguous episode IDs" in line:
                    overall_phase = "Episode Segmentation"
                elif "[*] Stage 7:" in line:
                    overall_phase = "Fitting Robust Scaler"
                elif "[*] Stage 8:" in line:
                    overall_phase = "Building LSTM Sequences"
                elif "[SUCCESS]" in line:
                    overall_phase = "COMPLETED"

if __name__ == "__main__":
    try:
        tail_log()
    except KeyboardInterrupt:
        print("\nExiting watcher.")
