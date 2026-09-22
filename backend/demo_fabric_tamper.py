import time
import hashlib
import sys
import os

# Ensure backend module is accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fabric_bridge import notarize_alert, query_alert, notarize_model, _query_fabric_command

def main():
    print("=" * 60)
    print("HYPERLEDGER FABRIC TAMPER-DETECTION DEMO")
    print("=" * 60)
    print()

    # 1. Honest operation: Record a real hazard alert
    print("[1] System generates a CRITICAL hazard alert during live inference...")
    window_id = "W_DEMO_001"
    timestamp = "2026-09-18T10:00:00Z"
    true_severity = "HIGH"
    
    # Hash of the payload
    alert_str = f"{window_id}-{timestamp}-0.95"
    alert_hash = hashlib.sha256(alert_str.encode()).hexdigest()[:16]

    print(f"    -> Alert Hash: {alert_hash}")
    print(f"    -> True Severity: {true_severity}")
    
    # Notarize on Fabric
    print("[*] Notarizing alert on Fabric channel 'shadowcat-notary-channel'...")
    notarize_alert(alert_hash, true_severity, timestamp)
    
    # 2. Local Database simulation
    local_database = {
        alert_hash: {
            "severity": true_severity,
            "timestamp": timestamp,
            "notes": "Botnet C2 traffic detected."
        }
    }
    print("[+] Alert securely logged in local database AND immutable ledger.")
    print()

    # 3. The Attack
    print("[2] ATTACKER INFILTRATES LOCAL NETWORK...")
    print("    -> Attacker gains root access to the SIEM / local database.")
    print("    -> Attacker modifies the local alert to hide their tracks.")
    
    local_database[alert_hash]["severity"] = "LOW"
    local_database[alert_hash]["notes"] = "False positive. Benign traffic."
    print("    [!] Local database entry altered:")
    print(f"        {local_database[alert_hash]}")
    print()

    # 4. Auditor Verification
    print("[3] AUDITOR RUNS FABRIC VERIFICATION TOOL...")
    print(f"    [*] Querying Fabric ledger for alert hash: {alert_hash}")
    fabric_record = query_alert(alert_hash)
    
    if fabric_record:
        print(f"    [+] Record retrieved from peer0.org1.example.com")
        fabric_severity = fabric_record.get("severity")
        local_severity = local_database[alert_hash]["severity"]
        
        print(f"    -> Local Severity : {local_severity}")
        print(f"    -> Fabric Severity: {fabric_severity}")
        
        if local_severity != fabric_severity:
            print("\n    [!!!] TAMPERING DETECTED [!!!]")
            print("    The local database has been compromised!")
            print("    The immutable Hyperledger Fabric audit trail proves the original severity was HIGH.")
        else:
            print("\n    [OK] Records match.")
    else:
        print("    [-] Failed to retrieve record from Fabric.")

    print("\n=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()
