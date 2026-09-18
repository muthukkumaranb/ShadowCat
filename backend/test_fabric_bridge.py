import sys
import os

# Add the backend directory to the path so we can import the bridge
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fabric_bridge import notarize_model, notarize_alert, query_alert

def main():
    print("--- Testing Fabric Bridge ---")
    
    # 1. Test Model Notarization
    model_id = "test-model-v1"
    checkpoint = "/tmp/test.ckpt"
    schema = "/tmp/schema.json"
    
    success = notarize_model(model_id, checkpoint, schema)
    if not success:
        print("Failed to notarize model. Exiting.")
        return
        
    # 2. Test Alert Notarization
    alert_hash = "abc123def456"
    severity = "HIGH"
    timestamp = "2026-09-17T12:00:00Z"
    
    success = notarize_alert(alert_hash, severity, timestamp)
    if not success:
        print("Failed to notarize alert. Exiting.")
        return
        
    # 3. Test Alert Query
    result = query_alert(alert_hash)
    if result:
        print("Query returned successfully:")
        print(result)
    else:
        print("Query failed.")
        
if __name__ == "__main__":
    main()
