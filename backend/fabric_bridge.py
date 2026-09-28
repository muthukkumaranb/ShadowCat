import subprocess
import json
import os
import tempfile
import sys
from typing import Optional, List, Dict, Any

from pathlib import Path

# The base directory where the fabric experiment resides on the Windows host
FABRIC_HOST_ROOT = os.environ.get("SHADOWCAT_FABRIC_HOST_ROOT", str(Path(__file__).resolve().parents[1]))
FABRIC_DIR = os.path.join(FABRIC_HOST_ROOT, "fabric-experiment")

def _to_container_path(host_root: str) -> str:
    """Converts 'D:\\sih2026' -> '/d/sih2026' (the existing git-bash/MSYS convention this script already assumes)."""
    drive, _, rest = host_root.partition(":")
    rest = rest.replace("\\", "/")
    return f"/{drive.lower()}{rest}"

FABRIC_CONTAINER_ROOT = _to_container_path(FABRIC_HOST_ROOT)

def _run_fabric_command(fcn, args):
    """
    Executes a chaincode command using a temporary bash script mapped into an ephemeral Ubuntu docker container.
    This avoids the missing 'cli' container and powershell escaping madness.
    """
    # Construct the JSON arguments for the chaincode
    chaincode_args = {"function": fcn, "Args": [str(a) for a in args]}
    args_json = json.dumps(chaincode_args) 

    # The shell script that will be executed inside the container
    script_content = f"""#!/bin/bash
cd {FABRIC_CONTAINER_ROOT}/fabric-experiment/fabric-samples/test-network
export PATH=$PWD/../bin:$PATH
export FABRIC_CFG_PATH=$PWD/../config/

# Set up Org1 Admin environment
export CORE_PEER_TLS_ENABLED=true
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${{PWD}}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${{PWD}}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=peer0.org1.example.com:7051

peer chaincode invoke -o orderer.example.com:7050 --ordererTLSHostnameOverride orderer.example.com --tls --cafile "${{PWD}}/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem" -C shadowcat-notary-channel -n shadowcat_notary --peerAddresses peer0.org1.example.com:7051 --tlsRootCertFiles "${{PWD}}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt" --peerAddresses peer0.org2.example.com:9051 --tlsRootCertFiles "${{PWD}}/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt" -c '{args_json}'
"""
    
    # Write the script to a file on the host
    script_path = os.path.join(FABRIC_DIR, "invoke_temp.sh")
    with open(script_path, "w", newline='\n') as f:
        f.write(script_content)

    # Convert line endings in case Windows messed them up, then run it in the container
    cmd = [
        "docker", "run", "--rm", 
        "--add-host", "orderer.example.com:host-gateway",
        "--add-host", "peer0.org1.example.com:host-gateway",
        "--add-host", "peer0.org2.example.com:host-gateway",
        "-v", f"{FABRIC_HOST_ROOT}:{FABRIC_CONTAINER_ROOT}",
        "ubuntu", "sh", "-c",
        f"sed -i 's/\\r$//' {FABRIC_CONTAINER_ROOT}/fabric-experiment/invoke_temp.sh && bash {FABRIC_CONTAINER_ROOT}/fabric-experiment/invoke_temp.sh"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        return True, result.stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception) as e:
        err_msg = getattr(e, "stderr", None) or str(e)
        return False, err_msg

def _query_fabric_command(fcn, args):
    """
    Executes a chaincode query.
    """
    chaincode_args = {"function": fcn, "Args": [str(a) for a in args]}
    args_json = json.dumps(chaincode_args)

    script_content = f"""#!/bin/bash
cd {FABRIC_CONTAINER_ROOT}/fabric-experiment/fabric-samples/test-network
export PATH=$PWD/../bin:$PATH
export FABRIC_CFG_PATH=$PWD/../config/

# Set up Org1 Admin environment
export CORE_PEER_TLS_ENABLED=true
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_TLS_ROOTCERT_FILE=${{PWD}}/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=${{PWD}}/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
export CORE_PEER_ADDRESS=peer0.org1.example.com:7051

peer chaincode query -C shadowcat-notary-channel -n shadowcat_notary -c '{args_json}'
"""
    
    script_path = os.path.join(FABRIC_DIR, "query_temp.sh")
    with open(script_path, "w", newline='\n') as f:
        f.write(script_content)

    cmd = [
        "docker", "run", "--rm", 
        "--add-host", "peer0.org1.example.com:host-gateway",
        "-v", f"{FABRIC_HOST_ROOT}:{FABRIC_CONTAINER_ROOT}",
        "ubuntu", "sh", "-c",
        f"sed -i 's/\\r$//' {FABRIC_CONTAINER_ROOT}/fabric-experiment/query_temp.sh && bash {FABRIC_CONTAINER_ROOT}/fabric-experiment/query_temp.sh"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        return True, result.stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception) as e:
        err_msg = getattr(e, "stderr", None) or str(e)
        return False, err_msg


def notarize_model(model_id: str, checkpoint_path: str, schema_path: str) -> bool:
    print(f"[*] Notarizing model provenance on Fabric: {model_id}")
    success, out = _run_fabric_command("RecordModelProvenance", [model_id, checkpoint_path, schema_path])
    if success:
        print(f"[+] Successfully notarized model: {model_id}")
    else:
        print(f"[-] Failed to notarize model. Error:\n{out}")
    return success

def notarize_alert(alert_hash: str, severity: str, timestamp: str) -> bool:
    print(f"[*] Notarizing alert on Fabric: {alert_hash}")
    success, out = _run_fabric_command("NotarizeAlert", [alert_hash, severity, timestamp])
    if success:
        print(f"[+] Successfully notarized alert: {alert_hash}")
    else:
        print(f"[-] Failed to notarize alert. Error:\n{out}")
    return success

def query_alert(alert_hash: str) -> Optional[dict]:
    print(f"[*] Querying alert from Fabric: {alert_hash}")
    success, out = _query_fabric_command("QueryAlert", [alert_hash])
    if success:
        clean = out.strip()
        print(f"[+] Alert query successful: {clean}")
        try:
            return json.loads(clean)
        except Exception:
            return None
    else:
        print(f"[-] Failed to query alert. Error:\n{out}")
        return None

def record_prediction_lineage(
    lineage_id: str,
    raw_data_hash: str,
    feature_hash: str,
    model_id: str,
    prediction_hash: str,
    severity: str,
    timestamp: str,
    target_node: str = "172.31.69.21",
) -> bool:
    """
    Writes an atomic PredictionLineage record containing all 4 stage hashes to Fabric ledger.
    If severity is HIGH or CRITICAL, the Go chaincode autonomously creates an IncidentResponseRecord
    in the same transaction.
    """
    print(f"[*] Recording prediction lineage on Fabric: {lineage_id} (severity: {severity})")
    args = [
        lineage_id,
        raw_data_hash,
        feature_hash,
        model_id,
        prediction_hash,
        severity,
        timestamp,
        target_node,
    ]
    success, out = _run_fabric_command("RecordPredictionLineageWithTarget", args)
    if success:
        print(f"[+] Successfully recorded prediction lineage: {lineage_id}")
    else:
        print(f"[-] Failed to record prediction lineage. Error:\n{out}")
    return success

def query_prediction_lineage(lineage_id: str) -> Optional[dict]:
    """
    Queries the 4-stage PredictionLineage record from Fabric.
    """
    print(f"[*] Querying prediction lineage from Fabric: {lineage_id}")
    success, out = _query_fabric_command("QueryPredictionLineage", [lineage_id])
    if success:
        clean = out.strip()
        print(f"[+] Prediction lineage query successful: {clean}")
        try:
            return json.loads(clean)
        except Exception:
            return None
    else:
        print(f"[-] Failed to query prediction lineage. Error:\n{out}")
        return None

def query_incident_response(incident_id: str) -> Optional[dict]:
    """
    Queries an auto-triggered IncidentResponseRecord from Fabric.
    """
    print(f"[*] Querying incident response from Fabric: {incident_id}")
    success, out = _query_fabric_command("QueryIncidentResponse", [incident_id])
    if success:
        clean = out.strip()
        print(f"[+] Incident response query successful: {clean}")
        try:
            return json.loads(clean)
        except Exception:
            return None
    else:
        print(f"[-] Failed to query incident response. Error:\n{out}")
        return None

def query_all_incidents() -> List[dict]:
    """
    Queries all auto-triggered IncidentResponseRecord objects from Fabric ledger.
    """
    print("[*] Querying all incident responses from Fabric...")
    success, out = _query_fabric_command("QueryAllIncidents", [])
    if success:
        clean = out.strip()
        print(f"[+] All incidents query returned: {clean}")
        if not clean or clean == "null":
            return []
        try:
            parsed = json.loads(clean)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except Exception:
            return []
    else:
        print(f"[-] Failed to query all incidents. Error:\n{out}")
        return []
