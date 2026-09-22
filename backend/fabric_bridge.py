import subprocess
import json
import os
import tempfile
import sys

# The base directory where the fabric experiment resides on the Windows host
FABRIC_DIR = r"D:\sih2026\fabric-experiment"

def _run_fabric_command(fcn, args):
    """
    Executes a chaincode command using a temporary bash script mapped into an ephemeral Ubuntu docker container.
    This avoids the missing 'cli' container and powershell escaping madness.
    """
    
    # Construct the JSON arguments for the chaincode
    chaincode_args = {"function": fcn, "Args": args}
    args_json = json.dumps(chaincode_args) 

    # The shell script that will be executed inside the container
    script_content = f"""#!/bin/bash
cd /d/sih2026/fabric-experiment/fabric-samples/test-network
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
    # We use --add-host to map the container names back to the host, as we did in deployment
    cmd = [
        "docker", "run", "--rm", 
        "--add-host", "orderer.example.com:host-gateway",
        "--add-host", "peer0.org1.example.com:host-gateway",
        "--add-host", "peer0.org2.example.com:host-gateway",
        "-v", r"d:\sih2026:/d/sih2026",
        "ubuntu", "sh", "-c",
        "sed -i 's/\r$//' /d/sih2026/fabric-experiment/invoke_temp.sh && bash /d/sih2026/fabric-experiment/invoke_temp.sh"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        return True, result.stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception) as e:
        err_msg = getattr(e, "stderr", None) or str(e)
        return False, err_msg

def _query_fabric_command(fcn, args):
    """
    Executes a chaincode query.
    """
    chaincode_args = {"function": fcn, "Args": args}
    args_json = json.dumps(chaincode_args)

    script_content = f"""#!/bin/bash
cd /d/sih2026/fabric-experiment/fabric-samples/test-network
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
        "-v", r"d:\sih2026:/d/sih2026",
        "ubuntu", "sh", "-c",
        "sed -i 's/\r$//' /d/sih2026/fabric-experiment/query_temp.sh && bash /d/sih2026/fabric-experiment/query_temp.sh"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
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

def query_alert(alert_hash: str):
    print(f"[*] Querying alert from Fabric: {alert_hash}")
    success, out = _query_fabric_command("QueryAlert", [alert_hash])
    if success:
        print(f"[+] Alert query successful: {out.strip()}")
        return json.loads(out.strip())
    else:
        print(f"[-] Failed to query alert. Error:\n{out}")
        return None
