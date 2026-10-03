import os
import json
import pytest
import shutil
import tempfile
from pathlib import Path
from backend.audit_chain import append_entry, verify_chain, CHAIN_PATH, _save_chain, _load_chain

@pytest.fixture
def temp_env():
    # Setup temp dir inside repo to avoid cross-drive relative path issues
    temp_dir = tempfile.mkdtemp(dir=os.path.join(os.path.dirname(__file__)))
    
    # Store old chain path
    old_chain = None
    if os.path.exists(CHAIN_PATH):
        with open(CHAIN_PATH, "r") as f:
            old_chain = f.read()
            
    # Setup test chain path
    test_chain = os.path.join(temp_dir, "test_audit_chain.json")
    
    # Setup keys
    test_keys_dir = os.path.join(temp_dir, "keys")
    os.makedirs(test_keys_dir, exist_ok=True)
    priv_key_path = os.path.join(test_keys_dir, "test_private_key.pem")
    pub_key_path = os.path.join(temp_dir, "test_public_key.pem")
    
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization
    
    priv_key = ed25519.Ed25519PrivateKey.generate()
    pub_key = priv_key.public_key()
    
    with open(priv_key_path, "wb") as f:
        f.write(priv_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
        
    with open(pub_key_path, "wb") as f:
        f.write(pub_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))
        
    os.environ["SHADOWCAT_SIGNING_KEY"] = priv_key_path
    
    # Copy public key to backend where it's expected
    import backend.audit_chain
    real_pub_key = os.path.join(os.path.dirname(backend.audit_chain.__file__), "admin_public_key.pem")
    old_pub_key = None
    if os.path.exists(real_pub_key):
        with open(real_pub_key, "rb") as f:
            old_pub_key = f.read()
    shutil.copy(pub_key_path, real_pub_key)
    
    yield test_chain, temp_dir
    
    # Teardown
    if old_chain is not None:
        with open(CHAIN_PATH, "w") as f:
            f.write(old_chain)
    elif os.path.exists(CHAIN_PATH):
        os.remove(CHAIN_PATH)
        
    if old_pub_key is not None:
        with open(real_pub_key, "wb") as f:
            f.write(old_pub_key)
    else:
        if os.path.exists(real_pub_key):
            os.remove(real_pub_key)
            
    shutil.rmtree(temp_dir)
    del os.environ["SHADOWCAT_SIGNING_KEY"]


def test_clean_chain_verifies(temp_env):
    test_chain, temp_dir = temp_env
    
    # Create a fake forecast file
    forecast_path = os.path.join(temp_dir, "forecast_1.json")
    with open(forecast_path, "w") as f:
        json.dump({"test": "data"}, f)
        
    # Append
    append_entry(forecast_path, "forecast_payload", "Test forecast 1", chain_path=test_chain)
    
    is_valid, issues = verify_chain(chain_path=test_chain)
    assert is_valid, f"Chain should be valid, but got issues: {issues}"


def test_tampering_fails(temp_env):
    test_chain, temp_dir = temp_env
    
    forecast_path = os.path.join(temp_dir, "forecast_1.json")
    with open(forecast_path, "w") as f:
        json.dump({"test": "data"}, f)
        
    append_entry(forecast_path, "forecast_payload", "Test forecast 1", chain_path=test_chain)
    
    # Tamper with the file
    with open(forecast_path, "w") as f:
        json.dump({"test": "tampered"}, f)
        
    is_valid, issues = verify_chain(chain_path=test_chain)
    assert not is_valid, "Chain should be invalid after tampering with the forecast file"
    assert any("artifact has been modified since chaining" in issue for issue in issues)
