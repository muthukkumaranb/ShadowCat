import pandas as pd


def validate(df):
    """
    Takes cleaned UCS-canonical dataframe and validates all rows.
    
    Performs 4 validation checks:
      - Check 1: timestamp must not be null
      - Check 2: required identifying fields (source_id, destination_id, protocol) must not be null
      - Check 3: no negative packet_count, byte_count, duration values
      - Check 4: source_port and destination_port must be in valid range (0-65535)
    
    Args:
        df: DataFrame with canonical UCS column names (timestamp, source_id, destination_id, 
            source_port, destination_port, protocol, packet_count, byte_count, duration)
    
    Returns:
        tuple: (df_valid, report_dict) where:
            - df_valid: DataFrame containing only rows that pass all 4 checks
            - report_dict: Dictionary with check results and summary statistics
    """
    total = len(df)
    
    # Build a boolean mask - starts as "all rows valid", then we knock out bad ones
    valid_mask = pd.Series(True, index=df.index)
    
    # Check 1: timestamp must not be null
    check1 = df["timestamp"].notnull()
    check1_pass = check1.sum()
    check1_fail = (~check1).sum()
    valid_mask &= check1
    
    # Check 2: required identifying fields must not be null
    check2 = (
        df["source_id"].notnull() & 
        df["destination_id"].notnull() & 
        df["protocol"].notnull()
    )
    check2_pass = check2.sum()
    check2_fail = (~check2).sum()
    valid_mask &= check2
    
    # Check 3: no negative packet/byte/duration values
    check3 = (
        (df["packet_count"] >= 0) & 
        (df["byte_count"] >= 0) & 
        (df["duration"] >= 0)
    )
    check3_pass = check3.sum()
    check3_fail = (~check3).sum()
    valid_mask &= check3
    
    # Check 4: ports within valid range (0-65535), where present
    check4 = (
        df["source_port"].between(0, 65535) & 
        df["destination_port"].between(0, 65535)
    )
    check4_pass = check4.sum()
    check4_fail = (~check4).sum()
    valid_mask &= check4
    
    valid_count = valid_mask.sum()
    invalid_count = total - valid_count
    
    # Build report dictionary
    report_dict = {
        "total_rows": total,
        "checks": {
            "check_1_valid_timestamp": {"pass": int(check1_pass), "fail": int(check1_fail)},
            "check_2_required_fields": {"pass": int(check2_pass), "fail": int(check2_fail)},
            "check_3_non_negative_values": {"pass": int(check3_pass), "fail": int(check3_fail)},
            "check_4_valid_port_range": {"pass": int(check4_pass), "fail": int(check4_fail)},
        },
        "summary": {
            "valid_count": int(valid_count),
            "invalid_count": int(invalid_count),
            "valid_pct": float(valid_count / total * 100) if total > 0 else 0.0,
            "invalid_pct": float(invalid_count / total * 100) if total > 0 else 0.0,
        }
    }
    
    # Return validated dataframe and report
    df_valid = df[valid_mask].reset_index(drop=True)
    return df_valid, report_dict
