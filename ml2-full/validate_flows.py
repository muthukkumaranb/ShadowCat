"""
Standalone verification script for flow validation.
Loads cleaned flows, validates them, and saves output.

This script is used to verify validate_flows.py still works end-to-end.
The core logic is extracted to data_engineering.preprocessing.validation.validate()
for use in the pipeline.
"""
import pandas as pd
from data_engineering.preprocessing.validation import validate


def main():
    # Load cleaned flows from previous pipeline stage
    input_path = "data/intermediate/ugr16/cleaned_sorted.parquet"
    df = pd.read_parquet(input_path)
    
    total = len(df)
    print(f"Total rows loaded: {total:,}")
    print("-" * 60)
    
    # Run validation
    df_valid, report = validate(df)
    
    # Print detailed report
    print(f"Check 1 - valid timestamp: {report['checks']['check_1_valid_timestamp']['pass']:,} pass, {report['checks']['check_1_valid_timestamp']['fail']:,} fail")
    print(f"Check 2 - required fields present: {report['checks']['check_2_required_fields']['pass']:,} pass, {report['checks']['check_2_required_fields']['fail']:,} fail")
    print(f"Check 3 - no negative values: {report['checks']['check_3_non_negative_values']['pass']:,} pass, {report['checks']['check_3_non_negative_values']['fail']:,} fail")
    print(f"Check 4 - valid port range: {report['checks']['check_4_valid_port_range']['pass']:,} pass, {report['checks']['check_4_valid_port_range']['fail']:,} fail")
    print("-" * 60)
    print(f"TOTAL VALID FLOWS:   {report['summary']['valid_count']:,} ({report['summary']['valid_pct']:.4f}%)")
    print(f"TOTAL DROPPED FLOWS: {report['summary']['invalid_count']:,} ({report['summary']['invalid_pct']:.4f}%)")
    
    # Save validated flows
    output_path = "data/intermediate/ugr16/validated_flows.parquet"
    df_valid.to_parquet(output_path, index=False)
    print(f"\nSaved validated flows to: {output_path}")


if __name__ == "__main__":
    main()