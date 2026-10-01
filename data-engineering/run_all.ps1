python scripts\download_csvs_robust.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src\pipeline_runner.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts\run_extraction_v5.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts\rebuild_ucs_v5.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts\verify_pcap_reconciliation.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts\verify_item1_port_scan.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts\verify_ddos_untouched.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python scripts\diff_ucs_ml1_contract.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src\run_loeo_corrected.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

echo "All tasks passed successfully!"