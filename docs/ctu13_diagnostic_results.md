# CTU-13 Validation Diagnostic Results

This document records the exact, unedited results of downloading a real CTU-13 scenario and running it through the live ingestion pipeline to verify the protective `SchemaValidationError` layer.

## 1. Getting the Real CTU-13 Data

The official Stratosphere lab server `mcfp.felk.cvut.cz` was unreachable (timed out) from this environment. A CSV-converted version of the CTU-13 attack traffic was downloaded from a popular academic dataset mirror on GitHub.

**Download Command Used:**
```powershell
curl.exe -L -o CTU13_Attack_Traffic.csv https://raw.githubusercontent.com/imfaisalmalik/CTU13-CSV-Dataset/main/CTU13_Attack_Traffic.csv
```

**File Details:**
- **URL:** `https://raw.githubusercontent.com/imfaisalmalik/CTU13-CSV-Dataset/main/CTU13_Attack_Traffic.csv`
- **Size:** 9,341,949 bytes (~9.3 MB)
- **Rows:** 38,898

## 2. Running the Real Pipeline (Backend-Only)

Since interacting with and screenshotting a live Streamlit GUI was not possible in this headless Windows environment, the process was run backend-only to exactly mimic the UI path (`frontend/views/01_Telemetry_Ingestion.py`). 

A small test script (`frontend/test_ctu13.py`) was used to feed the real CSV into `data_provider.run_core_ml_inference(df, source_type="csv")`, which sequentially invokes `ucs_extractor.py` and the defensive validation layer.

**The Real Output:**
As designed, the pipeline immediately detected the schema mismatch and threw the expected `SchemaValidationError` rather than silently dropping data or crashing deeper in the stack.

```text
Loading CTU-13 dataset...
Loaded 38898 rows.
Running core ML inference...
D:\sih2026\.venv\Lib\site-packages\torch\jit\_script.py:1491: FutureWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.
  warnings.warn(
WARNING:root:Failed to load experimental FusedModel: No module named 'ml2'
[*] Notarizing model provenance on Fabric: lstm-gaussian-v3
[+] Successfully notarized model: lstm-gaussian-v3
Inference completed.
ERROR CAUGHT BY data_provider:
Expected 80 columns matching UCS schema, got 60; missing critical fields: [destination_port, protocol]. Details: ["Incompatible schema: missing critical flow fields ['destination_port', 'protocol']"]
TRACEBACK:
Traceback (most recent call last):
  File "D:\sih2026\frontend\data_provider.py", line 236, in run_core_ml_inference
    pred = predict(df, source_type=source_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\sih2026\backend\predict.py", line 995, in predict
    return pipeline.predict(raw_input=raw_input, source_type=source_type)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\sih2026\backend\predict.py", line 398, in predict
    window_df = self.extractor.extract(raw_input, source_type=source_type)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\sih2026\data-engineering\src\ucs_extractor.py", line 197, in extract
    raise SchemaValidationError(err_msg)
src.ucs_extractor.SchemaValidationError: Expected 80 columns matching UCS schema, got 60; missing critical fields: [destination_port, protocol]. Details: ["Incompatible schema: missing critical flow fields ['destination_port', 'protocol']"]
```

## 3. Verification

The main test suite was run directly after this experiment to confirm that the defensive layer didn't break functionality for standard CIC-IDS-2018 flows.

**Result:** `smoke_test.py` completed successfully with `ZERO DEFECTS`.
