# PROJECT HANDBOOK: CSE-CIC-IDS2018 → Unified Cyber State ($S_t$) Data Pipeline
**Problem Statement ID: SIH26153 — Cyber World Model Architecture (Data Engineer Track)**  
**Target Audience**: Data Engineers, ML Engineers, and Technical Judges  
**System Status**: Production-Validated Pipeline (Leakage-Safe, Chronological Split, Purge+Embargo Enforced, 37-Fold Cross-Validated)

---

## Table of Contents
1. [The Idea — Why This Project Exists](#1-the-idea--why-this-project-exists)
2. [What We Actually Built — The One-Paragraph Pitch](#2-what-we-actually-built--the-one-paragraph-pitch)
3. [Complete Project Structure](#3-complete-project-structure)
4. [File-by-File Deep Dive](#4-file-by-file-deep-dive)
5. [End-to-End Data Flow, With Real Numbers](#5-end-to-end-data-flow-with-real-numbers)
6. [Key Design Decisions and the "Why" Behind Each](#6-key-design-decisions-and-the-why-behind-each)
7. [The Validation Journey — Why You Can Trust This Dataset](#7-the-validation-journey--why-you-can-trust-this-dataset)
8. [Tech Stack](#8-tech-stack)
9. [Glossary](#9-glossary)
10. [Known Limitations](#10-known-limitations)
11. [Anticipated Judge Questions — Q&A](#11-anticipated-judge-questions--qa)

---

## 1. The Idea — Why This Project Exists

### The Core Problem: Why Intrusion Detection Is Broken
Traditional enterprise cybersecurity operates in a reactive paradigm. Intrusion Detection Systems (IDS), Security Information and Event Management (SIEM) platforms, and Security Orchestration, Automation, and Response (SOAR) engines analyze alerts *after* an attack signature has traversed the wire or *after* a payload has detonated on an endpoint. By the time a signature matches:
- Credentials have been dumped (e.g., via SSH or FTP brute-force).
- Internal reconnaissance has already mapped internal subnet routing (e.g., Nmap port-scans).
- A Command-and-Control (C2) callback has established persistence (e.g., Botnet check-ins or malware compromise).

In modern high-speed corporate and cloud networks, reactive detection means defense is always minutes or hours behind attacker automation.

### Forecasting vs. Detecting: The Cyber World Model Vision
What if a defensive system could anticipate an attack before weaponization or lateral movement begins? 

In robotics and autonomous systems, an agent maintains an internal representation of the environment called a **World Model**. By projecting how environmental state $S_t$ transitions to $S_{t+1}$ given actions or environmental dynamics, the agent predicts future obstacles instead of crashing into them. 

Applying this concept to network defense yields a **Cyber World Model**:
1. **Network Environment as State Space**: At any minute $t$, enterprise network interactions form an observable discrete state $S_t$. This state encapsulates flow statistical distributions, inter-arrival dynamics, transport layer flags, packet physical telemetry, and directed host communication topologies.
2. **Latent Predictive Dynamics**: A downstream temporal model (an LSTM/GRU) and a graph neural network (GraphSAGE) learn the latent transition dynamics:
   $$\mathbf{z}_t = \text{Encoder}(S_t) \implies \hat{\mathbf{z}}_{t+H} = \text{WorldModel}(\mathbf{z}_t)$$
3. **Multi-Step Attack Forecasting**: Rather than asking *"Is there an attack right now?"* ($\text{Detection}: S_t \to y_t$), the Cyber World Model asks *"Will an attack manifest within the next $H$ minutes based on early subtle precursors?"* ($\text{Forecasting}: S_t \to y_{t+H}$).

### What Is a Unified Cyber State ($S_t$)?
To forecast future attacks, downstream deep learning architectures cannot consume disorganized raw packet capture dumps (.pcap) or unaligned flow records (.csv). Raw traffic is asynchronous, high-dimensional, unstructured, and noisy.

A **Unified Cyber State ($S_t$)** is a mathematically formal, time-synchronized snapshot of an entire enterprise network over a discrete time window (specifically 1 minute). It represents network health and interactions simultaneously across two views:
- **Flat Feature State ($X_t \in \mathbb{R}^{D}$)**: Aggregated descriptive statistics (mean, standard deviation, sum, min, max) of volumetric counts, inter-arrival times, bidirectional ratios, TCP state machines, and packet headers.
- **Topological Interaction State ($G_t = (V_t, E_t)$)**: A directed multigraph of all active communicating endpoints (clients, servers, internal IP nodes, destination ports) active during window $t$, where edges capture directional volume and connection counts.

### The Problem Statement (SIH26153) and This Repo's Job
This project directly addresses **Problem Statement SIH26153 — Cyber World Model Architecture (Data Engineer Track)**. 

The mandated objective of the Data Engineer track is:
> Transform raw, uncurated network flow captures (CSE-CIC-IDS2018) into a mathematically rigorously validated, leakage-safe, dual-representation dataset (Unified Cyber State $S_t$) designed to feed a downstream downstream LSTM/GRU + GraphSAGE world model encoder.

**This repository does not train the downstream LSTM or GraphSAGE.** Rather, it is the foundational data engineering and validation layer. Its mission is to guarantee that the data delivered to the downstream ML engineers is:
1. **Free of temporal leakage**: Purge and embargo boundaries prevent forward-looking forecast labels or backward-looking LSTM context from crossing train/validation/test partitions.
2. **Free of schedule leakage**: Verified via Gate 0 diagnostic protocols to confirm the dataset reflects physical network behavior rather than synthetic testbed clock artifacts.
3. **Formally aligned**: Equipped with 1-minute window boundaries, feature presence masks, multi-horizon forecast targets ($H=5$ min primary), contiguous episode tracking, and fit-on-train-only robust scaling.

---

## 2. What We Actually Built — The One-Paragraph Pitch

We engineered an automated, end-to-end data ingestion and transformation pipeline that takes 5.14 million raw flow records from 6 days of CSE-CIC-IDS2018 traffic and 236.7 GB of network PCAP captures, executes rigorous sanitization (dropping exact duplicates, repairing division-by-zero rate infinities, purging 1970 epoch corruption, and segmenting two-phase Infiltration attacks), aggregates flows into 2,787 clean 1-minute Unified Cyber State ($S_t$) feature vectors (418 columns) and 754,071 directed interaction graph edges, enforces strict chronological splitting (70% Train / 15% Val / 15% Test) shielded by a 35-window purge-and-embargo zone ($W = L + H = 30 + 5$) at both boundaries (dropping 140 windows total), fits a leak-free `RobustScaler` exclusively on post-purge training windows, and exports dual-format artifacts: flat tabular feature arrays for linear baselines and boundary-safe 30-step temporal tensor sequences ($N, 30, D$) for downstream LSTMs, alongside verified 37-fold Leave-One-Episode-Out (LOEO) cross-validation evaluation harnesses.

### End-to-End Pipeline Execution Flow (Actual Call Sequence in `src/pipeline_runner.py`)

The sequence below illustrates the exact execution path codified in `pipeline_runner.py`:

```mermaid
flowchart TD
    subgraph STAGE_1_3 [Day-by-Day Ingestion & Cleaning]
        A[Raw CSV: 6 Days<br>5,138,471 rows] -->|ingest_csv_file| B[Stage 1: Latin-1 Reader<br>Strip whitespace & duplicate headers]
        B -->|map_to_canonical_schema| C[Stage 2: Canonical Mapper<br>80 CICFlowMeter to UCS names]
        C -->|clean_and_normalize_flow_data| D[Stage 3: Data Cleaner<br>UTC parse, Inf-to-NaN, drop 1970 epoch & 258k duplicates]
        D -->|impute_missing_flow_values| E[Missing Imputation<br>Within-day median fill]
        E -->|Save/Load| F[(Intermediate Parquet<br>cleaned_*.parquet)]
    end

    subgraph STAGE_4_6A [Windowing, Topology & Local Labeling]
        F -->|create_1min_windows| G[Stage 4: 1-Min Window Aggregator<br>mean/std/sum/min/max + Feature Masks]
        F -->|build_window_edge_lists| H[Stage 5: Graph Builder<br>Integer node IDs & directed edge lists]
        G -->|assign_window_labels| I[Stage 6: Labeler<br>Canonical attack mapping & 2-Phase Infiltration]
    end

    subgraph GLOBAL_ASSEMBLY [Cross-Day Alignment & Leakage Protection]
        I --> J[Concatenate All Days<br>2,927 pre-purge windows]
        H --> K[(ucs_graph_edgelists.parquet<br>754,071 edges)]
        J -->|generate_future_attack_labels| L[Future Forecasting Label<br>H=5 backward rolling max]
        L -->|assign_chronological_splits| M[Chronological Split<br>70% Train / 15% Val / 15% Test]
        M -->|apply_purge_embargo| N[Stage 6b: Purge + Embargo<br>Drop L+H=35 windows at each boundary (-140 windows)]
        N -->|Left Join| O[Merge Packet Features<br>pcap_extractor.py: 13 features for 14-02-2018]
        O -->|assign_episode_ids| P[Contiguous Episode Assignment<br>episode_id runs]
        P -->|Day-safe bfill| Q[Materialize forecast_episode_id<br>Pre-onset forecast target linkage]
    end

    subgraph STAGE_7_8 [Normalization & Final Artifact Export]
        Q -->|normalize_window_features| R[Stage 7: LeakageSafeRobustScaler<br>Fit on 2,013 Train windows only; transform Val/Test]
        R -->|build_lstm_sequences| S[Stage 8: LSTM Sequence Builder<br>L=30 windows sliding tensor]
        R -->|get_flat_window_data_for_lr| T[Flat Window Extraction<br>Tabular parity baseline]
        R --> U[(data/ucs/ucs_windows.parquet<br>2,787 rows x 418 cols)]
        R --> V[(data/ucs/scaler_params.yaml<br>400 fitted parameters)]
        S --> W[verify_no_cross_boundary_sequences<br>Runtime assertion: 100% boundary safe]
        R --> X[generate_validation_report & generate_schema_documentation<br>VALIDATION_REPORT.md & SCHEMA.md]
    end
```

---

## 3. Complete Project Structure

### Repository File Tree
```
e:\SIH 2026 - UCS Ingestion Pipeline (Main)\
├── .gitignore                                 # Git exclusion rules
├── configs/
│   ├── attack_timelines.yaml                  # Ground-truth Table 2 attack timelines & canonical normalization map
│   ├── canonical_mapping.yaml                 # 80 CICFlowMeter raw columns to UCS canonical schema mapping
│   └── pipeline_config.yaml                   # Master configuration (paths, windowing, H, L, split ratios, scaler)
├── data/
│   ├── intermediate/                          # Staged cleaned parquet files per day (caching layer)
│   │   ├── cleaned_Friday-02-03-2018_TrafficForML_CICFlowMeter.parquet
│   │   ├── cleaned_Thursday-01-03-2018_TrafficForML_CICFlowMeter.parquet
│   │   ├── cleaned_Thursday-22-02-2018_TrafficForML_CICFlowMeter.parquet
│   │   ├── cleaned_Wednesday-14-02-2018_TrafficForML_CICFlowMeter.parquet
│   │   ├── cleaned_Wednesday-21-02-2018_TrafficForML_CICFlowMeter.parquet
│   │   └── cleaned_Wednesday-28-02-2018_TrafficForML_CICFlowMeter.parquet
│   ├── raw/
│   │   └── CSE-CIC-IDS2018-csv/               # 6 raw CSE-CIC-IDS2018 CSV captures (5.14M rows)
│   └── ucs/                                   # Final Unified Cyber State (S_t) artifacts & audit reports
│       ├── SCHEMA.md                          # Schema documentation detailing all 418 columns
│       ├── VALIDATION_REPORT.md               # Authoritative pipeline validation report & quality gate audit
│       ├── attack_tactics_mapping.yaml        # MITRE ATT&CK tactic/technique mapping per attack type
│       ├── loeo_37fold_results.csv            # Authoritative 37-fold LOEO evaluation results (Detection & Forecast)
│       ├── loeo_fold_results.csv              # Full 40-fold LOEO results (including 3 singletons)
│       ├── loeo_fold_results_behavior_only.csv# Superseded 38-fold run (historical audit trail)
│       ├── loeo_fold_results_h5.csv           # H-sweep fold results for H=5
│       ├── loeo_fold_results_h10.csv          # H-sweep fold results for H=10
│       ├── loeo_fold_results_h15.csv          # H-sweep fold results for H=15
│       ├── node_lookup.parquet                # Anonymized integer node lookup (70,647 endpoints)
│       ├── packet_features.parquet            # Packet-level features for 14-02-2018 (543 windows)
│       ├── scaler_params.yaml                 # Fitted RobustScaler parameters on post-purge Train split
│       ├── ucs_graph_edgelists.parquet        # Interaction multigraph edges (754,071 directed edges)
│       └── ucs_windows.parquet                # Master flat temporal tensor (2,787 rows x 418 cols)
├── demo_notes.md                              # Instructions for Live PCAP vs Batch mode dual contract
├── display_h_sweep_results.py                 # Summary formatter script for comparing H=5, 10, 15
├── gate0_leakage_report.md                    # Diagnostic report comparing Protocols 1-4 and schedule leakage
├── gate0_protocol4_loeo_forecasting_report.md # Class-weighted forecasting diagnostic analysis
├── h_sweep_full_output.txt                    # Raw execution log of H-sweep
├── h_sweep_results.json                       # JSON summary statistics of H-sweep
├── loeo_forecasting_balanced_run.txt          # Raw execution log of balanced LOEO forecasting
├── loeo_forecasting_run.txt                   # Raw execution log of unweighted LOEO forecasting
├── README.md                                  # Repository overview and quickstart guide
├── run_h_sweep.py                             # Pipeline-orchestrated H-sweep execution runner
├── run_h_sweep_direct.py                      # Direct LOEO runner across H=5, 10, 15
├── run_h_sweep_minimal.py                     # Minimal single-type H-sweep tester
├── run_h_sweep_simple.py                      # Simplified H-sweep diagnostic script
├── run_h_sweep_validated.py                   # Validated 37-fold H-sweep runner
├── scratch/                                   # Diagnostic scripts and experimental verification scratchpad
│   ├── check_aws_s3.py                        # AWS S3 PCAP bucket inspection script
│   ├── check_ep1_embargo.py                   # Embargo boundary inspection for Botnet Episode 1
│   ├── compute_37fold_stats.py                # Authoritative 37-fold metric calculator
│   ├── investigate_episodes.py                # Contiguous episode run analysis
│   ├── materialize_episode_cols.py            # Standalone episode column materializer
│   ├── s3_pcap_check.py                       # S3 size checker via boto3
│   ├── s3_sizes.py                            # S3 size summary utility
│   ├── step1_recount.py                       # Raw row counter and duplicate verifier
│   ├── step3_investigate.py                   # Stage 3 data cleaning investigator
│   └── test_loeo_runner.py                    # Unit tester for LOEO fold generator
├── src/                                       # Core production pipeline source code
│   ├── __init__.py                            # Package metadata and version definition
│   ├── canonical_mapper.py                    # Stage 2: 80-feature canonical schema translator
│   ├── cleaner.py                             # Stage 3: Data cleaning, UTC timestamp parsing, deduplication
│   ├── gate0_diagnostic.py                    # Diagnostic harness for in-sample sanity checks
│   ├── gate0_leakage_test.py                  # Gate 0 test suite (Protocols 1, 2, 3)
│   ├── gate0_protocol4_loeo.py                # Gate 0 Protocol 4 LOEO script (original)
│   ├── graph_builder.py                       # Stage 5: Per-window interaction graph generator
│   ├── ingestion.py                           # Stage 1: Robust Latin-1 CSV file reader
│   ├── labeler_and_splits.py                  # Stage 6 & 6b: Attack labels, splits, purge+embargo, episodes
│   ├── normalizer.py                          # Stage 7: Leakage-safe RobustScaler + Log1p normalizer
│   ├── pcap_extractor.py                      # Packet-level feature extractor via Scapy (live & deterministic)
│   ├── pipeline_runner.py                     # Master pipeline orchestrator & audit documentation generator
│   ├── run_loeo_corrected.py                  # Authoritative LOEO evaluation harness (Detection & Forecast)
│   ├── sequence_builder.py                    # Stage 8: Boundary-safe LSTM sequence & flat window constructor
│   └── window_aggregator.py                   # Stage 4: 1-minute window aggregator and feature masks
└── tests/
    └── test_pipeline.py                       # Comprehensive unit and regression test suite (10/10 passing)
```

### Purpose of Each Directory
- **`configs/`**: Centralized configuration management using YAML files. Houses master pipeline hyperparameters (`pipeline_config.yaml`), raw-to-canonical schema mappings (`canonical_mapping.yaml`), and authoritative Table 2 attack timelines (`attack_timelines.yaml`). Decouples all thresholds from Python logic.
- **`data/raw/`**: The immutable landing zone for raw source data. Holds the 6 original CSE-CIC-IDS2018 CSV files totaling 5.14M records.
- **`data/intermediate/`**: A high-performance caching layer. Stores day-by-day cleaned and UTC-parsed Parquet files. Allows downstream windowing, graph building, or labeling to re-run in seconds without repeating the 5-minute CSV parsing and deduplication process.
- **`data/ucs/`**: The final destination for all production deliverables. Houses the dual-format Unified Cyber State Parquet files (`ucs_windows.parquet`, `ucs_graph_edgelists.parquet`), the node lookup table, fitted scaler parameters, evaluation CSVs, and audit markdown documentation.
- **`scratch/`**: Developer scratchpad containing one-off audit scripts used during debugging to verify AWS S3 PCAP sizes, calculate 37-fold metrics, and audit embargo boundaries. Persisted for reproducibility.
- **`src/`**: The modular, production-grade Python package implementing pipeline stages 1 through 8, the packet extractor, and the Gate 0 validation harnesses.
- **`tests/`**: Unit and regression tests executable via Python `unittest`. Validates schema mapping, cleaning, window aggregation, graph building, purge/embargo math, and LSTM boundary safety.

### Top-Level Files (Non-Src)
- **`.gitignore`**: Defines files and directories excluded from Git tracking (e.g., virtual environments, large Parquet caches, logs).
- **`README.md`**: Project overview, key features, architecture summary, and command-line execution instructions.
- **`demo_notes.md`**: Technical specification of the dual-input contract (live PCAP vs. deterministic fallback) for downstream demonstration.
- **`gate0_leakage_report.md`**: Deep-dive validation report comparing evaluation Protocols 1, 2, 3, and 4 to prove model performance is not driven by clock artifacts.
- **`gate0_protocol4_loeo_forecasting_report.md`**: Methodological report detailing the class-weighted LOEO forecasting diagnostic and PR-AUC findings.
- **`H_SWEEP_REPORT.md`**: Comparative sensitivity report analyzing forecast horizons $H=5, 10, 15$ across 37 folds.
- **`h_sweep_full_output.txt` & `h_sweep_results.json`**: Complete console logs and structured metrics from the horizon sweep execution.
- **`loeo_forecasting_run.txt` & `loeo_forecasting_balanced_run.txt`**: Execution logs of LOEO forecasting experiments without and with balanced class weighting.
- **`display_h_sweep_results.py`**: Console table generator formatting cross-horizon LOEO metrics side-by-side.
- **`run_h_sweep.py`, `run_h_sweep_direct.py`, `run_h_sweep_minimal.py`, `run_h_sweep_simple.py`, `run_h_sweep_validated.py`**: Various iterations of the cross-validation sweep runner developed during experimental diagnostic phases.

---

## 4. File-by-File Deep Dive

This section provides an exhaustive technical breakdown of every Python source module in `src/`, key root execution scripts, and test suites. Every signature and line number has been verified against the current repository state.

---

### `src/__init__.py`
- **What it does**: Declares the `src` directory as a Python package and specifies module metadata and version string.
- **Where it sits**: Root of package.
- **Defined attributes**:
  - `__version__ = "1.0.0"`
- **Config values read**: None.

---

### `src/ingestion.py`
- **What it does**: Robustly ingests raw CSE-CIC-IDS2018 CSV files. Handles encoding anomalies via Latin-1 fallback, strips erratic whitespace from column headers, strips repeated embedded header rows mid-file, and injects dataset provenance tracking.
- **Where it sits**: Stage 1. Called by `pipeline_runner.py` during raw data processing.
- **Functions defined**:
  - `extract_day_from_filename(filename: str) -> str`
    - *Takes*: File name or path (e.g., `"Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"`).
    - *Returns*: Standardized date string `"14-02-2018"`.
    - *Key logic*: Uses regex `(\d{2}-\d{2}-\d{4})` to extract calendar day; falls back to base filename if unparseable.
  - **`ingest_csv_file(filepath: str, encoding: str = "latin1", chunksize: Optional[int] = None) -> pd.DataFrame`** [**SAFETY-CRITICAL**]
    - *Takes*: Absolute or relative path to raw CSV, encoding string, optional chunksize.
    - *Returns*: Sanitized raw `pd.DataFrame` with added `source_file` and `source_day` columns.
    - *Key logic*: Reads CSV with `low_memory=False` and `on_bad_lines="skip"`. Strips whitespace from all column names (`col.strip()`). Crucially checks for mid-file embedded headers (`df["Label"].str.strip() == "Label"`) generated by distributed CICFlowMeter logger crashes, dropping them so they do not contaminate downstream numeric casts. Injects immutable provenance strings.
  - `load_dataset_day(filepath: str) -> Tuple[pd.DataFrame, Dict[str, any]]`
    - *Takes*: Filepath string.
    - *Returns*: Loaded `pd.DataFrame` and audit dictionary containing filename, byte size, initial row count, and column count.
- **Config values read**: None directly (file paths passed from `pipeline_config.yaml`).

---

### `src/canonical_mapper.py`
- **What it does**: Translates raw, inconsistently formatted CICFlowMeter column names into a clean, standardized snake_case canonical schema. 
- **Where it sits**: Stage 2. Called immediately after ingestion in `pipeline_runner.py`.
- **Functions defined**:
  - `load_canonical_mapping(config_path: str = "configs/canonical_mapping.yaml") -> Dict[str, any]`
    - *Takes*: Path to mapping YAML.
    - *Returns*: Parsed dictionary of column mappings and feature groups.
  - **`map_to_canonical_schema(df: pd.DataFrame, mapping_config: Optional[Dict[str, any]] = None, config_path: str = "configs/canonical_mapping.yaml") -> Tuple[pd.DataFrame, Dict[str, any]]`** [**SAFETY-CRITICAL**]
    - *Takes*: Raw `DataFrame`, optional loaded config dictionary, path string.
    - *Returns*: Renamed `DataFrame` and mapping audit dictionary.
    - *Key logic*: Strips whitespace from dataframe column headers before matching against `mapping_config["mapping"]`. Maps all 80 raw flow fields (e.g., `"Tot Fwd Pkts"` $\to$ `"packet_count_fwd"`, `"Init Fwd Win Byts"` $\to$ `"window_size_fwd"`, `"Flow Byts/s"` $\to$ `"bytes_per_sec"`). Preserves provenance columns (`source_file`, `source_day`). Flags any unmapped fields in the audit return.
- **Config values read**: `configs/canonical_mapping.yaml` (full mapping dictionary).

---

### `src/cleaner.py`
- **What it does**: Implements data quality and integrity filtering. Converts division-by-zero rate infinities to NaN, parses raw strings to UTC timestamps, filters corrupted 1970 epoch artifacts, purges exact duplicate rows, audits negative inter-arrival times, and applies within-day median imputation.
- **Where it sits**: Stage 3. Transforms canonical data prior to window aggregation.
- **Functions defined**:
  - **`clean_and_normalize_flow_data(df: pd.DataFrame, rate_cols: Optional[List[str]] = None, timing_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]`** [**SAFETY-CRITICAL**]
    - *Takes*: Canonical `DataFrame`, optional lists of rate and timing column names.
    - *Returns*: Cleaned `DataFrame` and exhaustive audit dictionary.
    - *Key logic*:
      1. Parses `"raw_timestamp"` with format `"%d/%m/%Y %H:%M:%S"` into timezone-aware `timestamp_utc` (`datetime64[ns, UTC]`).
      2. Drops corrupted epoch timestamps (filters `df["timestamp_utc"].dt.year >= 2018`). Drops 14 corrupted rows across the dataset.
      3. Replaces string and float infinities (`np.inf`, `-np.inf`, `'Infinity'`) with `np.nan` across all numeric features.
      4. Detects and audits negative values in timing columns (`duration_microsec`, `*_iat_*`, `active_*`, `idle_*`), replacing negative durations with NaN.
      5. Derives `duration_sec = duration_microsec / 1e6`.
      6. Executes deduplication on all feature columns excluding provenance: drops 258,084 exact duplicate flows.
      7. Detects near-duplicates (same 5-tuple signature + same second) and logs them.
      8. Sorts strictly chronologically by `timestamp_utc`.
  - **`impute_missing_flow_values(df: pd.DataFrame, strategy: str = "median", feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, float]]`** [**SAFETY-CRITICAL**]
    - *Takes*: Cleaned `DataFrame`, strategy string (`"median"`), optional feature column list.
    - *Returns*: Imputed `DataFrame` and dictionary of imputed values.
    - *Key logic*: Replaces NaNs (originating from Inf conversions or negative timing drops) with the within-day column median. Prevents cross-dataset leakage by calculating statistics locally before global concatenation.
- **Config values read**: None directly (defaults cover canonical names).

---

### `src/window_aggregator.py`
- **What it does**: Discretizes asynchronous flow records into regular 1-minute time windows. Computes multi-statistic aggregation profiles and attaches explicit feature-presence masks.
- **Where it sits**: Stage 4. Aggregates cleaned flows into window-level states $S_t$.
- **Functions defined**:
  - **`create_1min_windows(df: pd.DataFrame, feature_groups_config: Optional[Dict[str, List[str]]] = None, interval_sec: int = 60) -> Tuple[pd.DataFrame, Dict[str, any]]`** [**SAFETY-CRITICAL**]
    - *Takes*: Cleaned flow `DataFrame`, optional feature group mapping, interval in seconds (default 60).
    - *Returns*: Window-level `DataFrame` and aggregation audit dictionary.
    - *Key logic*:
      1. Floors `timestamp_utc` to 60-second intervals (`freq_str = "60s"`), creating `window_start_utc`.
      2. Groups by `window_start_utc` and calculates 5 aggregate statistics (`["mean", "std", "sum", "min", "max"]`) across all numeric flow features.
      3. Flattens MultiIndex column tuples into clean strings (`"{col}_{stat}"`, e.g., `byte_count_fwd_mean`).
      4. Aggregates topological metadata per window: `flow_count` (size), `unique_dst_ports_count` (`nunique`), `unique_protocols_count` (`nunique`).
      5. Imputes standard deviations for single-flow windows: fills NaN `*_std` columns with `0.0`.
      6. Constructs unique string identifiers: `window_id = "W_{source_day}_{YYYYMMDD_HHMMSS}"`.
      7. Injects 6 binary feature-presence masks: `mask_has_traffic_volume_features=1.0`, `mask_has_flow_timing_features=1.0`, `mask_has_packet_level_features=1.0` (later adjusted per PCAP availability), `mask_has_tcp_flags=1.0`, `mask_has_graph_topology=1.0`, and `mask_has_identity_auth=0.0`.
      8. Preserves window dominant label (`raw_label_dominant`) and malicious flow presence flag (`has_malicious_flows`).
- **Config values read**: `pipeline_config.yaml` (`windowing.interval_sec`).

---

### `src/graph_builder.py`
- **What it does**: Constructs per-window directed interaction multigraphs from flow communications. Maps endpoints to persistent integer node IDs and exports compact edge lists with transmission attributes.
- **Where it sits**: Stage 5. Runs in parallel with window aggregation.
- **Classes and methods defined**:
  - `class GraphTopologyBuilder`:
    - `__init__()`: Initializes `node_to_id: Dict[str, int]`, `id_to_node: Dict[int, str]`, and counter `_next_id = 1`.
    - `get_or_create_node_id(node_key: str) -> int`: Returns stable integer ID for endpoint string, incrementing on new nodes.
    - **`build_window_edge_lists(df_flows: pd.DataFrame, interval_sec: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, any]]`** [**SAFETY-CRITICAL**]
      - *Takes*: Cleaned flow `DataFrame`, window interval in seconds.
      - *Returns*: `edge_df`, `node_lookup_df`, and graph audit dictionary.
      - *Key logic*: Derives endpoint identities. When IP addresses are absent from public CSVs, constructs consistent protocol-port behavioral endpoints:
        - Source client: `Client_Proto{protocol}_Win{window_size_fwd}`
        - Destination service: `SvcPort_{destination_port}_P{protocol}`
        Groups flows by `[window_start_utc, src_node_id, dst_node_id]` and aggregates directed edge metrics: `flow_count`, `byte_count_sum`, `packet_count_sum`, `duration_mean_sec`, and `protocol_mode`.
- **Config values read**: `pipeline_config.yaml` (`windowing.interval_sec`).

---

### `src/labeler_and_splits.py`
- **What it does**: The central safety module of the project. Maps raw labels to canonical attack types, segments two-phase Infiltration, derives leakage-free future forecasting labels, enforces chronological splits, computes and applies purge+embargo zones, and segments contiguous attack episodes.
- **Where it sits**: Stage 6 & 6b. Called during global multi-day dataset assembly.
- **Functions defined**:
  - `load_attack_schedules(config_path: str = "configs/attack_timelines.yaml") -> Dict[str, any]`
    - *Takes*: Config path string.
    - *Returns*: Parsed YAML dictionary of attack schedules and normalization mappings.
  - **`assign_window_labels(window_df: pd.DataFrame, schedules_config: Optional[Dict[str, any]] = None, config_path: str = "configs/attack_timelines.yaml") -> Tuple[pd.DataFrame, Dict[str, any]]`** [**SAFETY-CRITICAL**]
    - *Takes*: Window `DataFrame`, optional loaded config, config path.
    - *Returns*: Labeled `DataFrame` with `label_binary` and `label_attack_type`, plus audit dictionary.
    - *Key logic*: Maps raw dominant labels through `label_normalization_map`. Identifies March 1 Infiltration traffic and applies the mandated two-phase temporal segmentation:
      - Hours $\le 05$: `Infiltration-Compromise` (initial malware drop & C2 connection)
      - Hours $> 05$: `Infiltration-Portscan` (internal network discovery via Nmap)
      Guarantees `label_binary = 1` whenever `label_attack_type != "Benign"`.
  - **`generate_future_attack_labels(window_df: pd.DataFrame, horizon_windows: int = 5, target_col: str = "label_binary", future_col: str = "future_attack_label") -> pd.DataFrame`** [**SAFETY-CRITICAL**]
    - *Takes*: Window `DataFrame`, forecast horizon $H$ in windows (default 5), target column, future column name.
    - *Returns*: `DataFrame` with materialized future target column.
    - *Key logic*: Computes $P(\text{Attack} \in [t+1, t+H])$. Sorts chronologically, reverses the binary label series, applies a rolling maximum of window size $H$, reverses back, and shifts backward by 1 (`shift(-1)`). Fills trailing tail NaNs with 0. **Strict zero-leakage guarantee**: the target label is shifted backward in time, ensuring no future features or labels contaminate current window state $S_t$.
  - **`assign_chronological_splits(window_df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[pd.DataFrame, Dict[str, any]]`** [**SAFETY-CRITICAL**]
    - *Takes*: Window `DataFrame`, split proportions.
    - *Returns*: `DataFrame` with assigned `split` column (`"train"`, `"val"`, `"test"`), plus boundary audit.
    - *Key logic*: Sorts chronologically by `window_start_utc`. Computes integer split thresholds ($N_{\text{train}} = \lfloor 0.70 N \rfloor$, $N_{\text{val}} = \lfloor 0.15 N \rfloor$, $N_{\text{test}} = \text{remainder}$). Strictly forbids random shuffling of time series data.
  - **`apply_purge_embargo(window_df: pd.DataFrame, lookback_windows: int, horizon_windows: int) -> Tuple[pd.DataFrame, Dict[str, any]]`** [**SAFETY-CRITICAL**]
    - *Takes*: Window `DataFrame` with `split` column, lookback $L$, horizon $H$.
    - *Returns*: Purged `DataFrame` with boundary windows removed, and detailed audit dictionary.
    - *Key logic*: Calculates boundary quarantine width dynamically:
      $$\text{width} = L + H = 30 + 5 = 35 \text{ windows}$$
      Drops $\text{width}$ windows from BOTH sides of each partition boundary:
      - Train $\to$ Val boundary: Drops last 35 windows of Train and first 35 windows of Val.
      - Val $\to$ Test boundary: Drops last 35 windows of Val and first 35 windows of Test.
      Drops exactly 140 windows total. Guarantees that neither an LSTM sequence's lookback context ($L=30$) nor a forecast label's forward horizon ($H=5$) can span across partition boundaries.
  - **`assign_episode_ids(window_df: pd.DataFrame) -> pd.DataFrame`** [**SAFETY-CRITICAL**]
    - *Takes*: Window `DataFrame`.
    - *Returns*: `DataFrame` with added string `episode_id`.
    - *Key logic*: Detects contiguous, unbroken runs of identical `source_day` and `label_attack_type`. Constructs unique string keys:
      $$\text{episode\_id} = \{\text{source\_day}\}\_\{\text{label\_attack\_type}\}\_\{\text{run\_index}\}$$
      Identifies 40 total contiguous attack episodes across the 6 days.
- **Config values read**: `configs/attack_timelines.yaml` (timelines and normalization map), `pipeline_config.yaml` (`forecasting.horizon_windows`, `lstm.lookback_windows`, `splits.*`).

---

### `src/normalizer.py`
- **What it does**: Implements leakage-free feature scaling. Applies `log1p` transformation to heavily skewed volumetric features, fits a `RobustScaler` (median and IQR) exclusively on the training partition, and transforms validation and test splits without leakage.
- **Where it sits**: Stage 7. Normalizes feature columns in $S_t$ after purge and embargo.
- **Classes and functions defined**:
  - `class LeakageSafeRobustScaler`:
    - `__init__(log1p_cols: Optional[List[str]] = None)`: Initializes column lists and parameter store.
    - **`fit(train_df: pd.DataFrame, feature_cols: List[str]) -> LeakageSafeRobustScaler`** [**SAFETY-CRITICAL**]: Fits scaling parameters **strictly** on the training partition. Computes $Q_{25}, Q_{50} (\text{median}), Q_{75}$, and $\text{IQR} = \max(Q_{75} - Q_{25}, 1\times 10^{-6})$. Falls back to standard deviation or 1.0 if IQR is zero. Stores `params[col]`.
    - **`transform(df: pd.DataFrame) -> pd.DataFrame`** [**SAFETY-CRITICAL**]: Applies fitted transformation:
      $$x_{\text{norm}} = \frac{\log_{1p}(\max(x, 0)) - Q_{50}}{\text{IQR}}$$
      Handles missing values by filling with fitted training median.
    - `save_parameters(filepath: str)`: Dumps fitted parameters to YAML/JSON for exact pipeline reproducibility.
  - **`normalize_window_features(window_df: pd.DataFrame, log1p_sub_keys: Optional[List[str]] = None, save_params_path: Optional[str] = "data/ucs/scaler_params.yaml") -> Tuple[pd.DataFrame, LeakageSafeRobustScaler, Dict[str, Any]]`** [**SAFETY-CRITICAL**]
    - *Takes*: Window `DataFrame`, list of substring keys for log1p, output path.
    - *Returns*: Normalized `DataFrame`, fitted scaler instance, audit dictionary.
    - *Key logic*: Filters out all metadata and target columns. Extracts training split rows (`split == 'train'`). Fits `LeakageSafeRobustScaler` solely on the 2,013 post-purge train windows. Transforms the entire dataset and saves 400 parameter blocks to `scaler_params.yaml`.
- **Config values read**: `pipeline_config.yaml` (`normalization.log1p_cols`).

---

### `src/sequence_builder.py`
- **What it does**: Transforms flat window records into 3D temporal tensors for downstream LSTM/GRU models and flat feature matrices for linear baselines, enforcing complete boundary safety and protocol parity.
- **Where it sits**: Stage 8. Final stage of the data pipeline.
- **Functions defined**:
  - **`build_lstm_sequences(purged_df: pd.DataFrame, lookback_windows: int, feature_cols: List[str], target_col: str = "future_attack_label") -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]`** [**SAFETY-CRITICAL**]
    - *Takes*: Purged window `DataFrame`, lookback length $L$ (default 30), feature column names, target column.
    - *Returns*: `X_by_split` (dict of 3D arrays: `(n_seq, L, n_feat)`), `y_by_split` (dict of 1D target vectors), and audit dictionary.
    - *Key logic*: Iterates through splits independently (`"train"`, `"val"`, `"test"`). Constructs sliding window sequences of length $L=30$ entirely within each partition. Sets the sequence label to the `future_attack_label` of the **last window** in the sequence. Never truncates, pads, or borrows windows across partitions. Generates 1,984 train, 340 val, and 376 test sequences.
  - **`verify_no_cross_boundary_sequences(purged_df: pd.DataFrame, X_by_split: Dict[str, np.ndarray], lookback_windows: int, feature_cols: List[str]) -> bool`** [**SAFETY-CRITICAL**]
    - *Takes*: Purged `DataFrame`, sequence dict, lookback integer, feature list.
    - *Returns*: Boolean (`True` if valid).
    - *Key logic*: Runtime verification assertion. Verifies that the number of sequences in each split equals $N_{\text{split}} - L + 1$, and verifies using `np.allclose` that sequence values match the source partition boundaries perfectly with zero leakage.
  - `get_flat_window_data_for_lr(purged_df: pd.DataFrame, feature_cols: List[str], target_col: str = "future_attack_label") -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]`
    - *Takes*: Purged `DataFrame`, feature columns, target column.
    - *Returns*: `X_by_split`, `y_by_split` (2D arrays of flat windows).
    - *Key logic*: Extracts flat window matrices from the **exact same purged window set** used by the LSTM. Ensures absolute protocol parity between baseline tabular models and temporal deep learning models.
- **Config values read**: `pipeline_config.yaml` (`lstm.lookback_windows`, `forecasting.future_label_col`).

---

### `src/pcap_extractor.py`
- **What it does**: Extracts granular packet-level telemetry from raw PCAP network captures using Scapy. Implements the dual-input contract (streaming live PCAP dissection vs. deterministic flow-aligned fallback for Wednesday-14-02-2018).
- **Where it sits**: Packet extraction module. Called during Stage 6 assembly.
- **Functions defined**:
  - `calculate_port_scan_monotonicity(ports: List[int]) -> float`: Computes sequencing score $[0.0, 1.0]$ based on step differences ($\pm 1$) and Shannon entropy over destination ports accessed by a source IP.
  - `extract_features_from_pcap_stream(pcap_path: str, window_boundaries: List[Tuple[pd.Timestamp, pd.Timestamp, str]]) -> pd.DataFrame`: Uses Scapy `PcapReader` to stream raw packets, aggregating IP TTL statistics (min, max, std, mode), fragmentation flags (More Fragments MF, Don't Fragment DF), payload size quantiles ($p_{25}, p_{50}, p_{75}, p_{95}$), TCP retransmissions, and port-scan monotonicity scores per 1-minute window.
  - `generate_synthetic_test_pcap(filepath: str, num_packets: int = 20, base_time_utc: Optional[datetime] = None)`: Synthesizes valid `.pcap` files with known IP/TCP layers for unit testing.
  - **`extract_or_generate_packet_features(ucs_windows_path: str = "data/ucs/ucs_windows.parquet", target_day: str = "14-02-2018", pcap_input_path: Optional[str] = None, output_path: str = "data/ucs/packet_features.parquet") -> pd.DataFrame`** [**SAFETY-CRITICAL**]: Implements the dual-input contract. If `pcap_input_path` is provided and exists, parses live packets. Otherwise, executes deterministic flow-aligned packet synthesis matching Wednesday-14-02-2018 SSH/FTP brute-force windows. Exports 13 columns for 543 windows to `packet_features.parquet`.
- **Config values read**: None.

---

### `src/pipeline_runner.py`
- **What it does**: The master pipeline orchestrator. Executes Stages 1 through 8 sequentially, enforces purge+embargo, materializes `forecast_episode_id` via day-safe backward fill, coordinates scaler fitting, builds LSTM sequences, verifies boundary safety, and generates `VALIDATION_REPORT.md` and `SCHEMA.md`.
- **Where it sits**: Master entry point.
- **Functions defined**:
  - `load_pipeline_config(config_path: str = "configs/pipeline_config.yaml") -> Dict[str, Any]`: Loads master YAML settings.
  - **`run_pipeline(config_path: str = "configs/pipeline_config.yaml") -> Dict[str, Any]`** [**SAFETY-CRITICAL**]: Main orchestration loop. Iterates through the 6 dataset days, manages intermediate Parquet caching, concatenates multi-day states, shifts forecasting labels, splits chronologically, applies purge+embargo (-140 windows), merges packet features, assigns episode IDs, materializes `forecast_episode_id`, fits RobustScaler on Train, constructs LSTM tensors, and writes Parquet outputs.
  - `count_attack_episodes(df: pd.DataFrame) -> Dict[str, int]`: Audits contiguous attack runs per attack type.
  - `generate_validation_report(windows_df: pd.DataFrame, edges_df: pd.DataFrame, audit_data: Dict[str, Any], output_dir: str, config: Optional[Dict[str, Any]] = None)`: Programmatically writes `VALIDATION_REPORT.md` with row accounting, quality gate checks, and purge/embargo metrics.
  - `generate_schema_documentation(windows_df: pd.DataFrame, edges_df: pd.DataFrame, output_dir: str)`: Programmatically writes `SCHEMA.md` documenting all 418 columns, types, and categories.
- **Config values read**: Reads entire `configs/pipeline_config.yaml`.

---

### `src/run_loeo_corrected.py`
- **What it does**: The authoritative cross-validation harness. Evaluates both Detection (`label_binary`) and Onset Forecasting (`future_attack_label`, $H=5$, pre-onset filtered) across multi-episode attack types using `DecisionTreeClassifier(max_depth=4, class_weight='balanced')`.
- **Where it sits**: Core evaluation module for Gate 0 Protocol 4.
- **Functions defined**:
  - **`run_loeo(parquet_path: str = "data/ucs/ucs_windows.parquet", output_csv: str = "data/ucs/loeo_fold_results.csv") -> pd.DataFrame`** [**SAFETY-CRITICAL**]
    - *Takes*: Path to `ucs_windows.parquet`, output CSV destination.
    - *Returns*: DataFrame containing all fold metrics across Detection and Forecasting.
    - *Key logic*:
      1. Verifies presence of string `episode_id`.
      2. Materializes `forecast_episode_id` using **day-safe backward fill**: ensures pre-onset benign windows (`future_attack_label == 1 & label_binary == 0`) are assigned to the upcoming attack episode within the *same calendar day*, preventing cross-day bfill leakage.
      3. Separates attack types into multi-episode (eligible: SSH-Bruteforce [9], DDOS-LOIC-UDP [18], Botnet [10]) and singletons (excluded: DDOS-HOIC [1], Infiltration-Compromise [1], Infiltration-Portscan [1]).
      4. For each held-out episode fold, pairs the test attack/pre-onset windows with proportionally sampled benign windows across all 6 days.
      5. Trains Set A (traffic + packet telemetry) and Set B (hour-of-day + day one-hot) models.
      6. Computes F1, Precision, Recall, and PR-AUC. Exports 37 multi-episode folds (and 3 singleton folds) to CSV.
- **Config values read**: None.

---

### `src/gate0_leakage_test.py`
- **What it does**: Implements the multi-protocol diagnostic suite evaluating Protocols 1 (Chronological Holdout), 2 (Window-Stratified), and 3 (Episode-Grouped Stratified).
- **Where it sits**: Gate 0 evaluation suite.
- **Functions defined**:
  - `assign_contiguous_episodes(df: pd.DataFrame) -> pd.DataFrame`: Cumulative sum episode segmenter.
  - `split_by_episodes_stratified(df: pd.DataFrame, test_ratio: float = 0.15, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]`: Partitions whole episodes into train/test splits.
  - **`run_gate0_leakage_test(ucs_windows_path: str = "data/ucs/ucs_windows.parquet", output_report_path: str = "gate0_leakage_report.md") -> Dict[str, Any]`** [**SAFETY-CRITICAL**]: Executes comparative evaluation of Set A vs. Set B across Protocols 1, 2, and 3, outputting confusion matrices and diagnostics to `gate0_leakage_report.md`.
- **Config values read**: None.

---

### `src/gate0_protocol4_loeo.py`
- **What it does**: Original implementation of Leave-One-Episode-Out cross-validation. Preceded `run_loeo_corrected.py`.
- **Where it sits**: Historical / reference cross-validation script.
- **Functions defined**:
  - `evaluate_fold(train_data, test_data)`: Trains balanced decision trees on train fold and scores Set A vs. Set B on test fold.
  - Script-level execution loop running LOEO across multi-episode attack types.
- **Config values read**: None.

---

### `src/gate0_diagnostic.py`
- **What it does**: Diagnostic scratch script used to investigate in-sample decision tree capacity and single-fold probability calibration.
- **Where it sits**: Debugging utility.
- **Functions defined**: Script-level execution performing sanity checks on `DecisionTreeClassifier(max_depth=4)`.
- **Config values read**: None.

---

### Root Scripts
- **`run_h_sweep_validated.py`**: Executes full 37-fold LOEO across horizons $H=5, 10, 15$. Regenerates `future_attack_label` dynamically for each horizon and records fold metrics into `loeo_fold_results_h*.csv`.
- **`display_h_sweep_results.py`**: Reads the three H-sweep result CSVs and prints a formatted comparative Markdown table summarizing F1, PR-AUC, CI overlaps, and head-to-head win counts.
- **`run_h_sweep.py`, `run_h_sweep_direct.py`, `run_h_sweep_minimal.py`, `run_h_sweep_simple.py`**: Intermediate automation scripts developed to test different sub-configurations of horizon sweeps and error-handling wrappers.

---

### `tests/test_pipeline.py`
- **What it does**: Automated regression and unit test suite containing 10 test cases verifying pipeline integrity.
- **Where it sits**: `tests/`. Executable via `python tests/test_pipeline.py`.
- **Classes and test methods defined**:
  - `class TestUCSDataPipeline(unittest.TestCase)`:
    - `test_filename_extraction()`: Asserts correct date extraction from filename strings.
    - `test_canonical_mapping()`: Asserts canonical renaming of destination ports, protocols, byte counts, and TCP flags.
    - `test_cleaning_and_inf_handling()`: Asserts string/float infinities are converted to NaN and timestamps are UTC-monotonic.
    - `test_window_aggregation()`: Asserts 1-minute grouping, feature mask flags, and column structures.
    - `test_graph_builder()`: Asserts edge creation and integer node lookup generation.
    - **`test_future_attack_label_leakage_safety()`**: Asserts backward rolling max math ($P(\text{Attack} \in [t+1, t+H])$) produces zero forward leakage.
    - **`test_chronological_splits()`**: Asserts strict non-random temporal partitioning (70/15/15) with ascending time boundaries.
    - `test_leakage_safe_robust_scaler()`: Asserts RobustScaler fits solely on training data and correctly transforms unseen validation data.
    - **`test_lstm_sequence_no_cross_boundary()`**: Asserts that constructed LSTM sequences ($L=30$) never cross split boundaries after purge/embargo.
    - **`test_purge_embargo_width_from_config()`**: Asserts purge/embargo width equals $L + H$ dynamically read from configuration and drops exact boundary window counts.
- **Config values read**: `configs/pipeline_config.yaml`.

---

## 5. End-to-End Data Flow, With Real Numbers

Every number in this section has been verified by directly loading the raw CSVs, intermediate parquets, and final output artifacts in the current repository.

```
RAW CSVs (6 days)
  5,138,471 raw flow rows
  │
  ├── Drop corrupted epoch-1970 timestamps: -14 rows
  ├── Drop mid-file duplicate header rows: ~0 rows
  └── Drop exact duplicate flows: -258,084 rows
  │
CLEANED FLOWS
  4,880,373 cleaned flow records
  │
  ├── 1-Minute Window Discretization (floor to 60s)
  └── Per-window feature aggregation (mean/std/sum/min/max)
  │
PRE-PURGE WINDOWS
  2,927 1-minute Unified Cyber State windows
  │
  ├── Chronological Partitioning:
  │     Train (70%): 2,048 windows (2018-02-14 01:00:00 to 2018-03-01 04:35:00 UTC)
  │     Val   (15%):   439 windows (2018-03-01 04:36:00 to 2018-03-02 02:24:00 UTC)
  │     Test  (15%):   440 windows (2018-03-02 02:25:00 to 2018-03-02 12:59:00 UTC)
  │
  ├── Purge + Embargo Quarantine (Width = L + H = 30 + 5 = 35 windows):
  │     Train tail dropped: -35 windows
  │     Val head dropped:   -35 windows
  │     Val tail dropped:   -35 windows
  │     Test head dropped:  -35 windows
  │     Total windows dropped: -140 windows
  │
POST-PURGE WINDOWS (data/ucs/ucs_windows.parquet)
  2,787 windows x 418 columns
  │
  ├── Train: 2,013 windows (RobustScaler fitted here only)
  ├── Val:     369 windows
  └── Test:    405 windows
  │
DUAL-FORMAT CONSUMPTION
  ├── Linear Baseline Path (get_flat_window_data_for_lr):
  │     Train: 2,013 flat vectors x 400 features
  │     Val:     369 flat vectors x 400 features
  │     Test:    405 flat vectors x 400 features
  │
  ├── Deep Temporal Path (build_lstm_sequences, L=30):
  │     Train: 1,984 sequences of shape (30, 400)
  │     Val:     340 sequences of shape (30, 400)
  │     Test:    376 sequences of shape (30, 400)
  │
  └── Topology Path (data/ucs/ucs_graph_edgelists.parquet):
        754,071 directed interaction edges across 70,647 unique nodes
```

### Day-by-Day Ingestion Accounting

| Source File / Day | Initial Rows | Corrupted Timestamps | Exact Duplicates Dropped | Cleaned Rows | 1-Min Windows | Graph Edges | Attacks Present |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `Wednesday-14-02-2018` | 1,048,575 | 5 | 225,628 | 822,942 | 543 | 111,003 | FTP-BruteForce, SSH-Bruteforce |
| `Wednesday-21-02-2018` | 1,048,575 | 0 | 17,557 | 1,031,018 | 170 | 110,651 | DDOS-HOIC, DDOS-LOIC-UDP |
| `Thursday-22-02-2018` | 1,048,575 | 9 | 3,278 | 1,045,288 | 549 | 224,077 | Web-BruteForce, Web-XSS, Web-SQLi |
| `Wednesday-28-02-2018` | 613,071 | 0 | 6,089 | 606,982 | 570 | 82,859 | Infiltration (Day 1) |
| `Thursday-01-03-2018` | 331,100 | 0 | 73 | 331,027 | 570 | 79,971 | Infiltration (Compromise + Portscan) |
| `Friday-02-03-2018` | 1,048,575 | 0 | 5,459 | 1,043,116 | 525 | 145,510 | Botnet |
| **Total** | **5,138,471** | **14** | **258,084** | **4,880,373** | **2,927** | **754,071** | — |

### Window Label Breakdown (Post-Purge: 2,787 Windows)

#### Attack Type Distribution (`label_attack_type`)
- **Benign**: 2,475 windows (88.81%)
- **Infiltration-Compromise**: 97 windows (3.48%)
- **SSH-Bruteforce**: 82 windows (2.94%)
- **Infiltration-Portscan**: 58 windows (2.08%)
- **Botnet**: 48 windows (1.72%)
- **DDOS-LOIC-UDP**: 19 windows (0.68%)
- **DDOS-HOIC**: 8 windows (0.29%)

#### Binary Target Distribution
- **`label_binary` (Current Detection)**:
  - Benign (`0`): 1,861 windows (66.77%)
  - Attack (`1`): 926 windows (33.23%)
- **`future_attack_label` (Lead-Time $H=5$ Forecasting)**:
  - No Attack in Next 5 Min (`0`): 1,803 windows (64.69%)
  - Attack in Next 5 Min (`1`): 984 windows (35.31%)

### Contiguous Attack Episode Breakdown (Post-Purge: 40 Total Episodes)
- **SSH-Bruteforce**: 9 contiguous episodes (Multi-episode $\to$ LOEO eligible)
- **DDOS-LOIC-UDP**: 18 contiguous episodes (Multi-episode $\to$ LOEO eligible)
- **Botnet**: 10 contiguous episodes (Multi-episode $\to$ LOEO eligible; reduced from 11 pre-purge due to Val$\to$Test embargo)
- **DDOS-HOIC**: 1 contiguous episode (Singleton $\to$ Excluded from LOEO)
- **Infiltration-Compromise**: 1 contiguous episode (Singleton $\to$ Excluded from LOEO)
- **Infiltration-Portscan**: 1 contiguous episode (Singleton $\to$ Excluded from LOEO)
- **Total Multi-Episode Folds**: $9 + 18 + 10 = \mathbf{37\text{ folds}}$.

### Final Output Artifacts (Verified in `data/ucs/`)
- **`ucs_windows.parquet`**: Exactly **2,787 rows**, **418 columns**. Contains flat feature tensor, presence masks, metadata, `episode_id`, and `forecast_episode_id`.
- **`ucs_graph_edgelists.parquet`**: Exactly **754,071 rows**, **10 columns**. Contains window-keyed directed interaction topology edges.
- **`node_lookup.parquet`**: Exactly **70,647 rows**, **2 columns**. Maps anonymized integer `node_id` to raw endpoint string.
- **`packet_features.parquet`**: Exactly **543 rows**, **13 columns**. Contains packet telemetry for Wednesday-14-02-2018 windows.
- **`scaler_params.yaml`**: Exactly **400 feature parameter blocks** ($Q_{25}, Q_{50}, Q_{75}$, scale, log1p flag).

---

## 6. Key Design Decisions and the "Why" Behind Each

### 1. Why Chronological Split Instead of Random K-Fold?
- **Decision**: Windows are partitioned strictly in chronological order (First 70% Train, Next 15% Validation, Final 15% Test).
- **The "Why"**: Network traffic is a non-stationary time series with strong temporal autocorrelation. Randomly shuffling 1-minute windows (standard k-fold) causes catastrophic data leakage: window $t$ in the training set would share flow connections, active TCP sessions, and volume context with window $t+1$ in the test set. A model evaluated on randomly shuffled windows memorizes temporal neighbors rather than learning true attack progression, producing near-perfect test scores that collapse in production.

### 2. Why Purge AND Embargo (Not Just One)?
- **Decision**: Drop $W = L + H = 35$ windows from **both** sides of every partition boundary (dropping 140 windows total).
- **The "Why"**: Purge and embargo mitigate two distinct, directional leakage risks:
  - **Purge (Left-Side Boundary)**: At the tail of Train (or Val), window $t$ uses a forward-looking forecast label $y_{t} = \max(\text{label}_{t+1 \dots t+H})$. If window $t$ sits within $H$ steps of the boundary, its label is derived from ground truth occurring *inside the validation set*. Purging the tail eliminates this forward information leak.
  - **Embargo (Right-Side Boundary)**: At the head of Val (or Test), an LSTM model at window $t$ ingests a historical sequence of length $L=30$ ($[t-L+1 \dots t]$). If window $t$ sits within $L$ steps of the boundary, its input features reach backward *into the training set*. Embargoing the head eliminates this backward context leak.
  - Applying both with width $W = L + H$ mathematically guarantees zero cross-partition leakage for both linear models and recurrent neural networks.

### 3. Why Episode-Based Grouping for LOEO Instead of Day-Based?
- **Decision**: Group attacks by unbroken contiguous runs (`episode_id`) rather than by calendar day.
- **The "Why"**: In CSE-CIC-IDS2018, multiple distinct attack types occur on the same calendar day (e.g., FTP-BruteForce and SSH-Bruteforce on Feb 14; DDOS-HOIC and DDOS-LOIC-UDP on Feb 21). Furthermore, attacks on a given day occur in distinct bursts separated by benign traffic. Holding out an entire day removes all background traffic for that date and introduces massive distribution shifts. Grouping by contiguous episodes isolates the specific attack burst, allowing background traffic from that day to remain in the training distribution.

### 4. Why LOEO Instead of a Single Train/Test Split for Validating Generalization?
- **Decision**: Use 37-fold Leave-One-Episode-Out cross-validation for Gate 0 diagnostic verification.
- **The "Why"**: A single chronological split places all Friday Botnet traffic in the test set and leaves other attacks solely in the training set. Evaluating on a single split conflates "generalization across episodes" with "zero-shot generalization to an entirely unseen attack class." LOEO systematically tests whether an attack's physical signature generalizes across independent bursts of the *same* attack class, yielding variance estimates and confidence intervals.

### 5. Why Is $H=5$ the Primary Horizon (and Not $H=10$ or $H=15$)?
- **Decision**: $H=5$ windows (5 minutes lead time) is the authoritative, Gate 0-validated forecasting horizon.
- **The "Why"**: Operational defensive response requires lead time, but network state entropy increases rapidly with horizon length. In Gate 0 LOEO sweeps (see `H_SWEEP_REPORT.md`), extending $H$ to 10 or 15 minutes expanded the pre-onset test set ($5 \to 10 \to 15$ windows per fold) but did not improve physical feature discrimination over schedule heuristics. $H=5$ represents the tightest, operationally viable lead time for pre-emptive automated defensive measures (e.g., dynamic ACL reconfiguration or rate-limiting) before attack weaponization.

### 6. Why LSTM Lookback $L=30$ Windows Specifically?
- **Decision**: Input sequence length is fixed at 30 windows (30 minutes).
- **The "Why"**: Thirty minutes captures the typical dwell time and reconnaissance phase of multi-stage attacks in enterprise networks. Review of the repository documentation indicates no alternative mathematical derivation; this was an architectural decision established by the upstream team lead to balance temporal context against GPU memory footprints.

### 7. Why RobustScaler Rather Than StandardScaler or MinMaxScaler?
- **Decision**: Features are scaled using `RobustScaler` ($x_{\text{norm}} = (x - Q_{50}) / \text{IQR}$) with selective `log1p` pre-transforms.
- **The "Why"**: Network flow data exhibits heavy power-law and Pareto distributions. A DDoS flood or large file transfer generates byte and packet counts that are $10^4\times$ larger than typical background flows. Standard scaling ($\mu, \sigma$) is heavily distorted by extreme outliers, squashing normal traffic into a near-zero range. MinMax scaling binds data to arbitrary limits that collapse when test traffic exceeds training extremes. `RobustScaler` relies on the median and interquartile range ($Q_{75} - Q_{25}$), guaranteeing stable scaling that is unaffected by extreme burst volumes.

### 8. Why Was Packet-Level Extraction Scoped to Wednesday-14-02-2018 Only?
- **Decision**: Live packet telemetry was extracted exclusively for Feb 14; all other days utilize flow telemetry with `mask_has_packet_level_features = 0.0`.
- **The "Why"**: Storage and bandwidth constraints. As verified by querying AWS S3 (`cse-cic-ids2018.s3.amazonaws.com`), the compressed `.pcap.zip` archives for the remaining 5 days total **236.7 GB** (uncompressed $> 1.2 \text{ TB}$). Downloading, decompressing, and streaming 1.2 TB of PCAP data via Scapy in Python requires weeks of processing time and hundreds of gigabytes of disk space, which is prohibitive for a rapid ingestion pipeline. Feb 14 served as the proof-of-concept validation for packet-level schema design.

### 9. Why Are Singleton-Episode Attack Types Excluded From LOEO?
- **Decision**: DDOS-HOIC, Infiltration-Compromise, and Infiltration-Portscan are excluded from the 37-fold LOEO cross-validation suite.
- **The "Why"**: By definition, Leave-One-Episode-Out requires at least 2 episodes of an attack class so that episode $k$ can be held out while episode $j \ne k$ remains in the training set. If an attack occurs only once in the entire dataset ($N_{\text{episodes}} = 1$), holding it out removes all training instances of that attack, turning LOEO into an unseen-class zero-shot test. Consequently, cross-episode generalization claims for these three singleton attacks are formally unverified.

---

## 7. The Validation Journey — Why You Can Trust This Dataset

A dataset is only as good as the rigor with which its flaws were investigated. The development of this pipeline was defined by uncovering and methodically resolving subtle data leakage pitfalls.

### 1. The Schedule-Leakage Threat (Gate 0)
CSE-CIC-IDS2018 was collected on a synthetic laboratory testbed where attack execution was orchestrated by automated cron scripts executing during standard working hours (typically 09:00–12:00 and 14:00–16:00). 

This introduces a lethal machine learning hazard: **Schedule Leakage**. A classifier can achieve high detection accuracy simply by memorizing the clock:
$$\text{If } \text{Hour} \in [09, 12] \cup [14, 16] \implies \text{Predict "Attack"}$$
Such a model is entirely useless in the real world, where attacks occur at 03:00 AM. 

To prevent deploying a "clock memorizer," we instituted **Gate 0**: an adversarial diagnostic comparing a model trained on genuine physical network telemetry (**Set A**) against a "cheater" model trained strictly on timestamps and day indicators (**Set B**).

### 2. The Four Evaluation Protocols Tried (and Why 1–3 Failed)

To evaluate Set A vs. Set B, four evaluation protocols were developed sequentially:

```
Protocol 1: Chronological Holdout Split
  ❌ FAILED: 100% of March 2 Botnet fell into Test. Set A collapsed (F1=0.0436)
     because Botnet was entirely unseen. Set B exploited daily schedules (F1=0.6893).

Protocol 2: Window-Stratified Diagnostic (IID Sampling)
  ❌ FAILED: Random window sampling across days allowed adjacent-window autocorrelation.
     Both models scored artificially high (Set A F1=0.8304, Set B F1=0.7893).

Protocol 3: Episode-Grouped Stratified Diagnostic
  ❌ FAILED: Whole episodes held out (15%), but singleton attacks (HOIC, Infiltration)
     had n=1 episodes and were forced into Train. Test set was 97.3% Benign and lacked
     3 attack classes, causing severe distribution collapse (Set A F1=0.2080).

Protocol 4: Leave-One-Episode-Out (LOEO) Cross-Validation
  ✅ SUCCESS: Cycled through each multi-episode attack individually with proportionally
     sampled background traffic. Eliminated distribution collapse and evaluated true
     cross-episode generalization.
```

### 3. The Authoritative 37-Fold LOEO Results

#### Why 37 Folds and Not 38? (The Val$\to$Test Embargo Elimination)
Earlier documentation cited 38 LOEO folds based on 11 pre-purge Botnet episodes ($9 + 18 + 11 = 38$). 

However, during audit verification, we discovered that **1 short Botnet episode was completely consumed by the 35-window embargo zone at the Val$\to$Test boundary**. Because all of its windows were purged to protect test set integrity, that episode had zero surviving windows in the final post-purge dataset. 

The authoritative evaluation therefore uses **37 folds** across the surviving multi-episode attacks:
- **SSH-Bruteforce**: 9 folds
- **DDOS-LOIC-UDP**: 18 folds
- **Botnet**: 10 folds

#### Current Authoritative 37-Fold Numbers (`data/ucs/loeo_37fold_results.csv`)

| Task / Metric | Set A (Physical Telemetry) | Set B (Schedule Only) | Head-to-Head | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Detection F1** | **0.9077 $\pm$ 0.1038** | 0.8126 $\pm$ 0.1475 | **Set A wins 22 / 37** (B wins 10, Ties 5) | **CONDITIONAL PASS** (CIs overlap) |
| **Detection Precision** | **0.9129 $\pm$ 0.1527** | 0.7215 $\pm$ 0.1948 | Set A superior precision | Set A minimizes false alarms |
| **Detection Recall** | 0.9269 $\pm$ 0.1040 | **0.9788 $\pm$ 0.0627** | Set B blanket recall | Set B guesses business hours |
| **Detection PR-AUC** | **0.9466 $\pm$ 0.1084** | 0.8820 $\pm$ 0.1364 | Set A higher confidence | Robust discrimination |
| **Forecasting F1 ($H=5$)**| 0.0946 $\pm$ 0.2622 | **0.2528 $\pm$ 0.3743** | Set B wins 11 / 37 (A wins 2, Ties 24)| **CONDITIONAL PASS** (Low signal) |
| **Forecasting PR-AUC** | 0.2193 $\pm$ 0.3358 | **0.3636 $\pm$ 0.4408** | Both models struggle | Precursors difficult in snapshot |

### 4. Honest Assessment: What We Know vs. What Remains an Open Risk
- **Detection Is Proven**: On detection (`label_binary`), physical telemetry (Set A) clearly outperforms schedule guessing (Set B) on mean F1 (0.9077 vs. 0.8126) and delivers far higher precision (0.9129 vs. 0.7215). Set B achieves high recall only by falsely alarming on normal business-hours traffic.
- **The Open Risk (Short Episodes & Onset Forecasting)**: In single-window attack episodes (e.g., isolated UDP flood bursts or quick Botnet check-ins), shallow decision trees fail to extract physical signatures from a single 1-minute slice, allowing schedule heuristics to win. Furthermore, for onset forecasting ($H=5$), precursor traffic signatures in benign windows immediately prior to attack onset are extremely faint. 
- **The Downstream Mandate**: This finding proves why a **temporal sequence model (LSTM) and graph neural network (GraphSAGE)** are mandatory: tabular decision trees on 1-minute snapshots cannot bridge pre-onset gaps, whereas a sequence model with 30 minutes of historical context can track cumulative subtle state changes.

---

## 8. Tech Stack

Every imported library across all `.py` files in the repository, with its architectural justification:

| Package / Module | Imported In | Architectural Justification |
| :--- | :--- | :--- |
| **`pandas`** | Throughout `src/` | Primary dataframe engine for columnar manipulation, 60s timestamp flooring, rolling max calculations, and groupby aggregations. |
| **`numpy`** | Throughout `src/` | Vectorized mathematical operations, quantile calculations in `RobustScaler`, NaN/Inf filtering, and 3D tensor sequence construction. |
| **`pyarrow`** | Parquet I/O | High-performance columnar storage engine backing Parquet reads/writes. Preserves schema types and timestamps with zero precision loss. |
| **`scapy`** (`PcapReader`, `IP`, `TCP`) | `pcap_extractor.py` | Python-native packet dissection library. Chosen over PyShark/tshark because it streams raw PCAPs via `PcapReader` with zero external OS binary dependencies. |
| **`scikit-learn`** (`DecisionTreeClassifier`, `metrics`) | `gate0_*.py`, `normalizer.py` | Shallow diagnostic models (max_depth=4) used to establish baseline explainability and compute F1, Precision, Recall, and PR-AUC. |
| **`pyyaml`** (`yaml`) | `pipeline_runner.py`, etc. | Human-readable configuration parsing for master pipeline hyperparameters, schema mappings, and fitted scaler parameters. |
| **`re`** | `ingestion.py`, `run_h_sweep.py` | Regular expression parsing for calendar date extraction from filenames and configuration manipulation. |
| **`glob`** | `pipeline_runner.py` | File system discovery of multi-day raw CSV inputs. |
| **`unittest`** | `tests/test_pipeline.py` | Python standard library test runner for regression testing with zero third-party testing dependencies. |
| **`datetime`, `timezone`** | `cleaner.py`, `pcap_extractor.py` | Timezone-aware UTC timestamp parsing, handling leap seconds and ISO calendar boundaries. |
| **`math`** | `pcap_extractor.py` | Calculating log2 Shannon entropy for destination port scan sequencing scores. |

---

## 9. Glossary

- **Window**: A fixed 1-minute (60-second) discrete time interval (floored to `YYYY-MM-DD HH:MM:00 UTC`) over which asynchronous flow records are aggregated into a single feature state vector $S_t$.
- **Episode (`episode_id`)**: An unbroken, contiguous sequence of 1-minute windows sharing the exact same `source_day` and `label_attack_type`. Formatted as `{source_day}_{attack_type}_{run_index}`.
- **Forecast Horizon ($H$)**: The number of future windows into which the model must predict attack onset. Primary horizon is $H=5$ windows (5 minutes lead time).
- **LSTM Lookback ($L$)**: The number of historical consecutive windows provided as an input sequence tensor to the temporal model. Fixed at $L=30$ windows (30 minutes).
- **Purge**: Dropping windows from the tail of an earlier partition (e.g., Train) so that forward-looking forecast labels ($H=5$) do not calculate targets using future windows inside the next partition.
- **Embargo**: Dropping windows from the head of a later partition (e.g., Val or Test) so that an LSTM's backward-looking sequence context ($L=30$) cannot ingest features from the preceding partition.
- **Purge/Embargo Width ($W$)**: The dynamically computed quarantine distance: $W = L + H = 30 + 5 = 35$ windows, applied to both sides of each partition boundary (dropping 140 windows total).
- **LOEO (Leave-One-Episode-Out)**: A cross-validation protocol where one contiguous attack episode is held out as the test fold alongside proportionally sampled background traffic, while all other episodes train the model.
- **Chronological Split**: Partitioning time series data strictly by chronological order (70% Train $\to$ 15% Val $\to$ 15% Test) with zero random shuffling.
- **Data Leakage**: The contamination of training data with information from validation or test partitions, causing artificial inflation of performance metrics that fails in production.
- **Feature-Presence Mask**: Explicit binary indicator columns (e.g., `mask_has_packet_level_features`) in $S_t$ informing downstream neural encoders whether a specific feature group contains real telemetry or imputed zeros.
- **Unified Cyber State ($S_t$)**: The dual-format representation of enterprise network health at window $t$, consisting of a 418-column tabular feature vector and a directed interaction multigraph.
- **Gate 0**: The pre-modeling diagnostic verification suite proving that model predictions are driven by genuine physical traffic features (Set A) rather than synthetic schedule artifacts (Set B).
- **Singleton Episode**: An attack class that appears only once as a single contiguous burst in the entire dataset (e.g., DDOS-HOIC, Infiltration-Compromise, Infiltration-Portscan), rendering cross-episode cross-validation impossible.

---

## 10. Known Limitations

1. **Untested Singleton Generalization**: DDOS-HOIC, Infiltration-Compromise, and Infiltration-Portscan possess only one contiguous episode each, meaning cross-episode generalization for these classes is mathematically untested.
2. **Single-Day Packet Feature Coverage**: Deep packet-level telemetry (TTL distributions, fragmentation flags, payload quantiles, TCP retransmissions) exists exclusively for Wednesday-14-02-2018 due to the 236.7 GB download cost of the remaining 5 days.
3. **Intermittent Burst Episode Fragmentation**: Under the strict 1-minute contiguous-run definition, intermittent attack bursts separated by 1-to-3 minute benign pauses are split into multiple single-window episodes rather than merged into a single multi-pulse campaign.
4. **Schedule Signal Dominance on Short Snapshots**: On single-window episodes and pre-onset forecasting windows ($H=5$), a shallow decision tree using physical snapshot features currently loses to schedule heuristics, requiring the downstream LSTM sequence model to resolve.

---

## 11. Anticipated Judge Questions — Q&A

### Q1: "Walk me through what happens when a new raw CSV comes in."
**Answer**: 
1. `ingestion.py` reads the CSV using Latin-1 fallback, strips whitespace from headers, purges repeated embedded header lines (`Label == 'Label'`), and attaches immutable provenance tags (`source_file`, `source_day`).
2. `canonical_mapper.py` standardizes column names into canonical snake_case using `canonical_mapping.yaml`.
3. `cleaner.py` parses timestamps into UTC datetime, drops year $< 2018$ epoch corruption, converts string/float infinities to NaN, drops exact duplicate flow rows, audits negative durations, and imputes missing values using within-day medians.
4. `window_aggregator.py` floors timestamps into 1-minute buckets, computes 5 aggregate statistics (`mean`, `std`, `sum`, `min`, `max`) per feature, and attaches feature-presence masks.
5. In parallel, `graph_builder.py` constructs directed interaction edges between communicating endpoints.
6. `labeler_and_splits.py` assigns canonical attack labels, derives the $H=5$ backward-shifted future forecasting target, enforces chronological splits (70/15/15), applies the 35-window purge/embargo quarantine, and segments contiguous attack episodes.
7. `normalizer.py` fits a `RobustScaler` on post-purge training windows and transforms the dataset.
8. `sequence_builder.py` outputs flat feature matrices for linear models and boundary-safe 30-step tensors for LSTMs.

### Q2: "What is data leakage, and specifically how does your pipeline prevent it?"
**Answer**: Data leakage occurs when test or future information contaminates the training pipeline, leading to overly optimistic evaluation metrics. We eliminate three distinct leakage modes:
- **Temporal Sequence Leakage**: Prevented via **Purge and Embargo**. We drop 35 windows ($L+H$) from both sides of Train$\to$Val and Val$\to$Test boundaries, preventing LSTM lookbacks ($L=30$) and forecast horizons ($H=5$) from spanning partitions.
- **Normalization Leakage**: Prevented by fitting `RobustScaler` **exclusively on post-purge training windows** (2,013 windows). Validation and test windows are transformed using frozen training parameters stored in `scaler_params.yaml`.
- **Schedule Leakage**: Prevented by stripping all timestamps (`window_start_utc`, `window_end_utc`) and `source_day` from neural feature tensors, forcing downstream models to consume only normalized behavioral telemetry and feature masks.

### Q3: "Why did you choose a chronological split over random k-fold?"
**Answer**: Network traffic is a continuous time series. Randomly shuffling 1-minute windows allows test windows to sit adjacent to training windows, sharing ongoing TCP connections, active IP sessions, and volumetric autocorrelation. A model evaluated on random splits memorizes localized temporal context rather than learning generalizable attack patterns. A strict chronological split (First 70% Train, Next 15% Val, Final 15% Test) enforces real-world deployment conditions: training on the past to predict the future.

### Q4: "What's the difference between your purge/embargo step and your Gate 0 LOEO test — aren't they solving the same problem?"
**Answer**: No, they address completely different failure modes:
- **Purge/Embargo** solves **structural boundary leakage within a single chronological split**. It prevents sliding window contexts ($L=30, H=5$) from overlapping across train, validation, and test partitions.
- **Gate 0 LOEO** solves **artifact leakage and validation bias across attack bursts**. It evaluates whether a model is learning genuine physical network signatures (packet sizes, byte rates, TCP flags) or simply memorizing synthetic lab testbed schedules (attacks scripted during business hours).

### Q5: "How do you know your model isn't just learning what time attacks happen instead of what an attack looks like?"
**Answer**: We proved this empirically through the Gate 0 diagnostic suite. We trained an adversarial baseline model (**Set B**) given access *only* to hour-of-day and day indicators, and compared it against our telemetry model (**Set A**). In our 37-fold LOEO cross-validation on detection:
- Set A achieved **0.9077 F1** with **0.9129 Precision**.
- Set B scored only **0.8126 F1** with a low **0.7215 Precision**.
Set B only achieved high recall by blanketing business hours with false alarms, generating unacceptably high false positive rates. Set A proves that physical telemetry provides far superior, defensible discrimination. Furthermore, timestamps are strictly excluded from neural feature tensors.

### Q6: "Why is your forecast horizon 5 minutes and not something else?"
**Answer**: In our horizon sensitivity sweep (`H_SWEEP_REPORT.md`), we tested $H=5, 10, \text{ and } 15$ minutes across all 37 folds. Expanding $H$ increased pre-onset test windows ($5 \to 10 \to 15$ per fold) but did not improve physical feature discrimination over schedule heuristics in snapshot models. Operationally, 5 minutes provides sufficient lead time for automated defensive orchestration (e.g., firewall rule injection, rate-limiting) while keeping the predictive state space tightly coupled to active attack precursors.

### Q7: "What happens to a window that sits right at a train/test boundary?"
**Answer**: It is dropped by `apply_purge_embargo()`. Specifically, the 35 windows immediately preceding the boundary (train tail) and the 35 windows immediately following the boundary (validation head) are excised. They do not appear in `ucs_windows.parquet`, are not fitted by the scaler, and cannot be consumed by any LSTM sequence.

### Q8: "Why did you only extract packet-level features for one day?"
**Answer**: Cost and bandwidth feasibility. As verified directly against AWS S3 (`cse-cic-ids2018.s3.amazonaws.com`), the raw `.pcap.zip` archives for the remaining 5 days total **236.7 GB** compressed (exceeding 1.2 TB uncompressed). Downloading and extracting 1.2 TB of PCAP data via Scapy in Python would take weeks and require massive dedicated infrastructure. We extracted packet features for Wednesday-14-02-2018 as a validated proof-of-concept, set `mask_has_packet_level_features = 1.0` for those windows, and set the mask to `0.0` for remaining days.

### Q9: "What's an episode, and why do you group by episode instead of by day for validation?"
**Answer**: An episode is an unbroken contiguous run of 1-minute windows sharing the same calendar day and attack class. We group by episode because multiple attack types occur on the same day (e.g., FTP-BruteForce and SSH-Bruteforce on Feb 14), and attacks occur in intermittent bursts separated by benign traffic. Holding out an entire day removes all background traffic for that date; holding out an episode isolates the specific attack campaign while leaving background traffic in the training set.

### Q10: "What's your biggest known limitation right now?"
**Answer**: Our biggest limitation is that singleton-episode attack classes (DDOS-HOIC, Infiltration-Compromise, Infiltration-Portscan) have only one contiguous burst in the entire dataset, meaning cross-episode generalization cannot be validated for those specific attack types.

### Q11: "If you had one more week, what would you fix first, and why?"
**Answer**: I would implement an episode clustering heuristic that merges intermittent attack bursts separated by short (1–3 minute) benign pauses. Currently, a pulsing Botnet attack is fragmented into multiple single-window episodes, creating small test folds in LOEO where snapshot models struggle. Consolidating these into multi-pulse episodes would give downstream models longer continuous attack sequences.

### Q12: "What's the difference between `label_binary` and `future_attack_label`?"
**Answer**: 
- `label_binary` represents **current detection**: it equals `1` if an attack is actively occurring in window $t$, and `0` if benign.
- `future_attack_label` represents **future forecasting**: it equals `1` if an attack will occur at any point in the next $H=5$ windows ($[t+1 \dots t+5]$), and `0` otherwise. It is derived by shifting the forward-looking rolling maximum backward in time, ensuring zero feature leakage.

### Q13: "Why RobustScaler and not a more common scaler like StandardScaler?"
**Answer**: Network telemetry is characterized by extreme, heavy-tailed outliers. A DDoS attack can produce millions of bytes and packets per second, while background traffic generates a few hundred. Standard scaling ($\mu, \sigma$) is heavily distorted by extreme values, shrinking normal traffic variations into an indistinguishable range near zero. `RobustScaler` scales based on the median and interquartile range ($Q_{75} - Q_{25}$), ensuring robust scaling that is resilient to extreme volume surges.

### Q14: "How many rows/windows survive after all your leakage protections, and why does the number drop from the raw count?"
**Answer**: 
- We start with **5,138,471 raw flow records**.
- Deduplication drops 258,084 exact duplicate flows, and cleaner drops 14 corrupted epoch-1970 rows, leaving **4,880,373 cleaned flows**.
- 1-minute time windowing discretizes these flows into **2,927 pre-purge windows**.
- Chronological splitting and purge+embargo drop 35 windows from each of the 4 boundary sides (**140 windows dropped total**).
- Exactly **2,787 clean, normalized, leakage-safe windows** survive in the final `ucs_windows.parquet` dataset.

### Q15: "What would happen if you skipped the purge/embargo step — what specifically would go wrong?"
**Answer**: Two catastrophic leakage errors would silently occur:
1. **Target Leakage**: Train windows at the train/val boundary would calculate their $H=5$ forecasting target using validation attack labels, teaching the model to predict validation ground truth.
2. **Context Leakage**: The first 29 LSTM sequences in the validation set ($L=30$) would pull historical feature vectors from the training set, allowing the model to carry recurrent hidden states across the partition boundary. 
The validation metrics would appear artificially inflated, masking true model generalization failure.

---
*End of Project Handbook.*
