# AIT-LDS v2.0 Feasibility Findings

**Dataset**: AIT Log Data Set V2.0 (DOI 10.5281/zenodo.5789064)
**Licence**: CC BY-NC-SA 4.0

## G-t1: Feasibility Inspection Answer
**NO** — AIT-LDS v2.0 (russellmitchell) has no enterprise-network captures, only 2 attacker-side PCAPs; ShadowCat's network-flow world model cannot be trained or evaluated on it.

## Transition Counts
- Only 9 `attackA->attackB` transitions were found.

## Subsequent Tasks
G-t2/G-t3: descriptive only (windows from 2 attacker-side PCAPs; 388/406 features absent, masked as NaN).
G-t4–G-t6: reverted (in-sample evaluation, zero-filled features).
