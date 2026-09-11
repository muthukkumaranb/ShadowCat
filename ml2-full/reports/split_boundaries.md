# Temporal Split Boundaries (Section 30-32)

Total windows: 561

## Scope note
This dataset spans only ~9.33 hours. This split validates the MECHANICS of chronological split + purge + embargo, per team lead direction — it is not a production-scale train/val/test split.

## Boundaries
- Train: 2016-08-29 00:07:00+00:00 to 2016-08-29 06:38:00+00:00
- Val start: 2016-08-29 06:39:00+00:00
- Val: 2016-08-29 06:39:00+00:00 to 2016-08-29 08:02:00+00:00
- Test start: 2016-08-29 08:03:00+00:00
- Test: 2016-08-29 08:03:00+00:00 to 2016-08-29 09:27:00+00:00

## Purge (Section 31)
- Train rows purged: 9
- Val rows purged: 9

## Embargo (Section 32)
- Val rows embargoed (until 2016-08-29 07:09:00+00:00): 30
- Test rows embargoed (until 2016-08-29 08:33:00+00:00): 30

## Final usable counts
- train: 383 usable / 392 total
- val: 45 usable / 84 total
- test: 55 usable / 85 total
