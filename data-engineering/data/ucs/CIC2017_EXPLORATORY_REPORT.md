# CIC-IDS2017 Exploratory Report

## Download Failure & Blocker
The exploratory investigation into the CIC-IDS2017 dataset is currently blocked. Real CIC-IDS2017 data could not be obtained in this environment for the following reasons:
1. **Canonical Source Unreachable**: The canonical UNB dataset link (`http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV/MachineLearningCVE/`) no longer hosts the direct CSV files at this path and instead redirects to an HTML webpage for the Canadian Institute for Cybersecurity, preventing automated download.
2. **Kaggle API Unauthorized**: While the `kaggle` package was successfully installed, the Kaggle API cannot be used because valid authentication credentials (`~/.kaggle/kaggle.json`) are missing from this environment.

As a result, no real CIC-IDS2017 files were downloaded. 

In accordance with strict data integrity requirements, no other dataset's data (such as CSE-CIC-IDS2018) was substituted, relabeled, or synthesized to fabricate a result. All previous fabricated data has been deleted.

Until real CIC-IDS2017 data can be successfully downloaded and its column headers/file sizes verified against the published schema, no schema mapping, leakage testing, or cross-dataset generalization evaluation can be performed. The integration is paused pending the availability of the real dataset.
