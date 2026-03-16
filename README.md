# Cleaning Detection

Detection of cleaning events from sensor/device data using multiple ML approaches (Gaussian Mixture Model, Variational Autoencoder, Fast Fourier Transform, Isolation Forest) and report-summary labeling.

## Overview

This project processes events from the **PulseListener API** and labeled report summaries (CSV) to detect and predict cleaning events. **theo**, **alvin**, and **nose-cap** are injection mold machines that create different parts; each has its own folder with machine-specific scripts and data.

## Project structure

Each machine folder (`theo/`, `alvin/`, `nose-cap/`) contains a similar set of scripts:

| Script | Purpose |
|--------|--------|
| `api.py` | Fetch events from PulseListener API |
| `clean.py` | Match report-summary cleaning labels to feature data and produce labeled CSVs |
| `add_events_to_csv.py` | Add event information to feature CSVs |
| `gmm.py` | Gaussian Mixture Model for cleaning detection |
| `vae_prediction.py` | Variational Autoencoder–based prediction |
| `ftt_prediction.py` | Fast Fourier Transform–based prediction |
| `isolation_forest_prediction.py` | Isolation Forest outlier-based detection |
| `outliers.py` | Outlier analysis |
| `optimize.py` | Hyperparameter / detection optimization |
| `plot_raw_probabilities.py` | Plot raw model probabilities |
| `plot_probabilities_over_time.py` | Plot probabilities over time |
| `analyze_fft_before_after_cleaning.py` | Fast Fourier Transform analysis around cleaning events |
| `time_predictions.py` | Time-windowed predictions |
| `generate_features_with_events.py` | (nose-cap) Generate features with event metadata |

## Requirements

- Python 3
- pandas, numpy
- scikit-learn (Gaussian Mixture Model, Isolation Forest, preprocessing)
- tensorflow/keras (Variational Autoencoder)
- matplotlib
- requests (API calls)

Install with:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib requests
```

## Data

- **Feature/label CSVs**: e.g. `24-121_all_features_labels.csv` (paths may differ per folder).
- **Report Summary CSV**: e.g. `Report Summary(24-121 Helimix Cup 2 cavity).csv` — used by `clean.py` to extract cleaning timestamps and label events.

Scripts assume these files (or equivalents) are present in the same directory when run, or paths are adjusted in the scripts.

## Usage

1. **Fetch events** (from the correct machine folder):
   ```bash
   cd theo   # or alvin / nose-cap
   python api.py
   ```
   Configure `device_id` and date range inside `api.py` as needed.

2. **Label data from report summary**:
   ```bash
   python clean.py
   ```
   Uses the report summary to find cleaning times and align labels with the feature CSV.

3. **Train / run models** (from the same folder):
   ```bash
   python gmm.py
   python vae_prediction.py
   python ftt_prediction.py
   python isolation_forest_prediction.py
   ```
   Exact usage (e.g. input file names, CLI args) may vary; check the top of each script.

4. **Visualize**:
   ```bash
   python plot_raw_probabilities.py
   python plot_probabilities_over_time.py
   ```

## API

Events are fetched from:

- `https://pulselistener.azurewebsites.net/api/fetchEvents`

Parameters: `device`, `from`, `to` (date range). See `api.py` in any variant for the exact interface.