# Hybrid Smart Grid Forecasting — TFT + LSTPrompt

> A hybrid time-series forecasting pipeline combining deep learning and large language model reasoning for household electricity consumption prediction.

---

## Overview

This project was developed as part of a summer internship exploring the intersection of **time-series forecasting** and **large language models (LLMs)** applied to smart energy systems.

Accurately predicting high-frequency energy measurements — such as minute-level voltage readings — is a critical challenge in smart grid optimization. The data presents complex temporal dependencies and significant noise, making it a non-trivial forecasting problem.

To address this, we designed and implemented a **hybrid forecasting pipeline** that combines:

- **Temporal Fusion Transformer (TFT)** — a state-of-the-art deep learning model for capturing long-range temporal patterns in multivariate time-series data.
- **LSTPrompt** — an LLM-based framework that performs zero-shot forecasting through structured natural language reasoning, used to refine and enhance TFT's preliminary predictions.

The result is a system that leverages the complementary strengths of both paradigms: TFT's precision in learning temporal structure, and LSTPrompt's interpretable reasoning capabilities.

---

## Pipeline Architecture

```
Raw Energy Data
      │
      ▼
Data Preprocessing
  (cleaning, normalization, windowing)
      │
      ▼
TFT Model — Preliminary Forecasts
      │
      ▼
LSTPrompt — LLM-based Refinement
  (prompt engineering + zero-shot reasoning)
      │
      ▼
Enhanced Predictions + Evaluation
```

---

## Features

- **Data preprocessing** — cleaning, normalization, and sliding-window construction for minute-level voltage measurements
- **TFT integration** — configurable training pipeline for the Temporal Fusion Transformer
- **LSTPrompt framework** — prompt engineering workflow that feeds TFT outputs into an LLM for zero-shot forecast refinement
- **Evaluation** — structured metrics (MAE, RMSE, MAPE) for comparing baseline vs. hybrid forecasts
- **Interpretability** — LLM-generated reasoning traces alongside numerical predictions

---

## Motivation

The growing adoption of smart meters and IoT sensors in energy infrastructure produces vast streams of high-frequency time-series data. Forecasting household electricity consumption enables:

- **Demand response** optimization
- **Load balancing** in smart grids
- **Anomaly detection** for unusual consumption patterns
- **Energy cost reduction** through better scheduling

Standard deep learning models often lack interpretability. This project explores whether LLM reasoning can bridge that gap — improving both accuracy and explainability in a production-relevant forecasting context.

---

## Tech Stack

| Component | Technology |
|---|---|
| Deep learning model | Temporal Fusion Transformer (PyTorch / PyTorch Forecasting) |
| LLM framework | LSTPrompt |
| Data processing | Python, Pandas, NumPy |
| Evaluation | scikit-learn metrics |
| Notebook environment | Jupyter |

---

## Getting Started

### Prerequisites

```bash
Python >= 3.9
pip install -r requirements.txt
```

### Installation

```bash
git clone https://github.com/NadaBhm/Hybrid-model-Smart-Grid-Forecasting.git
cd Hybrid-model-Smart-Grid-Forecasting
pip install -r requirements.txt
```

### Usage

1. **Preprocess the data**
   ```bash
   python preprocess.py --input data/raw/ --output data/processed/
   ```

2. **Train the TFT model**
   ```bash
   python train_tft.py --config configs/tft_config.yaml
   ```

3. **Run LSTPrompt refinement**
   ```bash
   python lstprompt_refine.py --predictions outputs/tft_predictions.csv
   ```

4. **Evaluate results**
   ```bash
   python evaluate.py --results outputs/hybrid_predictions.csv
   ```

> **Note:** Update the config files in `configs/` with your dataset paths and model hyperparameters before running.

---

## Dataset

The pipeline was tested on a household electricity consumption dataset containing minute-level voltage measurements. The data includes complex temporal patterns typical of real smart meter deployments.

If you wish to reproduce the experiments, place your dataset under `data/raw/` following the format described in `data/README.md`.

---

## Results

The hybrid TFT + LSTPrompt pipeline demonstrates improved forecasting performance and interpretability compared to the TFT baseline alone. Detailed results, plots, and metric comparisons are available in the `notebooks/` directory.

---

## Keywords

`time-series forecasting` · `large language models` · `temporal fusion transformer` · `LSTPrompt` · `electricity consumption` · `smart grid` · `hybrid model` · `zero-shot forecasting`

---

## Authors

Developed during a summer internship focused on AI applications in smart energy systems.

---

## License

This project is open-source. See [LICENSE](LICENSE) for details.
