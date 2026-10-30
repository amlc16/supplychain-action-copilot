# SupplyChain Action Copilot

An interactive machine learning application for identifying potentially late deliveries and helping supply chain managers prioritize orders for review.

**Live Demo:** [SupplyChain Action Copilot](https://supplychain-action-copilot.streamlit.app/)  
**Repository:** [GitHub](https://github.com/amlc16/supplychain-action-copilot)

## Overview

SupplyChain Action Copilot combines predictive analytics, interactive data visualization, and locally hosted generative AI to support shipment-risk analysis.

The application uses historical order data to estimate the probability of late delivery. Users can explore risk distributions, adjust the risk threshold, review prioritized orders, and export results for further analysis.

An optional LLM component uses Qwen 3.5 4B through Ollama to generate contextual recommendations for individual orders.

## Key Features

- **Delivery Risk Prediction:** Logistic Regression model that estimates the probability of late delivery.
- **Interactive Dashboard:** KPI summaries, risk distributions, and order-level analysis using Streamlit and Plotly.
- **Adjustable Risk Threshold:** Explore how different thresholds affect the number of flagged orders.
- **Priority Queue:** Review orders ranked by predicted late-delivery risk.
- **CSV Export:** Download the scored order queue for further analysis.
- **LLM-Powered Recommendations:** Generate order-specific recommendations using a locally hosted Qwen model through Ollama.

## Machine Learning Methodology

### Dataset

This project uses the [DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS dataset](https://data.mendeley.com/datasets/8gx2fvg2k6/5).

The original dataset contains 180,519 order-item records. For this project, the records were aggregated into **65,752 individual orders** to create the modeling dataset.

The prediction target is `Late_delivery_risk`, a binary variable indicating whether an order is associated with late-delivery risk in the historical data.

### Data Preparation

The preprocessing pipeline includes:

- Aggregating order-item records into individual orders.
- Engineering order-level features, including item counts, quantities, sales, and discounts.
- Extracting temporal features from order dates.
- Preparing numerical and categorical variables for modeling.
- Excluding potential target leakage variables, including `Delivery Status`, `Days for shipping (real)`, and `Order Status`.

### Model Development

Logistic Regression and Random Forest models were evaluated during experimentation.

A compact Logistic Regression model was selected using five features:

| Feature | Type |
|---|---|
| `scheduled_shipping_days` | Numerical |
| `order_month` | Numerical |
| `shipping_mode` | Categorical |
| `order_region` | Categorical |
| `order_type` | Categorical |

The compact model was selected because it achieved slightly better temporal holdout performance than the original 17-feature model while using fewer features.

### Temporal Validation

To evaluate performance on later historical orders, the dataset was sorted chronologically and divided into:

- **Training set:** First 80% of orders (52,601).
- **Test set:** Most recent 20% of orders (13,151).

The test period covers July 2017 through January 2018.

The compact model achieved the following results:

| Metric | Result |
|---|---:|
| Average Precision (AP) | 0.8156 |
| ROC-AUC | 0.7446 |
| Test set size | 13,151 orders |

These results describe performance on this historical holdout dataset. They should not be interpreted as guaranteed performance on future shipments or live supply chain operations.

### Feature Analysis and Limitations

Feature ablation experiments showed that shipping-related variables contribute most of the model's predictive performance.

When `shipping_mode` and `scheduled_shipping_days` were removed, Average Precision decreased from 0.8156 to approximately 0.5729.

This indicates that the model relies heavily on historical associations involving shipping characteristics. These features must be available and reliable at the time a prediction is made.

The application is a historical-data prototype, not a production-validated delivery forecasting system.

## Generative AI Integration

The project includes an optional LLM component using:

- **Model:** Qwen 3.5 4B
- **Runtime:** Ollama
- **Integration:** Ollama Python client

For a selected order, the application provides the LLM with the predicted risk score and available order characteristics. The model generates:

1. A risk interpretation.
2. Three recommended actions for review.
3. Operational information that should be verified.

The system prompt instructs the LLM not to invent carrier delays, route congestion, inventory shortages, or other unavailable operational information.

**Deployment note:** The public Streamlit application runs without the local LLM service. AI-generated recommendations are available when running the application locally with Ollama configured.

LLM outputs are suggestions for decision support, not verified operational instructions.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Plotly |
| Application | Streamlit |
| Generative AI | Qwen 3.5 4B, Ollama |
| Model Persistence | Joblib |
| Deployment | Streamlit Community Cloud |
| Version Control | Git, GitHub |

## Project Structure

```text
supplychain-action-copilot/
│
├── app.py
├── requirements.txt
├── .python-version
├── README.md
│
├── data/
│   └── orders_modeling.csv
│
├── models/
│   ├── late_risk_model_compact.joblib
│   └── compact_feature_columns.joblib
│
└── src/
    ├── prepare_data.py
    ├── inspect_data.py
    ├── inspect_features.py
    ├── train_model.py
    ├── train_compact_model.py
    ├── validate_temporal.py
    ├── evaluate_thresholds.py
    ├── feature_ablation.py
    ├── audit_model.py
    └── test_llm.py
```

## Getting Started

### Prerequisites

- Python 3.13
- Git

### Installation

Clone the repository:

```bash
git clone https://github.com/amlc16/supplychain-action-copilot.git

cd supplychain-action-copilot
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app.py
```

The application will open in your browser.

### Enable Local AI Recommendations

To use the optional LLM functionality locally:

1. Install [Ollama](https://ollama.com/).
2. Download the model:

```bash
ollama pull qwen3.5:4b
```

3. Make sure the Ollama service is running.
4. Start the application:

```bash
streamlit run app.py
```

Local AI recommendations are enabled by default. To disable them, set:

```bash
export ENABLE_LOCAL_LLM=false
```

On Windows PowerShell:

```powershell
$env:ENABLE_LOCAL_LLM="false"
```

## Deployment

The public application is hosted on [Streamlit Community Cloud](https://streamlit.io/cloud).

[Launch SupplyChain Action Copilot](https://supplychain-action-copilot.streamlit.app/)

The deployed dashboard provides machine learning predictions and interactive analytics. The locally hosted Ollama component is disabled in the cloud deployment.

## Future Improvements

- Integrate SHAP-based explanations to identify feature contributions to individual predictions.
- Incorporate live shipment tracking and carrier performance data.
- Evaluate additional models and calibration techniques.
- Extend recommendations using operational data and verified business constraints.
- Evaluate model performance on newer, independently collected shipment data.

## Dataset Attribution

Constante, Fabian; Silva, Fernando; Pereira, António (2019). *DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS*. Mendeley Data, Version 5.

https://doi.org/10.17632/8gx2fvg2k6.5

The original dataset is distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. Please refer to the original dataset record for licensing terms.

## Disclaimer

This project is an educational and portfolio prototype built using historical supply chain data. Predictions and recommendations are not guaranteed to reflect actual shipment conditions or future delivery outcomes.