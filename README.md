# CMSC 170 — Manual Naive Bayes Classifier for Order Cancellation Prediction

A from-scratch Naive Bayes classifier that predicts whether an online retail order will be cancelled. Built following the discrete/categorical approach from the CMSC 170 course pack.

## Features Used

| Feature | Description |
|---|---|
| `QuantityCat` | Categorized order quantity |
| `UnitPriceCat` | Categorized unit price |
| `InvoiceMonth` | Month of the invoice |
| `InvoiceHour` | Hour of the invoice |
| `Country` | Country of the customer |

**Target:** `IsCancelled` — `0` = Not Cancelled, `1` = Cancelled

## Prerequisites

- Python 3.10 or higher (developed on Python 3.14.4)

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/jlescarlan11/cmsc170-naive-bayes.git
cd cmsc170-naive-bayes
```

**2. Create and activate a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

## Running the Code

### Original dataset (`code.py`)

```bash
python code.py
```

The script will:

1. Load and clean the dataset
2. Split data 80/20 (stratified) into train and test sets
3. Build frequency and conditional probability tables (with Laplace smoothing)
4. Show a step-by-step worked example for one test row
5. Run predictions on the full test set
6. Print accuracy, F1 score, classification report, and confusion matrix
7. Save predictions to `naive_bayes_predictions.csv`

### Oversampled balanced dataset (`code_oversampling.py`)

```bash
python code_oversampling.py
```

Uses the same Naive Bayes approach on a class-balanced dataset (`Oversampling_Online_Retail.csv`, 50% cancelled / 50% not cancelled). Features are the same except `Country` is excluded.

The script will:

1. Load and clean the oversampled dataset
2. Split data 80/20 (stratified) into train and test sets
3. Build frequency and conditional probability tables (with Laplace smoothing)
4. Show a step-by-step worked example for one test row
5. Run predictions on the full test set
6. Print accuracy, F1 score, classification report, and confusion matrix
7. Save predictions to `naive_bayes_predictions_oversampling.csv`

## Output

| File | Description |
|---|---|
| `naive_bayes_predictions.csv` | Predictions from `code.py` — full test set with `P_Cancelled`, `P_NotCancelled`, `Predicted`, `IsCancelled` |
| `naive_bayes_predictions_oversampling.csv` | Predictions from `code_oversampling.py` — same columns, oversampled dataset |

## Project Structure

```
.
├── code.py                                   # Classifier — original dataset
├── code_oversampling.py                      # Classifier — oversampled balanced dataset
├── online_retail.csv                         # Original input dataset
├── Oversampling_Online_Retail.csv            # Oversampled balanced input dataset
├── naive_bayes_predictions.csv               # Output predictions from code.py (generated on run)
├── naive_bayes_predictions_oversampling.csv  # Output predictions from code_oversampling.py (generated on run)
├── requirements.txt                          # Python dependencies
└── README.md
```
