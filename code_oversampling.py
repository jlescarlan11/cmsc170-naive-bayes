"""
Manual Naive Bayes Classifier — Oversampled Dataset
Features: QuantityCat, UnitPriceCat, InvoiceMonth, InvoiceHour
Target:   IsCancelled (0 = Not Cancelled, 1 = Cancelled)
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# ============================================================
# CONFIG
# ============================================================
INPUT_PATH = 'Oversampling_Online_Retail.csv'
OUTPUT_DIR = '.'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'naive_bayes_predictions_oversampling.csv')


# ============================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)
print(f"Total rows: {len(df)}")

features = ['QuantityCat', 'UnitPriceCat', 'InvoiceMonth', 'InvoiceHour']
target = 'IsCancelled'

df = df[features + [target]].dropna().reset_index(drop=True)
df['InvoiceMonth'] = df['InvoiceMonth'].astype(str)

print(f"Rows after cleaning: {len(df)}")
print(f"\nClass distribution:")
print(df[target].value_counts())
print(f"Cancellation rate: {df[target].mean() * 100:.2f}%")


# ============================================================
# STEP 2: TRAIN/TEST SPLIT (80/20, stratified)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Train/Test split")
print("=" * 60)

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[target]
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Train size: {len(train_df)}")
print(f"Test size:  {len(test_df)}")


# ============================================================
# STEP 3: BUILD FREQUENCY / PROBABILITY TABLES
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Building probability tables from training data")
print("=" * 60)

total_train = len(train_df)
count_cancelled = (train_df[target] == 1).sum()
count_not_cancelled = (train_df[target] == 0).sum()

P_cancelled = count_cancelled / total_train
P_not_cancelled = count_not_cancelled / total_train

print("\n--- PRIOR PROBABILITIES ---")
print(f"{'IsCancelled':<15} {'Count':<10} {'Probability':<15}")
print("-" * 40)
print(f"{'No (0)':<15} {count_not_cancelled:<10} {P_not_cancelled:.4f} ({count_not_cancelled}/{total_train})")
print(f"{'Yes (1)':<15} {count_cancelled:<10} {P_cancelled:.4f} ({count_cancelled}/{total_train})")
print(f"{'Total':<15} {total_train:<10}")

prob_tables = {}

for feature in features:
    prob_tables[feature] = {}
    unique_values = sorted(train_df[feature].unique().tolist(), key=lambda x: str(x))
    num_unique = len(unique_values)

    print(f"\n--- {feature.upper()} ---")
    print(f"{'Value':<25} {'Count(Yes)':<12} {'Count(No)':<12} {'P(value|Yes)':<15} {'P(value|No)':<15}")
    print("-" * 80)

    for value in unique_values:
        count_v_yes = ((train_df[feature] == value) & (train_df[target] == 1)).sum()
        count_v_no  = ((train_df[feature] == value) & (train_df[target] == 0)).sum()

        # Laplace smoothing
        p_yes = (count_v_yes + 1) / (count_cancelled + num_unique)
        p_no  = (count_v_no  + 1) / (count_not_cancelled + num_unique)

        prob_tables[feature][value] = {1: p_yes, 0: p_no}

        print(f"{str(value):<25} {count_v_yes:<12} {count_v_no:<12} {p_yes:<15.6f} {p_no:<15.6f}")

    print(f"{'Total':<25} {count_cancelled:<12} {count_not_cancelled:<12}")


# ============================================================
# STEP 4: PREDICTION FUNCTION
# ============================================================

def predict_row(row, prob_tables, P_cancelled, P_not_cancelled, features, verbose=False):
    score_yes = P_cancelled
    score_no  = P_not_cancelled

    if verbose:
        print(f"\nStarting with priors:")
        print(f"  P(Yes) = {P_cancelled:.6f}")
        print(f"  P(No)  = {P_not_cancelled:.6f}")

    for feature in features:
        value = row[feature]

        if value in prob_tables[feature]:
            p_yes = prob_tables[feature][value][1]
            p_no  = prob_tables[feature][value][0]
            score_yes *= p_yes
            score_no  *= p_no

            if verbose:
                print(f"\n  {feature} = '{value}':")
                print(f"    P({value}|Yes) = {p_yes:.6f}")
                print(f"    P({value}|No)  = {p_no:.6f}")
                print(f"    Running score(Yes) = {score_yes:.10f}")
                print(f"    Running score(No)  = {score_no:.10f}")
        else:
            num_unique = len(prob_tables[feature])
            score_yes *= 1 / (count_cancelled + num_unique)
            score_no  *= 1 / (count_not_cancelled + num_unique)

    total = score_yes + score_no
    if total > 0:
        prob_yes = score_yes / total
        prob_no  = score_no  / total
    else:
        prob_yes = prob_no = 0.5

    if verbose:
        print(f"\n  Before normalization:")
        print(f"    score(Yes) = {score_yes:.10f}")
        print(f"    score(No)  = {score_no:.10f}")
        print(f"\n  After normalization:")
        print(f"    P(Yes|features) = {prob_yes:.6f}")
        print(f"    P(No|features)  = {prob_no:.6f}")

    prediction = 1 if prob_yes > prob_no else 0

    if verbose:
        winner = "Yes (Cancelled)" if prediction == 1 else "No (Not Cancelled)"
        print(f"\n  Prediction: {winner}")

    return prediction, prob_yes, prob_no


# ============================================================
# STEP 5: WORKED EXAMPLE
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Worked example (full calculation for one row)")
print("=" * 60)

sample_row = test_df.iloc[0]
print(f"\nSample input:")
for f in features:
    print(f"  {f:<20} = {sample_row[f]}")
print(f"  {'Actual label':<20} = {sample_row[target]} ({'Cancelled' if sample_row[target] == 1 else 'Not Cancelled'})")

predict_row(sample_row, prob_tables, P_cancelled, P_not_cancelled, features, verbose=True)


# ============================================================
# STEP 6: PREDICT ON FULL TEST SET
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Predicting on full test set")
print("=" * 60)

predictions, prob_yes_list, prob_no_list = [], [], []

for _, row in test_df.iterrows():
    pred, p_yes, p_no = predict_row(row, prob_tables, P_cancelled, P_not_cancelled, features)
    predictions.append(pred)
    prob_yes_list.append(p_yes)
    prob_no_list.append(p_no)

test_df['P_Cancelled']    = prob_yes_list
test_df['P_NotCancelled'] = prob_no_list
test_df['Predicted']      = predictions

print(f"Predictions complete. Sample (first 10 rows):")
print(test_df[features + ['P_Cancelled', 'P_NotCancelled', 'Predicted', target]].head(10))


# ============================================================
# STEP 7: EVALUATION METRICS
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Evaluation Metrics")
print("=" * 60)

y_true = test_df[target].values
y_pred = test_df['Predicted'].values

print(f"\nAccuracy:               {accuracy_score(y_true, y_pred) * 100:.2f}%")
print(f"F1 Score (Cancelled):   {f1_score(y_true, y_pred):.4f}")
print(f"F1 Score (Macro avg):   {f1_score(y_true, y_pred, average='macro'):.4f}")
print(f"F1 Score (Weighted avg):{f1_score(y_true, y_pred, average='weighted'):.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=['Not Cancelled', 'Cancelled']))

print("--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)
print(f"                          Predicted Not Cancelled  Predicted Cancelled")
print(f"Actual Not Cancelled              {cm[0][0]:>10}              {cm[0][1]:>10}")
print(f"Actual Cancelled                  {cm[1][0]:>10}              {cm[1][1]:>10}")

print("\n--- Manual F1 Calculation ---")
TP = cm[1][1]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[0][0]

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"True Positives (TP):  {TP}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"True Negatives (TN):  {TN}")
print(f"\nPrecision = {TP} / ({TP} + {FP}) = {precision:.4f}")
print(f"Recall    = {TP} / ({TP} + {FN}) = {recall:.4f}")
print(f"F1        = 2 * ({precision:.4f} * {recall:.4f}) / ({precision:.4f} + {recall:.4f}) = {f1:.4f}")


# ============================================================
# STEP 8: SAVE PREDICTIONS
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
test_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n\nPredictions saved to: {OUTPUT_PATH}")
