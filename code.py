"""
Manual Naive Bayes Classifier for Order Cancellation Prediction
Follows the discrete/categorical approach from CMSC 170 course pack

Features used: QuantityCat, UnitPriceCat, InvoiceMonth, InvoiceHour, Country
Target: IsCancelled (0 = Not Cancelled, 1 = Cancelled)
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score

# Show all columns in output (no truncation)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# ============================================================
# CONFIG: File paths (UPDATE THESE FOR YOUR MACHINE)
# ============================================================
INPUT_PATH = 'online_retail.csv'
OUTPUT_DIR = '.'
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'naive_bayes_predictions.csv')


# ============================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================
print("="*60)
print("STEP 1: Loading data")
print("="*60)

df = pd.read_csv(INPUT_PATH)
print(f"Total rows: {len(df)}")

features = ['QuantityCat', 'UnitPriceCat', 'InvoiceMonth', 'InvoiceHour', 'Country']
target = 'IsCancelled'

# Drop rows with missing values in our features
df = df[features + [target]].dropna().reset_index(drop=True)
# Make sure InvoiceMonth is treated as category (not numeric)
df['InvoiceMonth'] = df['InvoiceMonth'].astype(str)

print(f"Rows after cleaning: {len(df)}")
print(f"\nClass distribution:")
print(df[target].value_counts())
print(f"Cancellation rate: {df[target].mean()*100:.2f}%")


# ============================================================
# STEP 2: TRAIN/TEST SPLIT (80/20, stratified)
# ============================================================
print("\n" + "="*60)
print("STEP 2: Train/Test split")
print("="*60)

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df[target]
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")


# ============================================================
# STEP 3: BUILD FREQUENCY / PROBABILITY TABLES
# (This is the golf example approach from page 29)
# ============================================================
print("\n" + "="*60)
print("STEP 3: Building probability tables from training data")
print("="*60)

# Prior probabilities P(yes) and P(no)
total_train = len(train_df)
count_cancelled = (train_df[target] == 1).sum()
count_not_cancelled = (train_df[target] == 0).sum()

P_cancelled = count_cancelled / total_train
P_not_cancelled = count_not_cancelled / total_train

# Print prior probability table (like the "Play" table on page 29)
print("\n--- PRIOR PROBABILITIES (Class Distribution) ---")
print(f"{'IsCancelled':<15} {'Count':<10} {'Probability':<15}")
print("-" * 40)
print(f"{'No (0)':<15} {count_not_cancelled:<10} {P_not_cancelled:.4f} ({count_not_cancelled}/{total_train})")
print(f"{'Yes (1)':<15} {count_cancelled:<10} {P_cancelled:.4f} ({count_cancelled}/{total_train})")
print(f"{'Total':<15} {total_train:<10}")


# Build conditional probability tables for each feature
prob_tables = {}

for feature in features:
    prob_tables[feature] = {}
    unique_values = sorted(train_df[feature].unique().tolist(), key=lambda x: str(x))
    num_unique = len(unique_values)
    
    # Print frequency + probability table for this feature (like the golf example, page 29)
    print(f"\n--- {feature.upper()} ---")
    print(f"{'Value':<25} {'Count(Yes)':<12} {'Count(No)':<12} {'P(value|Yes)':<15} {'P(value|No)':<15}")
    print("-" * 80)
    
    for value in unique_values:
        # Count occurrences for each class
        count_value_cancelled = ((train_df[feature] == value) & (train_df[target] == 1)).sum()
        count_value_not_cancelled = ((train_df[feature] == value) & (train_df[target] == 0)).sum()
        
        # Laplace smoothing: add 1 to numerator, add num_unique to denominator
        # This prevents probability = 0 for values not seen with a class
        P_value_given_cancelled = (count_value_cancelled + 1) / (count_cancelled + num_unique)
        P_value_given_not_cancelled = (count_value_not_cancelled + 1) / (count_not_cancelled + num_unique)
        
        prob_tables[feature][value] = {
            1: P_value_given_cancelled,      # P(value | Cancelled)
            0: P_value_given_not_cancelled   # P(value | Not Cancelled)
        }
        
        # Print row in the table
        print(f"{str(value):<25} {count_value_cancelled:<12} {count_value_not_cancelled:<12} {P_value_given_cancelled:<15.6f} {P_value_given_not_cancelled:<15.6f}")
    
    print(f"{'Total':<25} {count_cancelled:<12} {count_not_cancelled:<12}")


# ============================================================
# STEP 4: PREDICTION FUNCTION
# (Apply the formula from page 28 of course pack)
# P(y|x1,x2,...,xn) ∝ P(y) * P(x1|y) * P(x2|y) * ... * P(xn|y)
# ============================================================

def predict_row(row, prob_tables, P_cancelled, P_not_cancelled, features, verbose=False):
    """
    Predict whether an order will be cancelled using Naive Bayes.
    Returns: (prediction, P_cancelled_given_x, P_not_cancelled_given_x)
    
    If verbose=True, prints out the calculation step-by-step (like page 30 of course pack)
    """
    # Start with prior probabilities
    score_cancelled = P_cancelled
    score_not_cancelled = P_not_cancelled
    
    if verbose:
        print(f"\nStarting with priors:")
        print(f"  P(Yes) = {P_cancelled:.6f}")
        print(f"  P(No)  = {P_not_cancelled:.6f}")
    
    # Multiply by likelihoods for each feature
    for feature in features:
        value = row[feature]
        
        if value in prob_tables[feature]:
            p_yes = prob_tables[feature][value][1]
            p_no = prob_tables[feature][value][0]
            score_cancelled *= p_yes
            score_not_cancelled *= p_no
            
            if verbose:
                print(f"\n  {feature} = '{value}':")
                print(f"    P({value}|Yes) = {p_yes:.6f}")
                print(f"    P({value}|No)  = {p_no:.6f}")
                print(f"    Running score(Yes) = {score_cancelled:.10f}")
                print(f"    Running score(No)  = {score_not_cancelled:.10f}")
        else:
            # Unseen value (e.g., country not in training set)
            num_unique = len(prob_tables[feature])
            score_cancelled *= 1 / (count_cancelled + num_unique)
            score_not_cancelled *= 1 / (count_not_cancelled + num_unique)
    
    # Normalize (like the golf example: divide by sum so probabilities add to 1)
    total = score_cancelled + score_not_cancelled
    if total > 0:
        prob_cancelled = score_cancelled / total
        prob_not_cancelled = score_not_cancelled / total
    else:
        prob_cancelled = 0.5
        prob_not_cancelled = 0.5
    
    if verbose:
        print(f"\n  Before normalization:")
        print(f"    score(Yes) = {score_cancelled:.10f}")
        print(f"    score(No)  = {score_not_cancelled:.10f}")
        print(f"\n  After normalization (divide by sum, so they add to 1):")
        print(f"    P(Yes|features) = {score_cancelled:.10f} / ({score_cancelled:.10f} + {score_not_cancelled:.10f}) = {prob_cancelled:.6f}")
        print(f"    P(No|features)  = {score_not_cancelled:.10f} / ({score_cancelled:.10f} + {score_not_cancelled:.10f}) = {prob_not_cancelled:.6f}")
    
    # Predict the class with higher probability
    prediction = 1 if prob_cancelled > prob_not_cancelled else 0
    
    if verbose:
        winner = "Yes (Cancelled)" if prediction == 1 else "No (Not Cancelled)"
        print(f"\n  Since P({'Yes' if prediction == 1 else 'No'}) > P({'No' if prediction == 1 else 'Yes'}),")
        print(f"  Therefore IsCancelled = '{winner}'")
    
    return prediction, prob_cancelled, prob_not_cancelled


# ============================================================
# STEP 5: WORKED EXAMPLE (like page 30 of course pack)
# Show the full calculation for one sample row
# ============================================================
print("\n" + "="*60)
print("STEP 5: Worked example (full calculation for one row)")
print("="*60)

# Pick the first row of test set as our worked example
sample_row = test_df.iloc[0]
print(f"\nSample input (Today equivalent):")
print(f"  QuantityCat   = {sample_row['QuantityCat']}")
print(f"  UnitPriceCat  = {sample_row['UnitPriceCat']}")
print(f"  InvoiceMonth  = {sample_row['InvoiceMonth']}")
print(f"  InvoiceHour   = {sample_row['InvoiceHour']}")
print(f"  Country       = {sample_row['Country']}")
print(f"  Actual label  = {sample_row['IsCancelled']} ({'Cancelled' if sample_row['IsCancelled'] == 1 else 'Not Cancelled'})")

# Predict with verbose=True to show all the math
pred, p_yes, p_no = predict_row(
    sample_row, prob_tables, P_cancelled, P_not_cancelled, features, verbose=True
)


# ============================================================
# STEP 6: PREDICT ON FULL TEST SET
# ============================================================
print("\n" + "="*60)
print("STEP 6: Predicting on full test set")
print("="*60)

predictions = []
prob_cancelled_list = []
prob_not_cancelled_list = []

for idx, row in test_df.iterrows():
    pred, p_yes, p_no = predict_row(row, prob_tables, P_cancelled, P_not_cancelled, features, verbose=False)
    predictions.append(pred)
    prob_cancelled_list.append(p_yes)
    prob_not_cancelled_list.append(p_no)

# Add predictions to test set for review
test_df['P_Cancelled'] = prob_cancelled_list
test_df['P_NotCancelled'] = prob_not_cancelled_list
test_df['Predicted'] = predictions

print(f"Predictions complete. Sample (first 10 rows):")
print(test_df[['QuantityCat', 'UnitPriceCat', 'InvoiceMonth', 'InvoiceHour', 'Country',
               'P_Cancelled', 'P_NotCancelled', 'Predicted', 'IsCancelled']].head(10))


# ============================================================
# STEP 7: COMPUTE F1 SCORE AND OTHER METRICS
# ============================================================
print("\n" + "="*60)
print("STEP 7: Evaluation Metrics")
print("="*60)

y_true = test_df['IsCancelled'].values
y_pred = test_df['Predicted'].values

print(f"\nAccuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
print(f"\nF1 Score (Cancelled class): {f1_score(y_true, y_pred):.4f}")
print(f"F1 Score (Macro avg): {f1_score(y_true, y_pred, average='macro'):.4f}")
print(f"F1 Score (Weighted avg): {f1_score(y_true, y_pred, average='weighted'):.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=['Not Cancelled', 'Cancelled']))

print("--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)
print(f"                          Predicted Not Cancelled  Predicted Cancelled")
print(f"Actual Not Cancelled              {cm[0][0]:>10}              {cm[0][1]:>10}")
print(f"Actual Cancelled                  {cm[1][0]:>10}              {cm[1][1]:>10}")

print("\n--- Manual F1 Calculation (for verification) ---")
TP = cm[1][1]  # True Positive: predicted cancelled, actual cancelled
FP = cm[0][1]  # False Positive: predicted cancelled, actual not cancelled
FN = cm[1][0]  # False Negative: predicted not cancelled, actual cancelled
TN = cm[0][0]  # True Negative: predicted not cancelled, actual not cancelled

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"True Positives (TP):  {TP}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")
print(f"True Negatives (TN):  {TN}")
print(f"\nPrecision = TP / (TP + FP) = {TP} / ({TP} + {FP}) = {precision:.4f}")
print(f"Recall    = TP / (TP + FN) = {TP} / ({TP} + {FN}) = {recall:.4f}")
print(f"F1        = 2 * (P * R) / (P + R) = {f1:.4f}")


# ============================================================
# STEP 8: SAVE PREDICTIONS TO CSV
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
test_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n\nPredictions saved to: {OUTPUT_PATH}")