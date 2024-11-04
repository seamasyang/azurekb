import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score

# - ŷ=0 and y =0, True negatives (TN)
# - ŷ=1 and y =0, False positives (FP)
# - ŷ=0 and y =1, False negative (FN)
# - ŷ=1 and y =1, True positive (TP)

# accuracy = (TN + TP) / (TN + TP + FN + TP); proportion of predications that model got right
# recall = TP / (TP + FN); proportion of positive case that model identified correctly
# precision = TP / (TP + FP); proportion of predicted positive cases where the true label is actually positive
# f1 = (2 *Precision* Recall) / (Precision + Recall)

data = [
    {"query": "aaa", "expected_answer": "a10", "predicated_answer": "a10"},  # tp
    {"query": "Aaa", "expected_answer": "a20", "predicated_answer": "a20"},  # tp
    {"query": "aAa", "expected_answer": "a30", "predicated_answer": "a30"},  # tp
    {"query": "aAa", "expected_answer": "a30", "predicated_answer": "a30"},  # tp
    {"query": "aAa", "expected_answer": "a30", "predicated_answer": "a30"},  # tp
    
    {"query": "bbb", "expected_answer": "b10", "predicated_answer": ""},      # fn

    {"query": "ddd", "expected_answer": "", "predicated_answer": ""},         # tn
    {"query": "Ddd", "expected_answer": "", "predicated_answer": ""},         # tn

    {"query": "eee", "expected_answer": "", "predicated_answer": "e10"},       # fp    
    {"query": "Eee", "expected_answer": "", "predicated_answer": "e10"}       # fp


]

# Initialize counters
tp = 0  # True Positives
fn = 0  # False Negatives
tn = 0  # True Negatives
fp = 0  # False Positives

# Classify each entry in the data based on expected and predicted answers
for entry in data:
    expected = entry["expected_answer"]
    predicted = entry["predicated_answer"]
    
    if expected and predicted == expected:
        tp += 1  # True Positive
    elif expected and not predicted:
        fn += 1  # False Negative
    elif not expected and not predicted:
        tn += 1  # True Negative
    elif not expected and predicted:
        fp += 1  # False Positive

# Calculate metrics
accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
recall = tp / (tp + fn) if (tp + fn) else 0
precision = tp / (tp + fp) if (tp + fp) else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

print(f"Accuracy: {accuracy:.2f}")
print(f"Recall: {recall:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1_score:.2f}")
