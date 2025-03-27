import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score

TP = 40
TN = 30
FP = 20
FN = 10


y_true = np.array([1] * TP + [0] * TN + [1] * FP + [0] * FN)
y_pred = np.array([1] * TP + [1] * FP + [0] * TN + [0] * FN)

cm = confusion_matrix(y_true, y_pred)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
f1_measure = f1_score(y_true, y_pred)

print(f"Matriz de confusión:\n{cm}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1-Measure: {f1_measure:.2f}")