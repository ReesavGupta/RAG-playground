from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import pandas as pd

# Example usage function for evaluation

def evaluate(true_labels, pred_labels):
    acc = accuracy_score(true_labels, pred_labels)
    auc = roc_auc_score(true_labels, pred_labels)
    cm = confusion_matrix(true_labels, pred_labels)
    return acc, auc, cm

if __name__ == "__main__":
    # Load predictions and true labels from model_comparison.csv
    df = pd.read_csv('evaluate/model_comparison.csv')
    print("--- Fine-tuned Model ---")
    acc, auc, cm = evaluate(df['true_label'], df['fine_tuned_pred'])
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("\n--- Baseline Model ---")
    acc, auc, cm = evaluate(df['true_label'], df['baseline_pred'])
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")

