#!/usr/bin/env python3
"""
Check accuracy of predictions against test data labels
"""
import numpy as np
import pandas as pd
from pred import predict_all

if __name__ == "__main__":
    print("Running predictions on test_data.csv...")
    predictions = predict_all("test_data.csv")

    # Load test data to get actual labels
    df_test = pd.read_csv("test_data.csv")

    # Check if labels exist
    if 'label' not in df_test.columns:
        print("\nError: 'label' column not found in test_data.csv")
        print("Available columns:", df_test.columns.tolist())
    else:
        actual_labels = df_test['label'].values
        predictions = np.array(predictions)

        # Calculate overall accuracy
        accuracy = np.mean(predictions == actual_labels)
        print(f"\n{'='*50}")
        print(f"Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"{'='*50}")

        # Per-class accuracy
        print("\nPer-Class Accuracy:")
        for label in np.unique(actual_labels):
            mask = actual_labels == label
            class_acc = np.mean(predictions[mask] == actual_labels[mask])
            total = np.sum(mask)
            correct = np.sum(predictions[mask] == actual_labels[mask])
            print(f"  {label}: {class_acc*100:.1f}% ({correct}/{total} correct)")

        # Confusion matrix
        print("\nConfusion Matrix:")
        labels = np.unique(actual_labels)
        print(f"{'':>12}", end="")
        for l in labels:
            print(f"{l:>12}", end="")
        print()

        for true_label in labels:
            print(f"{true_label:>12}", end="")
            for pred_label in labels:
                count = np.sum((actual_labels == true_label) & (predictions == pred_label))
                print(f"{count:>12}", end="")
            print()

        # Distribution comparison
        print("\n" + "="*50)
        print("Distribution Comparison:")
        print("="*50)
        from collections import Counter
        actual_counts = Counter(actual_labels)
        pred_counts = Counter(predictions)

        print(f"{'Label':>12} {'Actual':>12} {'Predicted':>12}")
        print("-"*50)
        for label in labels:
            print(f"{label:>12} {actual_counts[label]:>12} {pred_counts[label]:>12}")
