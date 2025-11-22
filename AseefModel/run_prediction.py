#!/usr/bin/env python3
"""
Simple runner script for pred.py
"""
from pred import predict_all

if __name__ == "__main__":
    print("Running predictions on test_data.csv...")
    predictions = predict_all("test_data.csv")

    print(f"\nTotal predictions: {len(predictions)}")
    print(f"\nFirst 10 predictions:")
    for i, pred in enumerate(predictions[:10], 1):
        print(f"  {i}. {pred}")

    # Count predictions by class
    from collections import Counter
    counts = Counter(predictions)
    print(f"\nPrediction distribution:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count} ({count/len(predictions)*100:.1f}%)")

