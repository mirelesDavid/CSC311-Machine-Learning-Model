import numpy as np
import pandas as pd
from collections import Counter
from pred import predict_all


def printConfusionMatrix(actualLabels, predictions, uniqueLabels):
    print("\nConfusion Matrix:")
    print(f"{'':>12}", end="")
    for label in uniqueLabels:
        print(f"{label:>12}", end="")
    print()

    for trueLabel in uniqueLabels:
        print(f"{trueLabel:>12}", end="")
        for predictedLabel in uniqueLabels:
            count = np.sum((actualLabels == trueLabel) & (predictions == predictedLabel))
            print(f"{count:>12}", end="")
        print()


def printDistributionComparison(actualLabels, predictions, uniqueLabels):
    actualCounts = Counter(actualLabels)
    predictedCounts = Counter(predictions)

    print(f"{'Label':>12} {'Actual':>12} {'Predicted':>12}")
    print("-"*50)
    for label in uniqueLabels:
        print(f"{label:>12} {actualCounts[label]:>12} {predictedCounts[label]:>12}")


def main():
    predictions = predict_all("test_data.csv")

    testData = pd.read_csv("test_data.csv")

    if 'label' not in testData.columns:
        print("\nError: 'label' column not found in test_data.csv")
        print("Available columns:", testData.columns.tolist())
        return

    actualLabels = testData['label'].values
    predictions = np.array(predictions)

    overallAccuracy = np.mean(predictions == actualLabels)
    print(f"\n{'='*50}")
    print(f"Overall Test Accuracy: {overallAccuracy:.4f} ({overallAccuracy*100:.2f}%)")
    print(f"{'='*50}")

    uniqueLabels = np.unique(actualLabels)

    print("\nPer-Class Accuracy:")
    for label in uniqueLabels:
        classMask = actualLabels == label
        classAccuracy = np.mean(predictions[classMask] == actualLabels[classMask])
        totalSamples = np.sum(classMask)
        correctPredictions = np.sum(predictions[classMask] == actualLabels[classMask])
        print(f"  {label}: {classAccuracy*100:.1f}% ({correctPredictions}/{totalSamples} correct)")

    printConfusionMatrix(actualLabels, predictions, uniqueLabels)
    printDistributionComparison(actualLabels, predictions, uniqueLabels)


if __name__ == "__main__":
    main()
