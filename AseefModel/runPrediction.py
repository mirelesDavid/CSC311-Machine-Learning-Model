from collections import Counter
from pred import predict_all


def printPredictionSummary(predictions, numSamplesToShow=10):
    print(f"\nFirst {numSamplesToShow} predictions:")
    for index, prediction in enumerate(predictions[:numSamplesToShow], 1):
        print(f"  {index}. {prediction}")

    predictionCounts = Counter(predictions)
    totalPredictions = len(predictions)

    print(f"\nPrediction distribution:")
    for label, count in sorted(predictionCounts.items()):
        percentage = (count / totalPredictions) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")


def main():
    print("Running predictions on test_data.csv...")
    predictions = predict_all("test_data.csv")
    printPredictionSummary(predictions)


if __name__ == "__main__":
    main()
