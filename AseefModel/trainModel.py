import pandas as pd
import numpy as np
from pred import extract_features, HierarchicalModel, Node

np.random.seed(42)


class GiniDecisionTree:
    def __init__(self, maxDepth=8, minSamplesSplit=5, maxFeatures='sqrt', classWeight=None):
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.maxFeatures = maxFeatures
        self.classWeight = classWeight
        self.tree = None

    def _calculateGini(self, labels, weights):
        if len(labels) == 0:
            return 0

        classes = np.unique(labels)
        totalWeight = np.sum(weights)

        if totalWeight == 0:
            return 0

        gini = 1.0
        for classLabel in classes:
            probability = np.sum(weights[labels == classLabel]) / totalWeight
            gini -= probability ** 2

        return gini

    def _buildTree(self, features, labels, weights, currentDepth):
        numSamples, numFeatures = features.shape

        shouldStopSplitting = (
            currentDepth >= self.maxDepth or
            numSamples < self.minSamplesSplit or
            len(np.unique(labels)) <= 1
        )

        if shouldStopSplitting:
            leafValue = {}
            totalWeight = np.sum(weights)
            for classLabel in np.unique(labels):
                leafValue[classLabel] = np.sum(weights[labels == classLabel]) / totalWeight if totalWeight > 0 else 0
            return Node(value=leafValue)

        numFeaturesToTry = self._getNumFeaturesToTry(numFeatures)
        selectedFeatures = np.random.choice(numFeatures, numFeaturesToTry, replace=False)

        bestGain = -1
        bestFeature = None
        bestThreshold = None
        currentGini = self._calculateGini(labels, weights)
        totalWeight = np.sum(weights)

        for featureIndex in selectedFeatures:
            thresholds = np.unique(features[:, featureIndex])
            if len(thresholds) > 15:
                thresholds = np.percentile(thresholds, np.linspace(10, 90, 10))

            for threshold in thresholds:
                leftMask = features[:, featureIndex] <= threshold
                rightMask = ~leftMask

                if np.sum(leftMask) == 0 or np.sum(rightMask) == 0:
                    continue

                leftWeights = weights[leftMask]
                rightWeights = weights[rightMask]
                leftWeightSum = np.sum(leftWeights)
                rightWeightSum = np.sum(rightWeights)

                leftGini = self._calculateGini(labels[leftMask], leftWeights)
                rightGini = self._calculateGini(labels[rightMask], rightWeights)

                informationGain = currentGini - (
                    (leftWeightSum / totalWeight) * leftGini +
                    (rightWeightSum / totalWeight) * rightGini
                )

                if informationGain > bestGain:
                    bestGain = informationGain
                    bestFeature = featureIndex
                    bestThreshold = threshold

        if bestGain <= 1e-6:
            leafValue = {}
            totalWeight = np.sum(weights)
            for classLabel in np.unique(labels):
                leafValue[classLabel] = np.sum(weights[labels == classLabel]) / totalWeight if totalWeight > 0 else 0
            return Node(value=leafValue)

        leftMask = features[:, bestFeature] <= bestThreshold
        rightMask = features[:, bestFeature] > bestThreshold

        leftNode = self._buildTree(
            features[leftMask],
            labels[leftMask],
            weights[leftMask],
            currentDepth + 1
        )
        rightNode = self._buildTree(
            features[rightMask],
            labels[rightMask],
            weights[rightMask],
            currentDepth + 1
        )

        return Node(feature=bestFeature, threshold=bestThreshold, left=leftNode, right=rightNode)

    def _getNumFeaturesToTry(self, totalFeatures):
        if self.maxFeatures == 'sqrt':
            numFeaturesToTry = int(np.sqrt(totalFeatures))
        elif self.maxFeatures == 'log2':
            numFeaturesToTry = int(np.log2(totalFeatures))
        else:
            numFeaturesToTry = totalFeatures

        return max(1, min(numFeaturesToTry, totalFeatures))

    def fit(self, features, labels):
        self.classes_ = np.unique(labels)
        weights = np.ones(len(labels))

        if self.classWeight == 'balanced':
            classes, counts = np.unique(labels, return_counts=True)
            classWeights = {
                classLabel: len(labels) / (len(classes) * count)
                for classLabel, count in zip(classes, counts)
            }
            weights = np.array([classWeights[label] for label in labels])
        elif isinstance(self.classWeight, dict):
            weights = np.array([self.classWeight.get(label, 1.0) for label in labels])

        self.tree = self._buildTree(features, labels, weights, 0)

    def predict_proba(self, features):
        predictions = []
        baseProbabilities = {classLabel: 0.0 for classLabel in self.classes_}

        for sample in features:
            currentNode = self.tree
            while currentNode.value is None:
                if sample[currentNode.feature] <= currentNode.threshold:
                    currentNode = currentNode.left
                else:
                    currentNode = currentNode.right

            probabilities = baseProbabilities.copy()
            probabilities.update(currentNode.value)
            predictions.append([probabilities[classLabel] for classLabel in self.classes_])

        return np.array(predictions)


class RandomForest:
    def __init__(self, numEstimators=100, maxDepth=8, minSamplesSplit=5, maxFeatures='sqrt', classWeight=None):
        self.numEstimators = numEstimators
        self.treeParams = {
            'maxDepth': maxDepth,
            'minSamplesSplit': minSamplesSplit,
            'maxFeatures': maxFeatures,
            'classWeight': classWeight
        }
        self.trees = []
        self.classes_ = None

    def fit(self, features, labels):
        self.classes_ = np.unique(labels)
        numSamples = len(labels)

        for _ in range(self.numEstimators):
            bootstrapIndices = np.random.choice(numSamples, numSamples, replace=True)
            tree = GiniDecisionTree(**self.treeParams)
            tree.fit(features[bootstrapIndices], labels[bootstrapIndices])
            self.trees.append(tree)

    def predict_proba(self, features):
        if not self.trees:
            return np.zeros((len(features), len(self.classes_)))

        allProbabilities = np.array([tree.predict_proba(features) for tree in self.trees])
        return np.mean(allProbabilities, axis=0)

    def predict(self, features):
        probabilities = self.predict_proba(features)
        predictedIndices = np.argmax(probabilities, axis=1)
        return self.classes_[predictedIndices]


class SpecialistCommittee:
    def __init__(self):
        self.balancedModel = None
        self.geminiModel = None
        self.claudeModel = None
        self.classes_ = None

    def fit(self, features, labels):
        self.classes_ = np.unique(labels)
        self.balancedModel.fit(features, labels)
        self.geminiModel.fit(features, labels)
        self.claudeModel.fit(features, labels)

    def predict_proba(self, features):
        balancedProbs = self.balancedModel.predict_proba(features)
        geminiProbs = self.geminiModel.predict_proba(features)
        claudeProbs = self.claudeModel.predict_proba(features)
        return (balancedProbs + geminiProbs + claudeProbs) / 3.0

    def predict(self, features):
        probabilities = self.predict_proba(features)
        predictedIndices = np.argmax(probabilities, axis=1)
        return self.classes_[predictedIndices]


def main():
    print("Loading Training Data")
    trainingData = pd.read_csv("training_data.csv")

    features = extract_features(trainingData)
    labels = trainingData['label'].values

    featureMean = features.mean(0)
    featureStd = features.std(0) + 1e-8
    normalizedFeatures = (features - featureMean) / featureStd

    print("Training Stage 1: Gatekeeper...")
    binaryLabels = np.where(labels == 'ChatGPT', 'ChatGPT', 'Other')
    gatekeeperModel = RandomForest(
        numEstimators=200,
        maxDepth=8,
        minSamplesSplit=2,
        classWeight='balanced'
    )
    gatekeeperModel.fit(normalizedFeatures, binaryLabels)

    print("Training Stage 2: Specialist Committee")
    nonChatGptMask = labels != 'ChatGPT'
    committee = SpecialistCommittee()

    committee.balancedModel = RandomForest(
        numEstimators=150,
        maxDepth=8,
        minSamplesSplit=4,
        classWeight='balanced'
    )
    committee.geminiModel = RandomForest(
        numEstimators=150,
        maxDepth=6,
        minSamplesSplit=4,
        classWeight={'Gemini': 1.5, 'Claude': 1.0}
    )
    committee.claudeModel = RandomForest(
        numEstimators=150,
        maxDepth=6,
        minSamplesSplit=4,
        classWeight={'Gemini': 1.0, 'Claude': 1.0}
    )

    committee.fit(normalizedFeatures[nonChatGptMask], labels[nonChatGptMask])

    print("Saving to model.npz")
    hierarchicalModel = HierarchicalModel(gatekeeperModel, committee, featureMean, featureStd)

    def flattenTree(rootNode, classLabels):
        nodeList = []
        leafValues = []

        def traverse(node):
            if node.value is not None:
                valueIndex = len(leafValues)
                leafValues.append([float(node.value.get(c, 0.0)) for c in classLabels])
                nodeList.append((-1, 0.0, -1, -1, valueIndex))
                return len(nodeList) - 1

            leftIndex = traverse(node.left)
            rightIndex = traverse(node.right)
            nodeList.append((int(node.feature), float(node.threshold), leftIndex, rightIndex, -1))
            return len(nodeList) - 1

        traverse(rootNode)

        features = np.array([int(t[0]) for t in nodeList], dtype=int)
        thresholds = np.array([float(t[1]) for t in nodeList], dtype=float)
        leftIndices = np.array([int(t[2]) for t in nodeList], dtype=int)
        rightIndices = np.array([int(t[3]) for t in nodeList], dtype=int)
        valueIndices = np.array([int(t[4]) for t in nodeList], dtype=int)
        leafValueArray = np.array(leafValues, dtype=float) if leafValues else np.zeros((0, len(classLabels)), dtype=float)

        return features, thresholds, leftIndices, rightIndices, valueIndices, leafValueArray

    modelData = {}

    def serializeRandomForest(randomForest, prefix):
        classLabels = np.array(randomForest.classes_)
        modelData[f'{prefix}_classes'] = classLabels
        modelData[f'{prefix}_n_trees'] = randomForest.numEstimators

        for treeIndex, tree in enumerate(randomForest.trees):
            features, thresholds, leftIndices, rightIndices, valueIndices, leafValues = flattenTree(
                tree.tree,
                randomForest.classes_
            )
            modelData[f'{prefix}_tree{treeIndex}_features'] = features
            modelData[f'{prefix}_tree{treeIndex}_thresholds'] = thresholds
            modelData[f'{prefix}_tree{treeIndex}_lefts'] = leftIndices
            modelData[f'{prefix}_tree{treeIndex}_rights'] = rightIndices
            modelData[f'{prefix}_tree{treeIndex}_value_idxs'] = valueIndices
            modelData[f'{prefix}_tree{treeIndex}_leaf_values'] = leafValues

    serializeRandomForest(gatekeeperModel, 'gatekeeper')
    serializeRandomForest(committee.balancedModel, 'committee_model_bal')
    serializeRandomForest(committee.geminiModel, 'committee_model_gem')
    serializeRandomForest(committee.claudeModel, 'committee_model_cla')

    modelData['mu'] = np.array(featureMean, dtype=float)
    modelData['sig'] = np.array(featureStd, dtype=float)

    np.savez('model.npz', **modelData)
    print("Created 'model.npz'.")


if __name__ == "__main__":
    main()
