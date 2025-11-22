"""
pred.py
Predictions using Hierarchical Random Forest.
Requires: model.npz (Pre-trained model)
"""
import numpy as np
import pandas as pd


class _SimpleRegex:
    """Lightweight regex replacement to avoid importing re module"""

    class _Match:
        def __init__(self, matchedString):
            self._matchedString = matchedString

        def group(self, index=0):
            if index == 1:
                return self._matchedString
            return None

    @staticmethod
    def match(pattern, string):
        stringToMatch = str(string).lstrip()
        if not stringToMatch:
            return None

        digitCount = 0
        while digitCount < len(stringToMatch) and stringToMatch[digitCount].isdigit():
            digitCount += 1

        if digitCount == 0:
            return None

        return _SimpleRegex._Match(stringToMatch[:digitCount])


re = _SimpleRegex


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    def __init__(self, maxDepth=8, minSamplesSplit=5, maxFeatures='sqrt', classWeight=None):
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.maxFeatures = maxFeatures
        self.classWeight = classWeight
        self.tree = None

    def _predictSingle(self, sample, node):
        if node.value is not None:
            return node.value
        if sample[node.feature] <= node.threshold:
            return self._predictSingle(sample, node.left)
        return self._predictSingle(sample, node.right)

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
            predictions.append([probabilities.get(c, 0.0) for c in self.classes_])

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

    def predict_proba(self, features):
        if not self.trees:
            return np.zeros((len(features), len(self.classes_)))
        allProbabilities = np.mean([tree.predict_proba(features) for tree in self.trees], axis=0)
        return allProbabilities

    def predict(self, features):
        probabilities = self.predict_proba(features)
        predictedIndices = np.argmax(probabilities, axis=1)
        return self.classes_[predictedIndices]

    def fit(self, features, labels):
        from collections import Counter
        self.classes_ = np.unique(labels)
        numSamples = len(features)

        for _ in range(self.numEstimators):
            bootstrapIndices = np.random.choice(numSamples, numSamples, replace=True)
            sampledFeatures = features[bootstrapIndices]
            sampledLabels = labels[bootstrapIndices]

            tree = DecisionTree(**self.treeParams)
            tree.classes_ = self.classes_
            tree.tree = self._buildTree(sampledFeatures, sampledLabels, depth=0)
            self.trees.append(tree)

    def _buildTree(self, features, labels, depth):
        numSamples, numFeatures = features.shape
        numClasses = len(np.unique(labels))

        shouldStopSplitting = (
            depth >= self.treeParams['maxDepth'] or
            numSamples < self.treeParams['minSamplesSplit'] or
            numClasses == 1
        )

        if shouldStopSplitting:
            leafValue = self._calculateLeafValue(labels)
            return Node(value=leafValue)

        numFeaturesToTry = int(np.sqrt(numFeatures)) if self.treeParams['maxFeatures'] == 'sqrt' else numFeatures
        selectedFeatures = np.random.choice(numFeatures, numFeaturesToTry, replace=False)

        bestFeature, bestThreshold = self._findBestSplit(features, labels, selectedFeatures)
        if bestFeature is None:
            leafValue = self._calculateLeafValue(labels)
            return Node(value=leafValue)

        leftMask = features[:, bestFeature] <= bestThreshold
        rightMask = features[:, bestFeature] > bestThreshold

        leftNode = self._buildTree(features[leftMask], labels[leftMask], depth + 1)
        rightNode = self._buildTree(features[rightMask], labels[rightMask], depth + 1)

        return Node(feature=bestFeature, threshold=bestThreshold, left=leftNode, right=rightNode)

    def _findBestSplit(self, features, labels, featureIndices):
        bestGain = -1
        bestFeatureIndex = None
        bestThreshold = None

        for featureIndex in featureIndices:
            thresholds = np.unique(features[:, featureIndex])
            for threshold in thresholds:
                informationGain = self._calculateInformationGain(labels, features[:, featureIndex], threshold)
                if informationGain > bestGain:
                    bestGain = informationGain
                    bestFeatureIndex = featureIndex
                    bestThreshold = threshold

        return bestFeatureIndex, bestThreshold

    def _calculateInformationGain(self, labels, featureColumn, threshold):
        parentEntropy = self._calculateEntropy(labels)

        leftMask = featureColumn <= threshold
        rightMask = featureColumn > threshold

        if len(labels[leftMask]) == 0 or len(labels[rightMask]) == 0:
            return 0

        totalSamples = len(labels)
        leftSamples = len(labels[leftMask])
        rightSamples = len(labels[rightMask])

        leftEntropy = self._calculateEntropy(labels[leftMask])
        rightEntropy = self._calculateEntropy(labels[rightMask])

        childEntropy = (leftSamples / totalSamples) * leftEntropy + (rightSamples / totalSamples) * rightEntropy

        return parentEntropy - childEntropy

    def _calculateEntropy(self, labels):
        from collections import Counter
        labelCounts = Counter(labels)
        probabilities = [count / len(labels) for count in labelCounts.values()]
        return -sum(p * np.log2(p + 1e-9) for p in probabilities)

    def _calculateLeafValue(self, labels):
        from collections import Counter
        labelCounts = Counter(labels)
        totalSamples = len(labels)
        return {classLabel: count / totalSamples for classLabel, count in labelCounts.items()}


class SpecialistCommittee:
    def __init__(self):
        self.balancedModel = None
        self.geminiModel = None
        self.claudeModel = None
        self.classes_ = None

    def predict_proba(self, features):
        balancedProbs = self.balancedModel.predict_proba(features)
        geminiProbs = self.geminiModel.predict_proba(features)
        claudeProbs = self.claudeModel.predict_proba(features)
        return (balancedProbs + geminiProbs + claudeProbs) / 3.0

    def predict(self, features):
        probabilities = self.predict_proba(features)
        predictedIndices = np.argmax(probabilities, axis=1)
        return self.classes_[predictedIndices]

    def fit(self, features, labels):
        self.classes_ = np.unique(labels)
        self.balancedModel.fit(features, labels)
        self.geminiModel.fit(features, labels)
        self.claudeModel.fit(features, labels)


class HierarchicalModel:
    def __init__(self, gatekeeper, committee, featureMean, featureStd):
        self.gatekeeper = gatekeeper
        self.committee = committee
        self.featureMean = featureMean
        self.featureStd = featureStd

    def predict(self, rawFeatures):
        normalizedFeatures = (rawFeatures - self.featureMean) / self.featureStd

        chatGptIndex = np.where(self.gatekeeper.classes_ == 'ChatGPT')[0][0]
        chatGptProbabilities = self.gatekeeper.predict_proba(normalizedFeatures)[:, chatGptIndex]

        specialistPredictions = self.committee.predict(normalizedFeatures)

        finalPredictions = []
        for i, probability in enumerate(chatGptProbabilities):
            if probability > 0.55:
                finalPredictions.append('ChatGPT')
            else:
                finalPredictions.append(specialistPredictions[i])

        return finalPredictions


def extract_features(dataframe):
    chatGptKeywords = ['day', 'song', 'life', 'translation', 'learning', 'certain', 'stuck', 'tricky', 'slightly', 'concepts']
    geminiKeywords = ['phone', 'integration', 'collab', 'video', 'previous', 'overview', 'docs', 'ever', 'google', 'creative']
    claudeKeywords = ['implementation', 'cursor', 'running', 'frontend', 'apps', 'less', 'before', 'documents', 'schedule', 'used']
    trapKeywords = ['trick', 'trap', 'weird', 'ignore', 'simon', 'jailbreak', 'limit', 'instruction', 'code word']

    def calculateKeywordScore(text, keywords, weight=1.0):
        if pd.isna(text) or text == '':
            return 0.0
        lowerText = str(text).lower()
        return sum(weight for keyword in keywords if keyword in lowerText)

    textColumns = [
        "In your own words, what kinds of tasks would you use this model for?",
        "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
        "When you verify a response from this model, how do you usually go about it?"
    ]

    textFeatures = []
    for columnName in textColumns:
        columnFeatures = []
        if columnName in dataframe.columns:
            for text in dataframe[columnName]:
                features = [
                    calculateKeywordScore(text, chatGptKeywords, 1.0),
                    calculateKeywordScore(text, geminiKeywords, 2.0),
                    calculateKeywordScore(text, claudeKeywords, 1.5),
                    calculateKeywordScore(text, trapKeywords, 3.0)
                ]
                columnFeatures.append(features)
        else:
            for _ in range(len(dataframe)):
                columnFeatures.append([0.0, 0.0, 0.0, 0.0])
        textFeatures.append(np.array(columnFeatures))

    textFeatureMatrix = np.hstack(textFeatures)

    def extractRating(value):
        match = re.match(r'^(\d+)', str(value))
        return int(match.group(1)) if match else 0

    ratingColumns = [
        'How likely are you to use this model for academic tasks?',
        'Based on your experience, how often has this model given you a response that felt suboptimal?',
        'How often do you expect this model to provide responses with references or supporting evidence?',
        'How often do you verify this model\'s responses?'
    ]

    ratingData = []
    for columnName in ratingColumns:
        if columnName in dataframe.columns:
            ratingData.append(dataframe[columnName].apply(extractRating).values)
        else:
            ratingData.append(np.zeros(len(dataframe)))

    ratingMatrix = np.array(ratingData).T / 5.0

    taskTypes = ['Math', 'code', 'Data', 'Explaining', 'Converting', 'essays', 'Drafting']
    taskMatrix = np.zeros((len(dataframe), len(taskTypes)))

    taskColumnName = 'Which types of tasks do you feel this model handles best? (Select all that apply.)'
    if taskColumnName in dataframe.columns:
        for rowIndex, response in enumerate(dataframe[taskColumnName]):
            if pd.notna(response):
                for taskIndex, taskType in enumerate(taskTypes):
                    if taskType in str(response):
                        taskMatrix[rowIndex, taskIndex] = 1

    allFeatures = np.hstack([ratingMatrix, textFeatureMatrix, taskMatrix])
    return allFeatures


_MODEL = None


def predict_all(filename):
    """
    Make predictions for the data in filename.
    Loads 'model.npz' which must be in the same directory.
    """
    global _MODEL

    def loadRandomForestFromNpz(npzFile, prefix):
        classLabels = np.array(npzFile[f'{prefix}_classes'])
        numTrees = int(npzFile[f'{prefix}_n_trees'])
        classList = classLabels.tolist()

        treeData = []
        for treeIndex in range(numTrees):
            treePrefix = f'{prefix}_tree{treeIndex}_'
            features = npzFile[treePrefix + 'features']
            thresholds = npzFile[treePrefix + 'thresholds']
            leftIndices = npzFile[treePrefix + 'lefts']
            rightIndices = npzFile[treePrefix + 'rights']
            valueIndices = npzFile[treePrefix + 'value_idxs']
            leafValues = npzFile[treePrefix + 'leaf_values']

            treeData.append({
                'features': features,
                'thresholds': thresholds,
                'lefts': leftIndices,
                'rights': rightIndices,
                'value_idxs': valueIndices,
                'leaf_values': leafValues
            })

        class ArrayBasedRandomForest:
            def __init__(self, classes, trees):
                self.classes_ = np.array(classes)
                self._trees = trees

            def predict_proba(self, features):
                if len(self._trees) == 0:
                    return np.zeros((len(features), len(self.classes_)))

                allTreeProbabilities = []
                for treeDict in self._trees:
                    treeProbabilities = []
                    treeFeatures = treeDict['features']
                    treeThresholds = treeDict['thresholds']
                    leftIndices = treeDict['lefts'].astype(int)
                    rightIndices = treeDict['rights'].astype(int)
                    valueIndices = treeDict['value_idxs'].astype(int)
                    leafValues = treeDict['leaf_values']

                    for sample in features:
                        nodeIndex = len(treeFeatures) - 1
                        while True:
                            featureIndex = int(treeFeatures[nodeIndex])
                            if featureIndex == -1:
                                probability = leafValues[valueIndices[nodeIndex]]
                                treeProbabilities.append(probability)
                                break

                            if sample[featureIndex] <= treeThresholds[nodeIndex]:
                                nodeIndex = int(leftIndices[nodeIndex])
                            else:
                                nodeIndex = int(rightIndices[nodeIndex])

                    allTreeProbabilities.append(np.array(treeProbabilities))

                averageProbabilities = np.mean(np.stack(allTreeProbabilities, axis=0), axis=0)
                return averageProbabilities

            def predict(self, features):
                probabilities = self.predict_proba(features)
                predictedIndices = np.argmax(probabilities, axis=1)
                return self.classes_[predictedIndices]

        return ArrayBasedRandomForest(classList, treeData)

    if _MODEL is None:
        try:
            npzFile = np.load('model.npz', allow_pickle=True)
        except FileNotFoundError:
            print("Error: model.npz not found. Please run training to create model.npz.")
            return ["ChatGPT"] * len(pd.read_csv(filename))

        gatekeeperModel = loadRandomForestFromNpz(npzFile, 'gatekeeper')
        balancedCommitteeModel = loadRandomForestFromNpz(npzFile, 'committee_model_bal')
        geminiCommitteeModel = loadRandomForestFromNpz(npzFile, 'committee_model_gem')
        claudeCommitteeModel = loadRandomForestFromNpz(npzFile, 'committee_model_cla')

        featureMean = npzFile['mu']
        featureStd = npzFile['sig']

        class Committee:
            def __init__(self, balancedModel, geminiModel, claudeModel):
                self.balancedModel = balancedModel
                self.geminiModel = geminiModel
                self.claudeModel = claudeModel
                self.classes_ = self.balancedModel.classes_

            def predict_proba(self, features):
                balancedProbs = self.balancedModel.predict_proba(features)
                geminiProbs = self.geminiModel.predict_proba(features)
                claudeProbs = self.claudeModel.predict_proba(features)
                return (balancedProbs + geminiProbs + claudeProbs) / 3.0

            def predict(self, features):
                probabilities = self.predict_proba(features)
                return self.classes_[np.argmax(probabilities, axis=1)]

        committee = Committee(balancedCommitteeModel, geminiCommitteeModel, claudeCommitteeModel)
        _MODEL = HierarchicalModel(gatekeeperModel, committee, featureMean, featureStd)

    testData = pd.read_csv(filename)
    features = extract_features(testData)

    return _MODEL.predict(features)
