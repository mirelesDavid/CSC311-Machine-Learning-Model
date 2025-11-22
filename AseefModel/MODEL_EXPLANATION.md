# Machine Learning Model Architecture

Complete explanation of the Hierarchical Random Forest model for AI model classification.

---

## 📊 **Overview**

This model classifies survey responses to determine which AI model (ChatGPT, Gemini, or Claude) a user was interacting with, achieving **74.19% accuracy**.

**Architecture:** Two-stage hierarchical Random Forest with specialist committee voting.

---

## **1. Feature Engineering: Finding Distinctive Keywords**

### **getDistinctiveKeywords.py**
Analyzes training data to find words uniquely associated with each AI model.

**How it works:**
```python
# For each AI model (ChatGPT, Gemini, Claude)
# Count how often each word appears in responses labeled as that model
# Compare to how often it appears in other models

# Example:
# "google" appears in 80% of Gemini responses
# "google" appears in 5% of ChatGPT responses
# → "google" is a distinctive keyword for Gemini
```

**Output:**
```python
chatGptKeywords = ['day', 'song', 'life', 'translation', 'learning', ...]
geminiKeywords = ['phone', 'google', 'collab', 'video', ...]
claudeKeywords = ['implementation', 'cursor', 'running', ...]
trapKeywords = ['trick', 'trap', 'jailbreak', ...] # Words used to trick the model
```

---

## **2. Feature Extraction: Converting Text to Numbers**

### **extract_features() function**
Machine learning models can't read text directly - they need **numbers**. This function converts survey responses into a feature vector.

**Input:** A survey response (text answers)
**Output:** A numerical array of 47 features

### **Feature Types:**

#### **A. Rating Features (4 features)**
```python
# Questions like "How likely are you to use this model for academic tasks?"
# Responses: "1 - Not likely", "5 - Very likely"
# Extract the number and normalize to 0-1 scale

ratingColumns = [
    'How likely are you to use this model for academic tasks?',
    'How often has this model given you a suboptimal response?',
    'How often do you expect references?',
    'How often do you verify responses?'
]
# Result: [0.8, 0.4, 0.6, 0.2] (4 features)
```

#### **B. Text/Keyword Features (36 features)**
```python
# For each text response, count keyword matches with WEIGHTS
# Example text: "I use this model for google docs and creative tasks"

calculateKeywordScore(text, chatGptKeywords, weight=1.0)   # Score: 0
calculateKeywordScore(text, geminiKeywords, weight=2.0)    # Score: 4 (2 keywords × 2.0)
calculateKeywordScore(text, claudeKeywords, weight=1.5)    # Score: 0
calculateKeywordScore(text, trapKeywords, weight=3.0)      # Score: 0

# Repeat for 3 text columns → 12 features per column → 36 total
```

**Why weights?**
- **Gemini keywords: 2.0×** - Very distinctive indicators
- **Claude keywords: 1.5×** - Moderate distinctiveness
- **Trap keywords: 3.0×** - Strong signals of manipulation attempts

#### **C. Task Features (7 features)**
```python
# Question: "Which types of tasks does this model handle best?"
# User selects: ["Math", "code", "Data"]

taskTypes = ['Math', 'code', 'Data', 'Explaining', 'Converting', 'essays', 'Drafting']
taskMatrix = [1, 1, 1, 0, 0, 0, 0]  # Binary: 1 if selected, 0 if not
```

### **Total Features:**
```
4 (ratings) + 36 (text keywords) + 7 (tasks) = 47 features per sample
```

---

## **3. Normalization**

Before training, we **normalize** the features:
```python
featureMean = features.mean(0)      # Average of each feature
featureStd = features.std(0) + 1e-8 # Standard deviation

normalizedFeatures = (features - featureMean) / featureStd

# Example:
# Before: [100, 5, 0.8]
# After:  [1.5, -0.3, 0.2]  (centered around 0, scaled to similar ranges)
```

**Why?** Prevents large features (like keyword counts) from dominating small features (like ratings 0-1).

---

## **4. The Model Architecture: Hierarchical Random Forest**

We use a **two-stage hierarchical model**:

```
                    ┌─────────────────┐
                    │  Input Features │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   GATEKEEPER    │
                    │ (Stage 1 Model) │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
        Is it ChatGPT?              Is it Gemini/Claude?
              │                             │
        ┌─────▼─────┐              ┌───────▼────────┐
        │  ChatGPT  │              │   COMMITTEE    │
        │ Prediction│              │ (Stage 2 Model)│
        └───────────┘              └───────┬────────┘
                                           │
                                  ┌────────┴────────┐
                                  │                 │
                            ┌─────▼─────┐    ┌─────▼──────┐
                            │   Gemini  │    │   Claude   │
                            │Prediction │    │ Prediction │
                            └───────────┘    └────────────┘
```

### **Stage 1: Gatekeeper (ChatGPT vs Others)**
```python
# Binary classifier: "Is this ChatGPT or not?"
gatekeeperModel = RandomForest(
    numEstimators=200,        # 200 decision trees
    maxDepth=8,              # Trees can be 8 levels deep
    minSamplesSplit=2,       # Split nodes with ≥2 samples
    classWeight='balanced'   # Handle class imbalance
)

# If probability(ChatGPT) > 0.55 → Predict "ChatGPT"
# Otherwise → Send to Stage 2
```

**Why separate ChatGPT?** ChatGPT has more distinctive patterns and is easier to identify.

### **Stage 2: Specialist Committee (Gemini vs Claude)**
```python
# Three specialized models that vote
committee = SpecialistCommittee()

# Model 1: Balanced (no bias)
balancedModel = RandomForest(
    numEstimators=150,
    maxDepth=8,
    classWeight='balanced'
)

# Model 2: Gemini Hunter (biased to catch Gemini)
geminiModel = RandomForest(
    numEstimators=150,
    maxDepth=6,
    classWeight={'Gemini': 1.5, 'Claude': 1.0}
)

# Model 3: Claude Hunter (neutral, deeper analysis)
claudeModel = RandomForest(
    numEstimators=150,
    maxDepth=6,
    classWeight={'Gemini': 1.0, 'Claude': 1.0}
)

# Final prediction = Average of 3 models (soft voting)
```

**Why a committee?** Reduces overfitting by combining diverse models with different biases.

---

## **5. Random Forest Explained**

A **Random Forest** is an ensemble of **Decision Trees**.

### **What's a Decision Tree?**

A decision tree is like a flowchart for making predictions:

```
                    ┌──────────────────────────┐
                    │ Is keyword score > 3.5?  │
                    └───────┬─────────┬────────┘
                            │         │
                        Yes │         │ No
                            │         │
                ┌───────────▼──┐   ┌──▼───────────┐
                │ Is rating<0.4│   │Is task='code'│
                └──────┬────┬──┘   └──┬────┬──────┘
                       │    │         │    │
                    ┌──▼─┐ ┌▼──┐   ┌─▼─┐ ┌▼──┐
                    │GPT │ │Gem│   │Cla│ │GPT│
                    └────┘ └───┘   └───┘ └───┘
                  Predictions (leaf nodes)
```

**How it's built:**
1. Start with all training data at the root
2. Find the **best feature + threshold** to split on (using Gini impurity)
3. Split data into left/right branches
4. Repeat recursively until stopping criteria (max depth, min samples, etc.)
5. Leaf nodes contain class probability distributions

---

## **6. Gini Impurity Explained**

**Gini measures how "mixed" or "impure" a group is.**

### **Formula:**
```python
Gini = 1 - Σ(probability_of_class_i)²
```

### **Examples:**

**Pure node (all same class):**
```python
Node: [ChatGPT, ChatGPT, ChatGPT, ChatGPT, ChatGPT]
probability_ChatGPT = 5/5 = 1.0

Gini = 1 - (1.0²) = 0  ← Perfect! (pure)
```

**Mixed node:**
```python
Node: [ChatGPT, ChatGPT, Gemini, Gemini, Claude]
prob_ChatGPT = 2/5 = 0.4
prob_Gemini = 2/5 = 0.4
prob_Claude = 1/5 = 0.2

Gini = 1 - (0.4² + 0.4² + 0.2²)
     = 1 - (0.16 + 0.16 + 0.04)
     = 0.64  ← Impure (mixed)
```

### **Information Gain:**
```python
# We want to REDUCE impurity with each split
InformationGain = ParentGini - WeightedChildGini

# Example split:
Parent: Gini = 0.64 (mixed)
Split on "keyword_score > 2.5"

Left child:  [ChatGPT, ChatGPT, ChatGPT] → Gini = 0.0
Right child: [Gemini, Gemini, Claude]    → Gini = 0.44

WeightedChildGini = (3/5 × 0.0) + (2/5 × 0.44) = 0.264
InformationGain = 0.64 - 0.264 = 0.376  ← Good split!
```

The tree picks the split with the **highest information gain** at each node.

---

## **7. GiniDecisionTree Class**

```python
class GiniDecisionTree:
    def _buildTree(self, features, labels, weights, currentDepth):
        # 1. Check stopping criteria
        if currentDepth >= maxDepth or numSamples < minSamplesSplit:
            return LeafNode(class_probabilities)

        # 2. Try random subset of features (feature bagging)
        selectedFeatures = random.choice(features, sqrt(numFeatures))

        # 3. For each feature, try different thresholds
        for featureIndex in selectedFeatures:
            for threshold in unique_values:
                # Calculate Gini impurity for this split
                leftGini = calculateGini(left_samples)
                rightGini = calculateGini(right_samples)

                # Calculate information gain
                gain = currentGini - weightedAverage(leftGini, rightGini)

                # Keep track of best split
                if gain > bestGain:
                    bestFeature = featureIndex
                    bestThreshold = threshold

        # 4. Recursively build left and right subtrees
        leftNode = buildTree(samples where feature <= threshold)
        rightNode = buildTree(samples where feature > threshold)

        return Node(feature=bestFeature, threshold=bestThreshold,
                   left=leftNode, right=rightNode)
```

---

## **8. Random Forest: Why Multiple Trees?**

A **Random Forest** trains many trees (e.g., 200) and averages their predictions.

### **Training Process:**

```python
class RandomForest:
    def fit(self, features, labels):
        for i in range(200):  # Train 200 trees
            # 1. Bootstrap: Sample data WITH replacement
            bootstrapIndices = random.choice(len(features), len(features))
            sampledData = features[bootstrapIndices]

            # 2. Train a decision tree on this sample
            tree = GiniDecisionTree()
            tree.fit(sampledData)
            self.trees.append(tree)

    def predict_proba(self, features):
        # 3. Average predictions from all trees (soft voting)
        allPredictions = [tree.predict(features) for tree in self.trees]
        return mean(allPredictions)
```

### **Why does this work better than a single tree?**

**Single Tree Problems:**
- ❌ Overfits to training data
- ❌ High variance (small changes in data = big changes in tree)
- ❌ Unstable predictions

**Random Forest Benefits:**
- ✅ **Bootstrap sampling** = different trees see different data
- ✅ **Random feature selection** = trees consider different features
- ✅ **Averaging** = reduces variance and overfitting
- ✅ More stable and accurate predictions!

---

## **9. Class Weights: Handling Imbalance**

```python
# Problem: Training data is imbalanced
# ChatGPT: 150 samples
# Gemini: 80 samples  ← Underrepresented!
# Claude: 120 samples

# Solution: Give Gemini samples higher weight
classWeight = {'Gemini': 1.5, 'Claude': 1.0}

# When calculating Gini, Gemini samples count 1.5×
# This forces the model to pay more attention to Gemini
```

**How it works:**
- Normal sample: contributes weight = 1.0 to Gini calculation
- Gemini sample: contributes weight = 1.5 to Gini calculation
- Result: Model learns to correctly classify Gemini despite fewer examples

---

## **10. The Complete Pipeline**

### **Training Phase:**
```python
1. Load training data (survey responses)
2. Extract features (47 numbers per sample)
3. Normalize features (mean=0, std=1)
4. Train Gatekeeper (200 trees, ChatGPT vs Others)
5. Train Committee (3 models × 150 trees each, Gemini vs Claude)
6. Save to model.npz (serialized tree structures)
```

### **Prediction Phase:**
```python
1. Load test data
2. Extract features (same 47 features)
3. Normalize using training stats (μ, σ)
4. Run through Gatekeeper
   → If prob(ChatGPT) > 0.55 → "ChatGPT"
   → Otherwise → Send to Committee
5. Committee votes (average of 3 models)
   → Return "Gemini" or "Claude"
```

---

## **11. Model Serialization (model.npz)**

The trained model is saved in NumPy's `.npz` format:

```python
model.npz contains:
├── gatekeeper_classes         # ['ChatGPT', 'Other']
├── gatekeeper_n_trees         # 200
├── gatekeeper_tree0_features  # Feature indices for each node
├── gatekeeper_tree0_thresholds # Split thresholds
├── gatekeeper_tree0_lefts     # Left child indices
├── gatekeeper_tree0_rights    # Right child indices
├── gatekeeper_tree0_value_idxs # Leaf value indices
├── gatekeeper_tree0_leaf_values # Class probabilities at leaves
├── ... (repeat for all 200 trees)
├── committee_model_bal_...    # Balanced model trees
├── committee_model_gem_...    # Gemini hunter trees
├── committee_model_cla_...    # Claude hunter trees
├── mu                         # Feature means (for normalization)
└── sig                        # Feature std devs (for normalization)
```

**Why .npz instead of pickle?**
- ✅ Faster loading
- ✅ Cross-platform compatible
- ✅ Smaller file size (~4MB vs ~10MB)
- ✅ No security risks from arbitrary code execution

---

## **12. Performance Analysis**

### **Test Results:**

| Metric | Accuracy | Notes |
|--------|----------|-------|
| **Overall** | **74.19%** | Strong performance |
| ChatGPT | 87.8% | Easiest to detect |
| Claude | 73.9% | Moderate difficulty |
| Gemini | 59.5% | Most challenging |

### **Confusion Matrix:**
```
                Predicted
Actual        ChatGPT    Claude    Gemini
ChatGPT          36         1         4
Claude            2        34        10
Gemini            5        10        22
```

### **Key Insights:**
- ✅ ChatGPT has very distinctive patterns (high accuracy)
- ⚠️ Gemini often confused with Claude (13 misclassifications)
- ✅ Low false positive rate for ChatGPT (only 2 Claude → ChatGPT)

---

## **13. Why This Architecture Works**

✅ **Hierarchical Structure**: Separates easy (ChatGPT) from hard (Gemini vs Claude)
✅ **Weighted Features**: Important keywords get more influence
✅ **Gini Impurity**: Efficient and effective splitting criterion
✅ **Random Forest**: Reduces overfitting through ensemble averaging
✅ **Committee Voting**: Diverse models with different biases vote together
✅ **Class Weights**: Handles imbalanced training data
✅ **Normalization**: Prevents feature scale issues
✅ **Bootstrap Sampling**: Creates diverse trees

**Result: 74.19% accuracy on unseen test data!** 🎉

---

## **14. Key Hyperparameters**

### **Gatekeeper:**
- `numEstimators`: 200 (more trees = better ChatGPT detection)
- `maxDepth`: 8 (deeper trees for complex patterns)
- `minSamplesSplit`: 2 (aggressive splitting)
- `classWeight`: 'balanced' (equal importance to both classes)

### **Committee:**
- **Balanced Model:**
  - `numEstimators`: 150
  - `maxDepth`: 8
  - `minSamplesSplit`: 4
  - `classWeight`: 'balanced'

- **Gemini Hunter:**
  - `numEstimators`: 150
  - `maxDepth`: 6 (shallower to prevent overfitting)
  - `minSamplesSplit`: 4
  - `classWeight`: {'Gemini': 1.5, 'Claude': 1.0} (bias toward Gemini)

- **Claude Hunter:**
  - `numEstimators`: 150
  - `maxDepth`: 6
  - `minSamplesSplit`: 4
  - `classWeight`: {'Gemini': 1.0, 'Claude': 1.0} (neutral)

### **Feature Weights:**
- ChatGPT keywords: 1.0×
- Gemini keywords: 2.0× (most distinctive)
- Claude keywords: 1.5×
- Trap keywords: 3.0× (strong signal)

---

## **15. Files Overview**

| File | Purpose |
|------|---------|
| `getDistinctiveKeywords.py` | Analyze training data to find distinctive keywords |
| `trainModel.py` | Train the hierarchical model and save to model.npz |
| `pred.py` | Load model and make predictions on new data |
| `model.npz` | Serialized trained model (trees + normalization stats) |
| `check_accuracy.py` | Evaluate model performance on test data |
| `run_prediction.py` | Simple script to run predictions and show distribution |

---

## **16. How to Use**

### **Training:**
```bash
python trainModel.py
# Output: model.npz
```

### **Prediction:**
```python
from pred import predict_all

predictions = predict_all("test_data.csv")
# Returns: ['ChatGPT', 'Gemini', 'Claude', ...]
```

### **Checking Accuracy:**
```bash
python check_accuracy.py
# Shows overall accuracy, per-class accuracy, and confusion matrix
```

---

## **17. Future Improvements**

Potential enhancements to increase accuracy:

1. **More distinctive keywords**: Deeper analysis of text patterns
2. **N-gram features**: Use word pairs/triplets instead of single words
3. **TF-IDF weighting**: Weight keywords by rarity across documents
4. **Deep learning**: Try neural networks (LSTM/Transformer) on text
5. **More training data**: Especially for Gemini (underrepresented)
6. **Feature engineering**: Add sentiment analysis, response length, punctuation patterns
7. **Ensemble methods**: Combine Random Forest with Gradient Boosting
8. **Hyperparameter tuning**: Grid search for optimal values

**Current bottleneck:** Gemini vs Claude distinction (similar response patterns)

---

**Model Version:** 1.0
**Last Updated:** 2025-01-22
**Accuracy:** 74.19%
