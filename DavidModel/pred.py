"""
Final Stable Random Forest (Pure Numpy)
Strategy: Two-Stage with "Committee" Voting
Goal: Prevent overfitting by averaging diverse models for Stage 2.
"""

import numpy as np
import pandas as pd
import re

np.random.seed(42)

# ============================================================================
# 1. PURE NUMPY RF ENGINE (Optimized)
# ============================================================================

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class PureDT:
    def __init__(self, max_depth=8, min_samples_split=5, max_features='sqrt', class_weight=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.class_weight = class_weight
        self.tree = None

    def _gini(self, y, weights):
        if len(y) == 0: return 0
        classes = np.unique(y)
        total = np.sum(weights)
        if total == 0: return 0
        gini = 1.0
        for c in classes:
            p = np.sum(weights[y == c]) / total
            gini -= p**2
        return gini

    def _split(self, X, y, weights, depth):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) <= 1:
            val = {}
            total = np.sum(weights)
            for c in np.unique(y):
                val[c] = np.sum(weights[y == c]) / total if total > 0 else 0
            return Node(value=val)

        if self.max_features == 'sqrt': n_try = int(np.sqrt(n_features))
        elif self.max_features == 'log2': n_try = int(np.log2(n_features))
        else: n_try = n_features
        n_try = max(1, min(n_try, n_features))
        
        feats = np.random.choice(n_features, n_try, replace=False)
        best_gain = -1; best_f = None; best_t = None
        current_gini = self._gini(y, weights)
        total_w = np.sum(weights)

        for f in feats:
            thresholds = np.unique(X[:, f])
            if len(thresholds) > 15: thresholds = np.percentile(thresholds, np.linspace(10, 90, 10))
            for t in thresholds:
                left = X[:, f] <= t
                right = ~left
                if np.sum(left) == 0 or np.sum(right) == 0: continue
                w_l, w_r = weights[left], weights[right]
                sum_w_l, sum_w_r = np.sum(w_l), np.sum(w_r)
                gain = current_gini - ((sum_w_l/total_w) * self._gini(y[left], w_l) + (sum_w_r/total_w) * self._gini(y[right], w_r))
                if gain > best_gain: best_gain = gain; best_f = f; best_t = t

        if best_gain <= 1e-6:
            val = {}
            total = np.sum(weights)
            for c in np.unique(y): val[c] = np.sum(weights[y == c]) / total if total > 0 else 0
            return Node(value=val)

        l_node = self._split(X[X[:, best_f] <= best_t], y[X[:, best_f] <= best_t], weights[X[:, best_f] <= best_t], depth+1)
        r_node = self._split(X[X[:, best_f] > best_t], y[X[:, best_f] > best_t], weights[X[:, best_f] > best_t], depth+1)
        return Node(feature=best_f, threshold=best_t, left=l_node, right=r_node)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        weights = np.ones(len(y))
        if self.class_weight == 'balanced':
            classes, counts = np.unique(y, return_counts=True)
            cw = {c: len(y)/(len(classes)*cnt) for c, cnt in zip(classes, counts)}
            weights = np.array([cw[yi] for yi in y])
        elif isinstance(self.class_weight, dict):
            weights = np.array([self.class_weight.get(yi, 1.0) for yi in y])
        self.tree = self._split(X, y, weights, 0)

    def predict_proba(self, X):
        res = []
        base = {c: 0.0 for c in self.classes_}
        for row in X:
            node = self.tree
            while node.value is None:
                if row[node.feature] <= node.threshold: node = node.left
                else: node = node.right
            p = base.copy(); p.update(node.value)
            res.append([p[c] for c in self.classes_])
        return np.array(res)

class PureRF:
    def __init__(self, n_estimators=100, max_depth=8, min_samples_split=5, max_features='sqrt', class_weight=None):
        self.n = n_estimators
        self.params = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'max_features': max_features, 'class_weight': class_weight}
        self.trees = []
        self.classes_ = None
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n = len(y)
        for _ in range(self.n):
            idx = np.random.choice(n, n, replace=True)
            t = PureDT(**self.params)
            t.fit(X[idx], y[idx])
            self.trees.append(t)
            
    def predict_proba(self, X):
        if not self.trees: return np.zeros((len(X), len(self.classes_)))
        probs = np.array([t.predict_proba(X) for t in self.trees])
        return np.mean(probs, axis=0)

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]

# ============================================================================
# 2. COMMITTEE CLASSIFIER (The Fix)
# ============================================================================

class SpecialistCommittee:
    """Combines 3 diverse models to prevent overfitting to one specific bias"""
    def __init__(self):
        # 1. The Conservative (Balanced)
        self.model_bal = PureRF(n_estimators=150, max_depth=8, min_samples_split=5, class_weight='balanced')
        # 2. The Gemini Hunter (High Recall for Gemini)
        self.model_gem = PureRF(n_estimators=150, max_depth=6, min_samples_split=5, class_weight={'Gemini': 2.0, 'Claude': 1.0})
        # 3. The Claude Hunter (High Recall for Claude)
        self.model_cla = PureRF(n_estimators=150, max_depth=6, min_samples_split=5, class_weight={'Gemini': 1.0, 'Claude': 1.5})
        
        self.classes_ = None

    def fit(self, X, y):
        print("    Training Committee Member 1: Conservative...")
        self.model_bal.fit(X, y)
        print("    Training Committee Member 2: Gemini Hunter...")
        self.model_gem.fit(X, y)
        print("    Training Committee Member 3: Claude Hunter...")
        self.model_cla.fit(X, y)
        self.classes_ = self.model_bal.classes_

    def predict_proba(self, X):
        p1 = self.model_bal.predict_proba(X)
        p2 = self.model_gem.predict_proba(X)
        p3 = self.model_cla.predict_proba(X)
        # Average the probabilities (Soft Voting)
        return (p1 + p2 + p3) / 3.0

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]

# ============================================================================
# 3. FEATURE EXTRACTION
# ============================================================================

def extract_features(df):
    def safe_rate(x):
        m = re.match(r'^(\d+)', str(x))
        return int(m.group(1)) if m else 0
    
    r_cols = [
        'How likely are you to use this model for academic tasks?',
        'Based on your experience, how often has this model given you a response that felt suboptimal?',
        'How often do you expect this model to provide responses with references or supporting evidence?',
        'How often do you verify this model\'s responses?'
    ]
    ratings = np.array([df[c].apply(safe_rate).values for c in r_cols]).T / 5.0

    def get_score(text, kws, w=1.0):
        if pd.isna(text) or text == '': return 0.0
        txt = str(text).lower()
        return sum(w for k in kws if k in txt)

    gpt_kw = ['day', 'song', 'life', 'translation', 'learning', 'certain', 'stuck', 'tricky', 'slightly', 'concepts']
    gem_kw = ['phone', 'integration', 'collab', 'video', 'previous', 'overview', 'docs', 'ever', 'google', 'creative']
    cla_kw = ['implementation', 'cursor', 'running', 'frontend', 'apps', 'less', 'before', 'documents', 'schedule', 'used']
    trap_kw = ['trick', 'trap', 'weird', 'ignore', 'simon', 'jailbreak', 'limit', 'instruction', 'code word']

    txt_cols = [
        "In your own words, what kinds of tasks would you use this model for?",
        "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
        "When you verify a response from this model, how do you usually go about it?"
    ]
    
    feats = []
    for col in txt_cols:
        col_f = []
        for t in df[col]:
            f = [get_score(t, gpt_kw, 1.0), get_score(t, gem_kw, 2.0), get_score(t, cla_kw, 1.5), get_score(t, trap_kw, 3.0)]
            col_f.append(f)
        feats.append(np.array(col_f))
    txt_feats = np.hstack(feats)
    
    all_tasks = ['Math', 'code', 'Data', 'Explaining', 'Converting', 'essays', 'Drafting']
    task_mat = np.zeros((len(df), len(all_tasks)))
    for i, r in enumerate(df['Which types of tasks do you feel this model handles best? (Select all that apply.)']):
        if pd.notna(r):
            for j, t in enumerate(all_tasks):
                if t in str(r): task_mat[i, j] = 1

    X = np.hstack([ratings, txt_feats, task_mat])
    y = df['label'].values
    return X, y

# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Loading Data...")
    train = pd.read_csv("training_data.csv")
    test = pd.read_csv("test_data.csv")
    
    print("Extracting Features...")
    X_train, y_train = extract_features(train)
    X_test, y_test = extract_features(test)
    
    mu, sig = X_train.mean(0), X_train.std(0) + 1e-8
    X_train = (X_train - mu) / sig
    X_test = (X_test - mu) / sig
    
    print("\nTraining Hierarchical Model (Stable Committee Version)...")
    
    # --- STAGE 1: GATEKEEPER ---
    print("  Stage 1: Training Gatekeeper (ChatGPT vs Others)...")
    y_train_bin = np.where(y_train == 'ChatGPT', 'ChatGPT', 'Other')
    # Using settings we know are robust (from your 91% val run)
    rf_gate = PureRF(n_estimators=200, max_depth=8, min_samples_split=2, class_weight='balanced')
    rf_gate.fit(X_train, y_train_bin)
    
    # --- STAGE 2: SPECIALIST COMMITTEE ---
    print("  Stage 2: Training Specialist Committee (Gemini vs Claude)...")
    mask_train = y_train != 'ChatGPT'
    # Using Committee instead of single over-tuned model
    committee = SpecialistCommittee()
    committee.fit(X_train[mask_train], y_train[mask_train])
    
    # --- PREDICTION ---
    print("Predicting...")
    idx_gpt = np.where(rf_gate.classes_ == 'ChatGPT')[0][0]
    prob_gpt = rf_gate.predict_proba(X_test)[:, idx_gpt]
    pred_spec = committee.predict(X_test)
    
    final_preds = []
    for i, p in enumerate(prob_gpt):
        # Safe threshold
        if p > 0.55: final_preds.append('ChatGPT')
        else: final_preds.append(pred_spec[i])
        
    final_preds = np.array(final_preds)
    
    # Results
    acc = np.mean(final_preds == y_test)
    print(f"\nFinal Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    print("\nPer-Class Accuracy:")
    for c in np.unique(y_test):
        mask = y_test == c
        s = np.mean(final_preds[mask] == y_test[mask])
        print(f"  {c}: {s*100:.1f}%")