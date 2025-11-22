"""
pred.py
Predictions using Hierarchical Random Forest.
Requires: model.npz (Pre-trained model)
"""
import numpy as np
import pandas as pd

# lightweight replacement for `re.match(r'^(\d+)', s)` without importing `re`
class _SimpleRe:
    class _Match:
        def __init__(self, s):
            self._s = s
        def group(self, i=0):
            if i == 1:
                return self._s
            return None

    @staticmethod
    def match(pattern, string):
        # Only supports patterns like r'^(\d+)' used in this file:
        s = str(string).lstrip()
        if not s:
            return None
        i = 0
        while i < len(s) and s[i].isdigit():
            i += 1
        if i == 0:
            return None
        return _SimpleRe._Match(s[:i])

# expose as `re` so existing code calling `re.match(...)` continues to work
re = _SimpleRe

# ============================================================================
# 1. MODEL DEFINITIONS (Required for pickle to load the classes)
# ============================================================================

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature, self.threshold, self.left, self.right, self.value = feature, threshold, left, right, value

class PureDT:
    def __init__(self, max_depth=8, min_samples_split=5, max_features='sqrt', class_weight=None):
        self.max_depth, self.min_samples_split, self.max_features, self.class_weight = max_depth, min_samples_split, max_features, class_weight
        self.tree = None
    
    # Methods required for inference only
    def _predict_one(self, x, node):
        if node.value is not None: return node.value
        if x[node.feature] <= node.threshold: return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict_proba(self, X):
        # Simplified inference logic for speed
        res = []
        base = {c: 0.0 for c in self.classes_}
        for row in X:
            node = self.tree
            while node.value is None:
                if row[node.feature] <= node.threshold: node = node.left
                else: node = node.right
            p = base.copy(); p.update(node.value)
            # Handle implicit classes if pickle didn't save them in tree
            res.append([p.get(c, 0.0) for c in self.classes_])
        return np.array(res)

class PureRF:
    def __init__(self, n_estimators=100, max_depth=8, min_samples_split=5, max_features='sqrt', class_weight=None):
        self.n = n_estimators
        self.params = {'max_depth': max_depth, 'min_samples_split': min_samples_split, 'max_features': max_features, 'class_weight': class_weight}
        self.trees = []
        self.classes_ = None
            
    def predict_proba(self, X):
        if not self.trees: return np.zeros((len(X), len(self.classes_)))
        probs = np.mean([t.predict_proba(X) for t in self.trees], axis=0)
        return probs

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        from collections import Counter
        n_samples, n_features = X.shape
        
        for _ in range(self.n):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            
            # Train a decision tree
            tree = PureDT(**self.params)
            tree.classes_ = self.classes_
            tree.tree = self._build_tree(X_sample, y_sample, depth=0)
            self.trees.append(tree)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        num_classes = len(np.unique(y))
        
        if (depth >= self.params['max_depth'] or n_samples < self.params['min_samples_split'] or num_classes == 1):
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_features, 
                                     int(np.sqrt(n_features)) if self.params['max_features'] == 'sqrt' else n_features, 
                                     replace=False)
        
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        if best_feat is None:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)
        
        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = X[:, best_feat] > best_thresh
        
        left = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(feature=best_feat, threshold=best_thresh, left=left, right=right)
    
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat in feat_idxs:
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                gain = self._information_gain(y, X[:, feat], thresh)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat
                    split_thresh = thresh
        return split_idx, split_thresh
    
    def _information_gain(self, y, feature_column, threshold):
        parent_entropy = self._entropy(y)
        
        left_idxs = feature_column <= threshold
        right_idxs = feature_column > threshold
        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0
        
        n = len(y)
        n_left, n_right = len(y[left_idxs]), len(y[right_idxs])
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right
        
        ig = parent_entropy - child_entropy
        return ig
    
    def _entropy(self, y):
        from collections import Counter
        hist = Counter(y)
        ps = [v / len(y) for v in hist.values()]
        return -sum(p * np.log2(p + 1e-9) for p in ps)
    
    def _calculate_leaf_value(self, y):
        from collections import Counter
        hist = Counter(y)
        total = len(y)
        return {cls: count / total for cls, count in hist.items()}

class SpecialistCommittee:
    def __init__(self):
        self.model_bal = None
        self.model_gem = None
        self.model_cla = None
        self.classes_ = None

    def predict_proba(self, X):
        p1 = self.model_bal.predict_proba(X)
        p2 = self.model_gem.predict_proba(X)
        p3 = self.model_cla.predict_proba(X)
        return (p1 + p2 + p3) / 3.0

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model_bal.fit(X, y)
        self.model_gem.fit(X, y)
        self.model_cla.fit(X, y)

class HierarchicalModel:
    def __init__(self, gatekeeper, committee, mu, sig):
        self.gatekeeper = gatekeeper
        self.committee = committee
        self.mu = mu
        self.sig = sig
        
    def predict(self, X_raw):
        # Normalize using training stats saved in the object
        X = (X_raw - self.mu) / self.sig
        
        # Gatekeeper
        idx_gpt = np.where(self.gatekeeper.classes_ == 'ChatGPT')[0][0]
        prob_gpt = self.gatekeeper.predict_proba(X)[:, idx_gpt]
        
        # Committee
        pred_spec = self.committee.predict(X)
        
        final_preds = []
        for i, p in enumerate(prob_gpt):
            if p > 0.55: 
                final_preds.append('ChatGPT')
            else:
                final_preds.append(pred_spec[i])
        return final_preds

# ============================================================================
# 2. FEATURE EXTRACTION (For Test Data)
# ============================================================================

def extract_features(df):
    # Confirmed Keywords
    gpt_kw = ['day', 'song', 'life', 'translation', 'learning', 'certain', 'stuck', 'tricky', 'slightly', 'concepts']
    gem_kw = ['phone', 'integration', 'collab', 'video', 'previous', 'overview', 'docs', 'ever', 'google', 'creative']
    cla_kw = ['implementation', 'cursor', 'running', 'frontend', 'apps', 'less', 'before', 'documents', 'schedule', 'used']
    trap_kw = ['trick', 'trap', 'weird', 'ignore', 'simon', 'jailbreak', 'limit', 'instruction', 'code word']

    def get_score(text, kws):
        txt = str(text).lower()
        return sum(1 for k in kws if k in txt)

    text_cols = [
        "In your own words, what kinds of tasks would you use this model for?",
        "Think of one task where this model gave you a suboptimal response. What did the response look like, and why did you find it suboptimal?",
        "When you verify a response from this model, how do you usually go about it?"
    ]
    
    # Handle missing columns in test data gracefully
    existing_cols = [c for c in text_cols if c in df.columns]
    if not existing_cols:
        df_text = pd.Series([""] * len(df))
    else:
        df_text = df[existing_cols].apply(lambda row: ' '.join(row.values.astype(str)).lower(), axis=1)
    
    txt_feats = []
    for t in df_text:
        txt_feats.append([get_score(t, gpt_kw), get_score(t, gem_kw), get_score(t, cla_kw), get_score(t, trap_kw)])
    
    def safe_rate(x):
        m = re.match(r'^(\d+)', str(x))
        return int(m.group(1)) if m else 0
    
    r_cols = [
        'How likely are you to use this model for academic tasks?',
        'Based on your experience, how often has this model given you a response that felt suboptimal?',
        'How often do you expect this model to provide responses with references or supporting evidence?',
        'How often do you verify this model\'s responses?'
    ]
    
    r_data = []
    for c in r_cols:
        if c in df.columns:
            r_data.append(df[c].apply(safe_rate).values)
        else:
            r_data.append(np.zeros(len(df)))
            
    ratings = np.array(r_data).T / 5.0
    
    all_tasks = ['Math', 'code', 'Data', 'Explaining', 'Converting', 'essays', 'Drafting']
    task_mat = np.zeros((len(df), len(all_tasks)))
    
    if 'Which types of tasks do you feel this model handles best? (Select all that apply.)' in df.columns:
        for i, r in enumerate(df['Which types of tasks do you feel this model handles best? (Select all that apply.)']):
            if pd.notna(r):
                for j, t in enumerate(all_tasks):
                    if t in str(r): task_mat[i, j] = 1

    X = np.hstack([ratings, np.array(txt_feats), task_mat])
    return X

# ============================================================================
# 3. PREDICT_ALL
# ============================================================================

# Global model cache
_MODEL = None


def predict_all(filename):
    """
    Make predictions for the data in filename.
    Loads 'model.pkl' which must be in the same directory.
    """
    global _MODEL
    
    # 1. Load Model (numpy .npz storage, no pickle/json at runtime)
    def load_rf_from_npz(npz, prefix):
        # prefix examples: 'gatekeeper', 'committee_model_bal', etc.
        classes = np.array(npz[f'{prefix}_classes'])
        n_trees = int(npz[f'{prefix}_n_trees'])

        class_list = classes.tolist()

        # For each tree, load node arrays
        trees = []
        for i in range(n_trees):
            p = f'{prefix}_tree{i}_'
            feats = npz[p + 'features']
            th = npz[p + 'thresholds']
            left = npz[p + 'lefts']
            right = npz[p + 'rights']
            vidx = npz[p + 'value_idxs']
            leaf_vals = npz[p + 'leaf_values']
            trees.append({'features': feats, 'thresholds': th, 'lefts': left, 'rights': right, 'value_idxs': vidx, 'leaf_values': leaf_vals})

        # Build a lightweight RF object that exposes predict_proba and predict
        class ArrayRF:
            def __init__(self, classes, trees):
                self.classes_ = np.array(classes)
                self._trees = trees

            def predict_proba(self, X):
                if len(self._trees) == 0:
                    return np.zeros((len(X), len(self.classes_)))
                probs = []
                for t in self._trees:
                    tree_probs = []
                    feats = t['features']
                    th = t['thresholds']
                    left = t['lefts'].astype(int)
                    right = t['rights'].astype(int)
                    vidx = t['value_idxs'].astype(int)
                    leaf_vals = t['leaf_values']
                    for row in X:
                        node = len(feats) - 1  # root is the last node appended during serialization
                        while True:
                            f = int(feats[node])
                            if f == -1:
                                v = leaf_vals[vidx[node]]
                                tree_probs.append(v)
                                break
                            if row[f] <= th[node]:
                                node = int(left[node])
                            else:
                                node = int(right[node])
                    probs.append(np.array(tree_probs))
                probs = np.mean(np.stack(probs, axis=0), axis=0)
                return probs

            def predict(self, X):
                idx = np.argmax(self.predict_proba(X), axis=1)
                return self.classes_[idx]

        return ArrayRF(class_list, trees)

    if _MODEL is None:
        try:
            npz = np.load('model.npz', allow_pickle=True)
        except FileNotFoundError:
            print("Error: model.npz not found. Please run training to create model.npz.")
            return ["ChatGPT"] * len(pd.read_csv(filename))

        gate = load_rf_from_npz(npz, 'gatekeeper')
        comm_bal = load_rf_from_npz(npz, 'committee_model_bal')
        comm_gem = load_rf_from_npz(npz, 'committee_model_gem')
        comm_cla = load_rf_from_npz(npz, 'committee_model_cla')

        # Load normalization stats
        mu = npz['mu']
        sig = npz['sig']

        # Build committee wrapper
        class Comm:
            def __init__(self, bal, gem, cla):
                self.model_bal = bal
                self.model_gem = gem
                self.model_cla = cla
                # assume all share same classes
                self.classes_ = self.model_bal.classes_

            def predict_proba(self, X):
                return (self.model_bal.predict_proba(X) + self.model_gem.predict_proba(X) + self.model_cla.predict_proba(X)) / 3.0

            def predict(self, X):
                probs = self.predict_proba(X)
                return self.classes_[np.argmax(probs, axis=1)]

        _MODEL = HierarchicalModel(gate, Comm(comm_bal, comm_gem, comm_cla), mu, sig)

    # 2. Load Test Data
    df = pd.read_csv(filename)
    
    # 3. Extract Features
    X = extract_features(df)
    
    # 4. Predict (Normalization is handled inside the model object)
    return _MODEL.predict(X)
