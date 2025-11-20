"""
pred.py
Predictions using Hierarchical Random Forest.
Requires: model.pkl (Pre-trained model)
"""
import sys
import numpy as np
import pandas as pd
import pickle
import os

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
    
    # 1. Load Model
    if _MODEL is None:
        try:
            with open('model.pkl', 'rb') as f:
                _MODEL = pickle.load(f)
        except FileNotFoundError:
            # If model is missing, we can't predict. 
            # Returning random as a desperate fallback or failing.
            print("Error: model.pkl not found.")
            return ["ChatGPT"] * len(pd.read_csv(filename))

    # 2. Load Test Data
    df = pd.read_csv(filename)
    
    # 3. Extract Features
    X = extract_features(df)
    
    # 4. Predict (Normalization is handled inside the model object)
    return _MODEL.predict(X)