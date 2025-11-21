import pandas as pd
import numpy as np
from pred import extract_features, PureRF, PureDT, SpecialistCommittee, HierarchicalModel, Node

import numpy as _np

def main():
    print("Loading Training Data...")
    df = pd.read_csv("training_data.csv")
    
    print("Extracting Features...")
    X = extract_features(df) # Only returns X here based on pred.py structure? 
    # WAIT: The pred.py extract_features only returns X. 
    # Let's handle y manually here since pred.py is for test data.
    y = df['label'].values
    
    # Calculate Normalization Stats
    mu = X.mean(0)
    sig = X.std(0) + 1e-8
    X_norm = (X - mu) / sig
    
    print("Training Stage 1: Gatekeeper...")
    y_bin = np.where(y == 'ChatGPT', 'ChatGPT', 'Other')
    rf_gate = PureRF(n_estimators=200, max_depth=8, min_samples_split=2, class_weight='balanced')
    rf_gate.fit(X_norm, y_bin)
    
    print("Training Stage 2: Specialist Committee...")
    mask = y != 'ChatGPT'
    
    # Initialize the committee (using the Optimal params you found: Depth 6, Gem 2.5, Cla 1.2)
    comm = SpecialistCommittee()
    
    # Manually init the sub-models with optimal params
    comm.model_bal = PureRF(n_estimators=150, max_depth=6, min_samples_split=5, class_weight='balanced')
    comm.model_gem = PureRF(n_estimators=150, max_depth=6, min_samples_split=5, class_weight={'Gemini': 2.5, 'Claude': 1.0})
    comm.model_cla = PureRF(n_estimators=150, max_depth=6, min_samples_split=5, class_weight={'Gemini': 1.0, 'Claude': 1.2})
    
    comm.fit(X_norm[mask], y[mask])
    
    print("Saving to model.npz (numpy format, no pickle)...")
    full_model = HierarchicalModel(rf_gate, comm, mu, sig)

    def flatten_tree(root, classes):
        # Post-order traversal so children indices are lower than parent
        nodes = []  # tuples: (feature, threshold, left_idx, right_idx, value_idx)
        leaf_values = []

        def rec(n):
            if n.value is not None:
                vidx = len(leaf_values)
                leaf_values.append([float(n.value.get(c, 0.0)) for c in classes])
                nodes.append((-1, 0.0, -1, -1, vidx))
                return len(nodes) - 1
            left_idx = rec(n.left)
            right_idx = rec(n.right)
            nodes.append((int(n.feature), float(n.threshold), left_idx, right_idx, -1))
            return len(nodes) - 1

        rec(root)

        feats = _np.array([int(t[0]) for t in nodes], dtype=int)
        thresholds = _np.array([float(t[1]) for t in nodes], dtype=float)
        lefts = _np.array([int(t[2]) for t in nodes], dtype=int)
        rights = _np.array([int(t[3]) for t in nodes], dtype=int)
        value_idxs = _np.array([int(t[4]) for t in nodes], dtype=int)
        leaf_vals = _np.array(leaf_values, dtype=float) if leaf_values else _np.zeros((0, len(classes)), dtype=float)
        return feats, thresholds, lefts, rights, value_idxs, leaf_vals

    out = {}
    # Save gatekeeper
    def serialize_rf(rf, prefix):
        classes = _np.array(rf.classes_)
        out[f'{prefix}_classes'] = classes
        out[f'{prefix}_n_trees'] = rf.n
        for i, tree in enumerate(rf.trees):
            feats, thresholds, lefts, rights, value_idxs, leaf_vals = flatten_tree(tree.tree, rf.classes_)
            out[f'{prefix}_tree{i}_features'] = feats
            out[f'{prefix}_tree{i}_thresholds'] = thresholds
            out[f'{prefix}_tree{i}_lefts'] = lefts
            out[f'{prefix}_tree{i}_rights'] = rights
            out[f'{prefix}_tree{i}_value_idxs'] = value_idxs
            out[f'{prefix}_tree{i}_leaf_values'] = leaf_vals

    serialize_rf(rf_gate, 'gatekeeper')
    serialize_rf(comm.model_bal, 'committee_model_bal')
    serialize_rf(comm.model_gem, 'committee_model_gem')
    serialize_rf(comm.model_cla, 'committee_model_cla')

    out['mu'] = _np.array(mu, dtype=float)
    out['sig'] = _np.array(sig, dtype=float)

    _np.savez('model.npz', **out)
    print("Done! Created 'model.npz'. Upload 'pred.py' and 'model.npz'.")

if __name__ == "__main__":
    main()