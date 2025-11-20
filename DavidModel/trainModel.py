import pickle
import pandas as pd
import numpy as np
# Import everything from pred to ensure class compatibility
from pred import extract_features, PureRF, PureDT, SpecialistCommittee, HierarchicalModel

def main():
    print("Loading Training Data...")
    df = pd.read_csv("training_data.csv")
    
    print("Extracting Features...")
    X, y = extract_features(df) # Only returns X here based on pred.py structure? 
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
    
    print("Saving to model.pkl...")
    full_model = HierarchicalModel(rf_gate, comm, mu, sig)
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(full_model, f)
    print("Done! Upload 'pred.py' and 'model.pkl'.")

if __name__ == "__main__":
    main()