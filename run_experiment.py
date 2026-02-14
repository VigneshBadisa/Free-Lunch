# import numpy as np
# import pandas as pd
# import copy

# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

# # Import from your codebase
# # Note the 'fipe.' prefix
# from fipe.discrepancy_oracle import DiscrepancyOracle
# from fipe.model_prep import prepare_discrepancy_models
# from fipe.feature import FeatureEncoder

# # 1. GENERATE DATA & TRAIN MODELS
# print("Training models...")
# X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Train Model A (AdaBoost)
# model_a = AdaBoostClassifier(n_estimators=10, random_state=42)
# model_a.fit(X_train, y_train)

# # Train Model B (Gradient Boosting - often simpler/different decision boundaries)
# model_b = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=2, random_state=42)
# model_b.fit(X_train, y_train)

# # 2. PREPARE INPUTS FOR ORACLE
# print("Preparing Discrepancy Oracle inputs...")
# combined_estimators, w_A, w_B = prepare_discrepancy_models(model_a, model_b)

# # --- FIX START ---
# # Convert numpy array to pandas DataFrame
# # FeatureEncoder expects a DataFrame so it can call .dropna() and use column names
# df_train = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])

# # Initialize the Feature Encoder with the DataFrame, not the numpy array
# encoder = FeatureEncoder(df_train)
# # 3. INITIALIZE ORACLE
# # We initialize it with the "Super Ensemble" (all trees)
# # and set the "base weights" to Model A.
# # oracle = DiscrepancyOracle(
# #     base=combined_estimators, 
# #     encoder=encoder, 
# #     weights=w_A
# # )
# super_ensemble_model = copy.deepcopy(model_a) 

# # We overwrite its internal list of trees with our combined list
# super_ensemble_model.estimators_ = combined_estimators

# # (Optional safety) Ensure the shell knows it has more estimators now
# # Some libraries check the length of this attribute
# if hasattr(super_ensemble_model, 'n_estimators'):
#     super_ensemble_model.n_estimators = len(combined_estimators)

# # 3. INITIALIZE ORACLE
# # Pass the 'super_ensemble_model' (the object) instead of the list
# oracle = DiscrepancyOracle(
#     base=super_ensemble_model, 
#     encoder=encoder, 
#     weights=w_A
# )
# # Build the Gurobi model (constraints for Model A)
# oracle.build()

# # 4. RUN SEARCH
# print("Searching for discrepancy points...")
# disagreements = []

# # We pass Model B's weights here. The oracle will try to find x such that:
# # Model A (fixed) predicts Class X
# # Model B (optimized) predicts Class Y != X
# for x_discrepancy in oracle.find_discrepancies(model2_weights=w_B):
#     print(f"\nFound discrepancy at input: {x_discrepancy}")
    
#     # Double check manually (optional, already done inside Oracle)
#     pred_a = model_a.predict([x_discrepancy])[0]
#     pred_b = model_b.predict([x_discrepancy])[0]
#     print(f"Model A predicts: {pred_a}")
#     print(f"Model B predicts: {pred_b}")
    
#     disagreements.append(x_discrepancy)

# print(f"\nTotal discrepancy points found: {len(disagreements)}")


import pandas as pd
import numpy as np
import copy
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from fipe.discrepancy_oracle import DiscrepancyOracle
from fipe.model_prep import prepare_discrepancy_models
from fipe.feature import FeatureEncoder

print("Generating data...")
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training models...")

# Model A: Standard AdaBoost (Depth = 1)
model_a = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=10,
    random_state=42
)
model_a.fit(X_train, y_train)

# Model B: AdaBoost with Depth = 2
model_b = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2),
    n_estimators=10,
    random_state=42
)
model_b.fit(X_train, y_train)



print("Preparing Discrepancy Oracle inputs...")
combined_estimators, w_A, w_B = prepare_discrepancy_models(model_a, model_b)

df_train = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(X_train.shape[1])])
encoder = FeatureEncoder(df_train)


super_ensemble_model = copy.deepcopy(model_a)
super_ensemble_model.estimators_ = combined_estimators
if hasattr(super_ensemble_model, 'n_estimators'):
    super_ensemble_model.n_estimators = len(combined_estimators)

oracle = DiscrepancyOracle(
    base=super_ensemble_model,
    encoder=encoder,
    weights=w_A  # Initialize with Model A active
)

oracle.build()

print("Searching for discrepancy points...")
disagreements = []

training_features = df_train.columns.tolist()

for x_discrepancy in oracle.find_discrepancies(
    model2_weights=w_B, 
    model_A=model_a, 
    model_B=model_b,
    feature_names=training_features
):

    x_input = x_discrepancy[training_features].values.reshape(1, -1)
    
    pred_a = model_a.predict(x_input)[0]
    pred_b = model_b.predict(x_input)[0]
    
    if pred_a != pred_b:
        print("\n" + "="*40)
        print(f">> CONFIRMED DISCREPANCY")
        print("="*40)
        print(f"Input:\n{x_discrepancy}")
        print(f"Model A (Depth 1): {pred_a}")
        print(f"Model B (Depth 2): {pred_b}")
        disagreements.append(x_discrepancy)

print(f"\nTotal discrepancy points found: {len(disagreements)}")