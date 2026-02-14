import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

def prepare_discrepancy_models(model1, model2):
    """
    Args:
        model1, model2: Trained sklearn ensemble models (AdaBoost or GradientBoosting).

    Returns:
        combined_estimators: List of all decision trees from both models.
        weights_1: Weight vector activating only Model 1 (Model 2 weights set to 0).
        weights_2: Weight vector activating only Model 2 (Model 1 weights set to 0).
    """
    # 1. Extract estimators and weights
    est1 = _get_estimators(model1)
    w1_raw = _get_weights(model1)

    est2 = _get_estimators(model2)
    w2_raw = _get_weights(model2)

    # 2. Concatenate Estimators into one list
    # This creates a "Super Ensemble" containing every tree from both models.
    combined_estimators = est1 + est2

    # 3. Create Weight Vectors (Padding with zeros)
    
    # Weights for Model 1: [ w1_tree1, w1_tree2, ..., 0, 0, ... ]
    # It effectively "turns off" all trees belonging to Model 2.
    weights_1 = np.concatenate([w1_raw, np.zeros(len(est2))])

    # Weights for Model 2: [ 0, 0, ..., w2_tree1, w2_tree2, ... ]
    # It effectively "turns off" all trees belonging to Model 1.
    weights_2 = np.concatenate([np.zeros(len(est1)), w2_raw])

    return combined_estimators, weights_1, weights_2

def _get_estimators(model):
    """Helper to extract trees from different sklearn model types."""
    if hasattr(model, 'estimators_'):
        # GradientBoosting stores trees as [[Tree], [Tree]] (n_est, n_classes)
        # AdaBoost stores trees as [Tree, Tree]
        estimators = np.array(model.estimators_).flatten().tolist()
        return estimators
    raise ValueError(f"Model {type(model)} must have .estimators_ attribute")

def _get_weights(model):
    """Helper to extract/generate weights."""
    if hasattr(model, 'estimator_weights_'):
        # AdaBoost has explicit weights
        return model.estimator_weights_
    
    if hasattr(model, 'learning_rate'):
        # Gradient Boosting trees are usually weighted by the learning rate
        # Note: GBDT handling can be complex if it has an initial bias (init=...). 
        # For standard comparisons, assuming uniform LR weights is usually sufficient.
        estimators = _get_estimators(model)
        return np.full(len(estimators), model.learning_rate)
        
    raise ValueError(f"Could not determine weights for model {type(model)}")