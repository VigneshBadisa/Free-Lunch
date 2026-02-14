import gurobipy as gp
import numpy as np
import pandas as pd
from collections.abc import Generator
import numpy.typing as npt

from .oracle import Oracle
from .typing import SNumber

class DiscrepancyOracle(Oracle):
    """
    A Deterministic Oracle that finds verified disagreement points using Ground Truth models.
    """
    
    MIN_MARGIN = 1e-4 

    def __init__(self, *args, **kwargs):
        if 'eps' not in kwargs:
            kwargs['eps'] = 1e-3
        super().__init__(*args, **kwargs)
        self.real_model_A = None
        self.real_model_B = None
        self.feature_names = None

    def find_discrepancies(self, model2_weights: npt.ArrayLike, model_A, model_B, feature_names: list[str] = None) -> Generator[SNumber, None, None]:
        self.new_weights = np.copy(model2_weights)
        self.real_model_A = model_A
        self.real_model_B = model_B
        self.feature_names = feature_names
        self.setParam("Seed", 42)
        self.setParam("PoolSearchMode", 2)
        self.setParam("Threads", 1) 
        
        yield from self._separate()

    def _extract_samples(self, majority_class: int, class_: int) -> Generator[SNumber, None, None]:
        param = gp.GRB.Param.SolutionNumber
        
        seen_points = []

        for i in range(self.SolCount):
            self.setParam(param, i)

            if self.PoolObjVal < self.MIN_MARGIN:
                continue

            x = self._feature_vars.Xn
            
            is_duplicate = False
            for seen_x in seen_points:
                if np.allclose(x, seen_x, atol=1e-5):
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            X = self.transform(x)
            
            if self.feature_names is not None:
                # Best case: Use the order passed from run_experiment
                cols = self.feature_names
            elif hasattr(self.encoder, 'features'):
                # Fallback: SORT the features so it's 'feature_0', 'feature_1'... every time
                # converting set -> list is random. sorted() fixes it.
                cols = sorted(list(self.encoder.features))
            else:
                cols = None

            if cols is not None:
                df_aligned = pd.DataFrame(X, columns=cols)
                X_input = df_aligned.values # Convert back to numpy
            else:
                X_input = X

            pred_A = self.real_model_A.predict(X_input)[0]
            pred_B = self.real_model_B.predict(X_input)[0]

            if pred_A == pred_B:
                continue
            
            # sachme discrepancy hai, add to memory and yield
            seen_points.append(x)
            
            if cols is not None:
                yield pd.Series(x, index=cols)
            else:
                yield x
                
        self.setParam(param, 0)