from collections.abc import Callable, Generator
from functools import partial

import gurobipy as gp
import numpy as np
import numpy.typing as npt

from .ocean import OCEAN
from .typing import SNumber


class DiscrepancyOracle(OCEAN):
    """
    Oracle to compute:
    max_x | h1_{y1(x)}(x) - h2_{y2(x)}(x) |
    """

    def __call__(self) -> float:
        max_val = -np.inf

        for c1 in range(self.n_classes):
            self.set_maj_class(model=1, maj_class=c1)

            for c2 in range(self.n_classes):
                self.set_maj_class(model=2, maj_class=c2)

                
                val1 = self._solve_pair(c1, c2, sign=+1)
                max_val = max(max_val, val1)

               
                val2 = self._solve_pair(c1, c2, sign=-1)
                max_val = max(max_val, val2)

                self.clear_maj_class(model=2)

            self.clear_maj_class(model=1)

        return max_val
    
    def _solve_pair(self, c1: int, c2: int, sign: int) -> float:
   

        wf1 = partial(self.weighted_function, model=1)
        wf2 = partial(self.weighted_function, model=2)

        if sign == +1:
            obj = wf1(class_=c1) - wf2(class_=c2)
        else:
            obj = wf2(class_=c2) - wf1(class_=c1)

        self.setObjective(obj, gp.GRB.MAXIMIZE)
        self.optimize(self._get_optimize_callback())

        return self.ObjBound
    
    @staticmethod
    def _get_optimize_callback():
        def cb(model: gp.Model, where: int):
            if where == gp.GRB.Callback.MIPSOL:
                val = model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)
                if val <= 0.0:
                    model.terminate()
        return cb


