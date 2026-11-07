#!/usr/bin/env python3
"""Observe committed LBFGS updates without replacing its step implementation."""
import torch

class ObservedLBFGS(torch.optim.LBFGS):
    def __init__(self, params, *, on_accepted, **kwargs):
        super().__init__(params, **kwargs)
        assert self.param_groups[0]['line_search_fn']=='strong_wolfe'
        self._trial_depth=0
        self._on_accepted=on_accepted
        self.accepted_iterations=[]

    def _directional_evaluate(self, closure, x, t, d):
        self._trial_depth+=1
        try:
            return super()._directional_evaluate(closure,x,t,d)
        finally:
            self._trial_depth-=1

    def _add_grad(self, step_size, update):
        super()._add_grad(step_size,update)
        if self._trial_depth==0:
            iteration=int(self.state[self._params[0]]['n_iter'])
            assert iteration>0 and (not self.accepted_iterations or iteration>self.accepted_iterations[-1])
            self.accepted_iterations.append(iteration)
            # A no-grad VAL observation only. Never zero/backward/touch params/RNG.
            self._on_accepted(iteration)
