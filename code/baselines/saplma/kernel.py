"""Named definitions exported verbatim; provenance in KERNEL_BINDINGS.json."""
import math
import torch

class SAPLMA(torch.nn.Module):
    """Original widths/ReLU; final sigmoid is applied during prediction.

Keras recovers pre-sigmoid logits for stable binary cross entropy; using
BCEWithLogitsLoss here expresses the same sigmoid+BCE training equation.
"""
    def __init__(self, width):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(width, 256), torch.nn.ReLU(),
                                          torch.nn.Linear(256, 128), torch.nn.ReLU(),
                                          torch.nn.Linear(128, 64), torch.nn.ReLU(),
                                          torch.nn.Linear(64, 1))
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.layers(x)

class KerasAdam(torch.optim.Optimizer):
    """Keras Adam epsilon_hat equation, not Torch's corrected-variance epsilon."""
    def __init__(self, params):
        super().__init__(params, {'lr': .001, 'beta1': .9, 'beta2': .999, 'epsilon_hat': 1e-7})

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is None
        for group in self.param_groups:
            b1, b2 = group['beta1'], group['beta2']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                assert bool(torch.isfinite(grad).all())
                state = self.state[parameter]
                if not state:
                    state.update(t=0, m=torch.zeros_like(parameter), v=torch.zeros_like(parameter))
                state['t'] += 1
                state['m'].mul_(b1).add_(grad, alpha=1 - b1)
                state['v'].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                alpha = group['lr'] * math.sqrt(1 - b2**state['t']) / (1 - b1**state['t'])
                parameter.addcdiv_(state['m'], state['v'].sqrt().add_(group['epsilon_hat']), value=-alpha)
