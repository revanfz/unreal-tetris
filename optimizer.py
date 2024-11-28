from __future__ import division

import torch
import torch.optim as optim

class SharedRMSprop(optim.RMSprop):
    """Implements RMSprop algorithm with shared states."""

    def __init__(
        self,
        params,
        lr=7e-4,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
    ):
        super(SharedRMSprop, self).__init__(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["grad_avg"] = p.data.new().resize_as_(p.data).zero_()
                state["square_avg"] = p.data.new().resize_as_(p.data).zero_()
                state["momentum_buffer"] = p.data.new().resize_as_(p.data).zero_()

                state["square_avg"].share_memory_()
                state["step"].share_memory_()
                state["grad_avg"].share_memory_()
                state["momentum_buffer"].share_memory_()

class SharedAdam(optim.Adam):
    """Implements Adam algorithm with shared states."""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=eps, amsgrad=amsgrad)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = p.data.new().resize_as_(p.data).zero_()
                state["exp_avg_sq"] = p.data.new().resize_as_(p.data).zero_()
                state["max_exp_avg_sq"] = p.data.new().resize_as_(p.data).zero_()

                state["step"].share_memory_()
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()
                state["max_exp_avg_sq"].share_memory_()