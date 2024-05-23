import torch
import torch.optim as optim


class SharedAdam(optim.Adam):
    def __init__(self, params, lr=1e-3,):
        super(SharedAdam, self).__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state["p"]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()
