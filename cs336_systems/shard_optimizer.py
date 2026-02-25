import torch
import torch.distributed as dist


class ShardOptimizer(torch.optim.Optimizer):

    def __init__(self, params, optimizer_cls: type[torch.optim.Optimizer], lr, weight_decay, betas, cautious_weight_decay):

        self.optimizer_cls = optimizer_cls
        self.weight_decay = weight_decay
        self.betas = betas
        self.lr = lr
        self.cautious_weight_decay = cautious_weight_decay
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.param_to_rank = {}
        self.params = list(params)
        self.optimizer = None
        super().__init__(self.params, defaults={})

    def step(self, closure=None):

        # call the step of the wrapped optimizers and synchronize afterward
        self.optimizer.step(closure=closure)
        for param in self.params:
            rank = self.param_to_rank[param]
            dist.broadcast(param.data, src=rank)

    def add_param_group(self, param_group: dict[str, any]):
        params = param_group["params"]

        local_params = []

        for i, param in enumerate(params):

            rank = i % self.world_size
            self.param_to_rank[param] = rank

            if rank == self.rank:
                local_params.append(param)

        if self.optimizer is None:

            if self.optimizer_cls.__name__ == "AdamW":
                self.optimizer = self.optimizer_cls(
                    local_params, self.lr, self.weight_decay, self.betas, 1e-7, self.cautious_weight_decay)
            elif self.optimizer_cls.__name__ == "Muon":
                self.optimizer = self.optimizer_cls(
                    local_params, self.lr, self.weight_decay, self.betas, self.cautious_weight_decay)

            else:
                raise ValueError(
                    f"Unknown optimizer class: {self.optimizer_cls.__name__}")

        else:
            self.optimizer.add_param_group({"params": local_params})

        super().add_param_group(param_group)
