import torch
import torch.distributed as dist


class DDPParameter(torch.nn.Module):

    def __init__(self, module: torch.nn.Module, sharded=False, use_muon=False):

        super().__init__()

        self.module = module
        self.handles = []
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.sharded = sharded
        self.use_muon = use_muon
        self.param_to_owner = self._construct_param_to_owner()
        for param in module.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    self._hook_function())

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        for param in self.module.parameters():
            if param.requires_grad:

                if self.sharded:
                    owner = self.param_to_owner[param]
                    if owner == self.rank:
                        param.grad.div_(self.world_size)

                    elif self.sharded:
                        param.grad = None

                else:
                    param.grad.div_(self.world_size)

    def _hook_function(self):
        def hook(param):
            if self.sharded:
                owner = self.param_to_owner[param]
                handle = dist.reduce(param.grad, dst=owner,
                                     op=dist.ReduceOp.SUM, async_op=True)
            else:
                handle = dist.all_reduce(
                    param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)

        return hook

    def _construct_param_to_owner(self):

        if self.sharded:

            param_to_owner = {}

            if self.use_muon:

                parameters_adam = []
                parameters_muon = []

                for name, param in self.module.named_parameters():
                    if param.ndim >= 2 and name not in ("output.W", "embedding.E"):
                        parameters_muon.append(param)
                    else:
                        parameters_adam.append(param)

                for params in [parameters_adam, parameters_muon]:
                    for i, param in enumerate(params):
                        rank = i % self.world_size
                        param_to_owner[param] = rank

            else:
                for i, param in enumerate(list(self.module.parameters())):
                    rank = i % self.world_size
                    param_to_owner[param] = rank

        else:

            param_to_owner = None

        return param_to_owner
