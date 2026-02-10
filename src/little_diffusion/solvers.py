import torch
from little_diffusion.core import ConditionalProbabilityPath, Trainer, Sampleable

class LinearProbabilityPath(ConditionalProbabilityPath):
    def __init__(self):
        super().__init__(p_simple=None, p_data=None)
    
    def sample_conditioning_variable(self, num_samples: int):
        raise NotImplementedError("For a single image dataset, you shouldn't use this.")
    
    def sample_conditional_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        while t.dim() < x1.dim():
            t = t.unsqueeze(-1)
        return (1 - t) * x0 + t * x1

    def conditional_vector_field(self, x_t: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        epsilon = 1e-5
        t_safe = torch.clamp(t, max=1.0 - epsilon)
        while t_safe.dim() < x1.dim():
            t_safe = t_safe.unsqueeze(-1)
        return (x1 - x_t) / (1 - t_safe)
    
    def conditional_score(self, x_t: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError("For a single image dataset, you shouldn't use this.")

class FlowMatchingTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, path: ConditionalProbabilityPath):
        super().__init__(model)
        self.path = path

    def get_train_loss(self, target: torch.Tensor, **kwargs):
        # target: (batch, 3, H, W)
        bs = target.shape[0]
        device = target.device

        t = torch.rand(bs, 1, device=device)
        x0 = torch.randn_like(target)

        xt = self.path.sample_conditional_path(x0, target, t)

        ut = self.path.conditional_vector_field(xt, target, t)
        ut_pred = self.model(xt, t)

        loss = torch.nn.functional.mse_loss(ut_pred, ut)
        return loss
        



    