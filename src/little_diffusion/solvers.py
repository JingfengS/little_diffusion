import torch
from little_diffusion.core import ConditionalProbabilityPath, Trainer

class LinearProbabilityPath(ConditionalProbabilityPath):
    def __init__(self):
        super().__init__(p_simple=None, p_data=None)
    
    def sample_conditioning_variable(self, num_samples: int):
        raise NotImplementedError("For a single image dataset, you shouldn't use this.")
    
    def sample_conditional_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        while t.dim() < x1.dim():
            t = t.unsqueeze(-1)
        return (1 - t) * x0 + t * x1

    def conditional_vector_field(self, x_0: torch.Tensor, x1: torch.Tensor):
        return x1 - x_0
    
    def conditional_score(self, x_t: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        raise NotImplementedError("For a single image dataset, you shouldn't use this.")

class FlowMatchingTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, path: ConditionalProbabilityPath):
        super().__init__(model)
        self.path = path

    def get_train_loss(self, target: torch.Tensor, labels: torch.Tensor = None, mask: torch.Tensor = None, **kwargs):
        # target: (batch, 3, H, W)
        bs = target.shape[0]
        device = target.device

        # 1. 采样时间 t (0.0 ~ 1.0)
        t = torch.rand(bs, device=device)
        x0 = torch.randn_like(target)

        # 2. 调用 Path 计算 xt 和目标速度 ut
        xt = self.path.sample_conditional_path(x0, target, t)
        ut = self.path.conditional_vector_field(x0, target)

        # 3. 适配 DiT：将 float 的 t 映射为 0~1000 的整数
        t_model = (t * 1000).long()

        # 4. 前向传播 (支持 CFG Labels)
        if labels is not None:
            ut_pred = self.model(xt, t_model, labels, mask=mask)
        else:
            ut_pred = self.model(xt, t_model, mask=mask)
        if mask is not None:
            # ut_pred: (BS, 4, H, W)
            # mask: (BS, 1, H, W) -> (BS, 4, H, W)
            diff = ut_pred - ut
            loss = (diff ** 2) * mask
            return loss.mean()
        else:
            return torch.nn.functional.mse_loss(ut_pred, ut)



    