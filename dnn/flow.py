import torch
import torch.nn.functional as F


class FlowMatching:
    """
        Conditional Flow Matching abstraction
    """
    def __init__(self, dnn, t_sampler="uniform"):
        self.dnn = dnn
        self.t_sampler = t_sampler
    
    def sample_t(self, B, device):
        assert self.t_sampler in ["uniform", "logit_normal"], f"specified time sampler : {self.t_sampler} is not implemented."
        if self.t_sampler == "uniform":
            t = torch.rand(B, device=device)
        elif self.t_sampler == "logit_normal":
            # concentrates more near 0/1 depending on sigma
            sigma = 1.0
            t = torch.randn(B, device=device)*sigma
            t = torch.sigmoid(t)
        return t

    def interp(self, z, eps, t):
        t = t.view(t.shape[0], 1, 1, 1, 1) # (B, T, C, H, W)
        x_t = t * z + (1 - t) * eps # simple condOT transport for now
        v_target = z - eps
        return x_t, v_target
    
    def training_step(self, z, c):
        """
        (z, c) ~ p_data(z,c) z in (B, T, C, H, W) c in (B)
        """

        device = z.device
        B = z.shape[0]

        eps = torch.randn_like(z)
        t = self.sample_t(B, device)
        x_t, u_target = self.interp(z, eps, t)

        u_theta = self.dnn(x_t, t, c)

        loss = ((u_theta - u_target)**2).mean()
        return loss
    

class FlowSampler:
    """
        Flow ODE sampler.

        args:
            sampler: ["euler", "heun"],
            u_theta : flow field,
    """

    def __init__(self, u_theta, sampler="euler"):
        assert sampler in ["euler", "heun"]
        self.sampler = sampler
        self.u_theta = u_theta

    @staticmethod
    def to_tensor(batch_size, t, device):
        t = torch.ones(batch_size, device=device, dtype=torch.float32) * t
        return t

    def euler_step(self, u_theta, x_t, t, dt, c, cfg_scale):
        
        t = self.to_tensor(x_t.shape[0], t, device=x_t.device)
        if cfg_scale == 1.0:
            x_t = x_t + dt * u_theta.forward(x_t, t, c)
        else:
            x_t = x_t + dt * u_theta.forward_with_cfg(x_t, t, c, cfg_scale=cfg_scale)

        return x_t
    
    def heun_step(self, u_theta, x_t, t, dt, c, cfg_scale, t_end=1.0):    
        t_next_scalar = min(t+dt, t_end)
        dt_eff = t_next_scalar - t

        t_current = self.to_tensor(x_t.shape[0], t=t, device=x_t.device)
        t_next = self.to_tensor(x_t.shape[0], t=t_next_scalar, device=x_t.device)

        if cfg_scale == 1.0:
            field_xt = u_theta.forward(x_t, t_current, c)
            x_next = x_t + dt_eff * field_xt
            field_next = u_theta.forward(x_next, t_next, c)
        else:
            field_xt = u_theta.forward_with_cfg(x_t, t_current, c, cfg_scale=cfg_scale)
            x_next = x_t + dt_eff * field_xt
            field_next = u_theta.forward_with_cfg(x_next, t_next, c, cfg_scale=cfg_scale)

        net_field = (field_xt + field_next) * 0.5
        x_t = x_t + dt_eff * net_field
        return x_t
    
    @torch.no_grad()
    def sample(self, x, c, steps=30, cfg_scale=3.0, t_end=1.0):

        t = 0.0
        dt = t_end / steps
        for i in range(steps):
            last = (i == steps - 1)

            if self.sampler == "heun" and (not last):
                x = self.heun_step(self.u_theta, x, t, dt, c, cfg_scale)
            else:
                x = self.euler_step(self.u_theta, x, t, dt, c, cfg_scale)
            
            t = (i+1) * t_end / steps
        return x
        # 0 0.33 0.66 1.0
    