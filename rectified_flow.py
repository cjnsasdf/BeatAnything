# rectified_flow.py
import torch
import torch.nn.functional as F

class RectifiedFlow:
    """
    Adapted for osu!mania latent space [B, T, H].
    """
    def create_flow(self, x1, t, x0=None):
        """
        Creates the linear interpolation flow x_t = t * x1 + (1 - t) * x0.
        Args:
            x1: Target latent from AE [B, T, H]. e.g., [B, 125, 64]
            t: Scalar time value [B] or [B, 1, 1] broadcastable.
            x0: Initial noise [B, T, H]. If None, sampled from standard normal.
        Returns:
            x_t: Interpolated state [B, T, H].
            x0: Noise state [B, T, H].
        """
        if x0 is None:
            x0 = torch.randn_like(x1)
        # Ensure t is broadcastable to [B, T, H]
        if t.dim() == 1:
            t = t.unsqueeze(-1).unsqueeze(-1) # [B] -> [B, 1, 1]
        # x1 and x0 are [B, T, H]
        # t will broadcast: [B, 1, 1] * [B, T, H] -> [B, T, H]
        x_t = t * x1 + (1 - t) * x0
        return x_t, x0

    def mse_loss(self, v_pred, x1, x0, weights=None):
        """
        Computes the MSE loss with optional weighting.
        """
        target_v = x1 - x0
        loss = (v_pred - target_v) ** 2
        
        if weights is not None:
            # weights shape: [B, T, 1] (Broadcasting to [B, T, H])
            loss = loss * weights
            
        return loss.mean()