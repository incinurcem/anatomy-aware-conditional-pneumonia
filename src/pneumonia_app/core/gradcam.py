"""Grad-CAM implementation."""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target = target_layer
        self.acts = None
        self.grads = None
        self.h1 = target_layer.register_forward_hook(self._fwd)
        self.h2 = target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, o): self.acts = o.detach()
    def _bwd(self, _m, _gi, go): self.grads = go[0].detach()

    def __call__(self, x, cond=None):
        self.model.zero_grad()
        logit = self.model(x, cond) if cond is not None else self.model(x)
        score = logit.squeeze()
        (score if score.dim() == 0 else score[0]).backward(retain_graph=False)

        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def close(self):
        self.h1.remove()
        self.h2.remove()


def overlay_heatmap(pil, cam, alpha=0.45):
    import matplotlib.cm as cm
    arr = np.array(pil).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    heat = cm.jet(cam)[..., :3]
    out = (1 - alpha) * arr + alpha * heat
    return np.clip(out * 255, 0, 255).astype(np.uint8)