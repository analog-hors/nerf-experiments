import torch
import nerf._util

def get_rays(
    x: torch.Tensor,
    y: torch.Tensor,
    width: int,
    height: int,
    focal: float,
    c2w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.shape == y.shape
    assert len(c2w.shape) >= 2 and c2w.shape[-2:] == (4, 4)
    assert nerf._util.can_broadcast(x.shape[:-1], c2w.shape[:-2])
    _device = nerf._util.get_device_if_same(y, x, c2w)

    # Create camera-relative direction vectors with shape (..., 3)
    dirs = torch.stack((
        (x - width / 2) / focal,
        -(y - height / 2) / focal,
        -torch.ones_like(x),
    ), -1)
    dirs /= torch.linalg.vector_norm(dirs, 2, -1).unsqueeze(-1)

    # Transform direction vectors using camera-to-world matrix
    rays_d = torch.sum(dirs[..., None, :] * c2w[..., :3, :3], -1)
    rays_o = torch.broadcast_to(c2w[..., :3, -1], rays_d.shape)

    return rays_o, rays_d

def stratified_samples(
    near: float,
    far: float,
    samples: int,
    batch_dims: tuple[int, ...],
    device: torch.device = nerf._util.CPU,
):
    t_vals = torch.rand((*batch_dims, samples), device=device) * ((far - near) / samples)
    t_vals += torch.linspace(near, far, samples, device=device)
    return t_vals

def render_rays(
    model: torch.nn.Module,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    t_vals: torch.Tensor,
) -> torch.Tensor:
    assert len(rays_o.shape) >= 1 and rays_o.shape[-1] == 3
    assert rays_d.shape == rays_o.shape
    assert nerf._util.can_broadcast(t_vals.shape[:-1], rays_o.shape[:-1])
    _device = nerf._util.get_device_if_same(rays_o, rays_d, t_vals)

    # Compute sample points and query model at each point
    points = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * t_vals.unsqueeze(-1)
    input_batch = points.reshape((-1, 3))
    network_output = nerf._util.chunked_inference(model, input_batch, 1024).reshape((*points.shape[:-1], 4))

    # Extract and process density and rgb outputs
    density = torch.relu(network_output[..., 3])
    rgb = torch.sigmoid(network_output[..., :3])

    # Do volume rendering to produce weights
    dists = torch.cat((t_vals[..., 1:] - t_vals[..., :-1], torch.full_like(t_vals[..., :1], 1e10)), -1)
    alpha = 1.0 - torch.exp(-density * dists)
    weights = alpha * nerf._util.exclusive_cumprod(1.0 - alpha, -1)
    
    # Apply weights to rgb to produce final colors
    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, -2) 

    return rgb_map
