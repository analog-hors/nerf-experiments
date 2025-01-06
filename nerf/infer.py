import torch
import nerf._util

def embed_points(points: torch.Tensor, freqs: int) -> torch.Tensor:
    linear = points.unsqueeze(-1)
    angles = linear * torch.exp2(torch.arange(freqs, device=points.device))
    return torch.cat((linear, torch.sin(angles), torch.cos(angles)), -1).flatten(-2)

def get_rays(height: int, width: int, focal: float, c2w: torch.Tensor, device: torch.device = nerf._util.CPU):
    # Create y and x tensors of shape (height, width, 1)
    y, x = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )

    # Create camera-relative direction vectors with shape (height, width, 3)
    dirs = torch.dstack((
        (x - width / 2) / focal,
        -(y - height / 2) / focal,
        -torch.ones_like(x, device=device),
    ))
    dirs /= torch.linalg.vector_norm(dirs, 2, -1).unsqueeze(-1)

    # Transform direction vectors using camera-to-world matrix
    rays_d = torch.sum(dirs.unsqueeze(-2) * c2w[:3, :3], 3)
    rays_o = torch.broadcast_to(c2w[:3, -1], rays_d.shape)

    return rays_o, rays_d

def render_rays(
    model: torch.nn.Module,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    samples: int,
    randomize: bool = False,
    device: torch.device = nerf._util.CPU,
) -> torch.Tensor:

    z_vals = torch.linspace(near, far, samples, device=device)
    if randomize:
        z_vals = z_vals + torch.rand((*rays_o.shape[:2], samples), device=device) * ((far - near) / samples)
    
    points = rays_o.unsqueeze(2) + rays_d.unsqueeze(2) * z_vals.unsqueeze(-1)
    
    input_batch = embed_points(points.reshape((-1, 3)), 6)
    network_output = nerf._util.chunked_inference(model, input_batch, 1024).reshape((*points.shape[:3], 4))

    sigma_a = torch.relu(network_output[..., 3])
    rgb = torch.sigmoid(network_output[..., :3])

    dists = torch.cat((z_vals[..., 1:] - z_vals[..., :-1], torch.full_like(z_vals[..., :1], 1e10)), -1)
    alpha = 1.0 - torch.exp(-sigma_a * dists)  
    weights = alpha * nerf._util.exclusive_cumprod(1.0 - alpha + 1e-10, -1)
    
    rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, -2) 

    return rgb_map
