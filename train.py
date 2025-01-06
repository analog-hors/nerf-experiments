import time
import torch, numpy as np
import nerf.model, nerf.infer
from PIL import Image

DEVICE = torch.device("cuda:0")
ITERATIONS = 1000
LOG_INTERVAL = 25

def load_numpy_dataset(path: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    data = np.load(path)
    images = torch.tensor(data["images"], device=device)
    poses = torch.tensor(data["poses"], device=device)
    focal = float(data["focal"])
    return images, poses, focal

images, poses, focal = load_numpy_dataset("datasets/tiny_nerf_data.npz", device=DEVICE)
height, width = images.shape[1:3]

model = nerf.model.Model(6, 8, 256, 4).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr = 0.0005)
loss_fn = torch.nn.MSELoss()

model.train()
running_start = time.time()
running_loss = 0
for iteration in range(ITERATIONS):
    index = np.random.randint(images.shape[0])
    target = images[index]
    pose = poses[index]
    
    rays_o, rays_d = nerf.infer.get_rays(height, width, focal, pose, device=DEVICE)
    output = nerf.infer.render_rays(model, rays_o, rays_d, 2.0, 6.0, 64, randomize=True, device=DEVICE)
    
    loss = loss_fn(output, target)
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    running_loss += loss.item()
    if (iteration + 1) % LOG_INTERVAL == 0:
        now = time.time()
        loss = running_loss / LOG_INTERVAL
        iters_per_sec = LOG_INTERVAL / (now - running_start)
        print(f"[{iteration + 1}/{ITERATIONS}] loss: {loss}, {iters_per_sec:.2f} iters/sec", flush=True)

        model.eval()
        with torch.no_grad():
            rays_o, rays_d = nerf.infer.get_rays(height, width, focal, poses[101], device=DEVICE)
            output = nerf.infer.render_rays(model, rays_o, rays_d, 2.0, 6.0, 64, device=DEVICE)
            Image.frombytes("RGB", (width, height), (output.cpu() * 255).byte().numpy().tobytes()).save(f"inferred.png")
        model.train()

        running_start = now
        running_loss = 0

torch.save(model.state_dict(), "model.bin")
