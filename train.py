import time
import torch, numpy as np
import nerf.model, nerf.infer
from PIL import Image

DEVICE = torch.device("cuda:0")
ITERATIONS = 1000
LOG_INTERVAL = 25

NEAR = 2.0
FAR = 6.0
TRAINING_SAMPLES = 64

SAMPLE_POSE = 0
SAMPLE_SAMPLES = 64

def load_numpy_dataset(path: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    data = np.load(path)
    images = torch.tensor(data["images"], device=device)
    poses = torch.tensor(data["poses"], device=device)
    focal = float(data["focal"])
    return images, poses, focal

images, poses, focal = load_numpy_dataset("datasets/tiny_nerf_data.npz", device=DEVICE)
height, width = images.shape[1:3]

rays_o, rays_d = nerf.infer.get_rays(height, width, focal, poses, device=DEVICE)

model = nerf.model.Model(6, 8, 256, 4).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr = 0.0005)
loss_fn = torch.nn.MSELoss()

model.train()
running_start = time.time()
running_loss = 0
for iteration in range(ITERATIONS):
    index = np.random.randint(images.shape[0])

    output = nerf.infer.render_rays(
        model,
        rays_o[index],
        rays_d[index],
        NEAR,
        FAR,
        TRAINING_SAMPLES,
        randomize=True,
        device=DEVICE,
    )

    loss = loss_fn(output, images[index])
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
            sample_rays_o, sample_rays_d = nerf.infer.get_rays(
                height,
                width,
                focal,
                poses[SAMPLE_POSE],
                device=DEVICE,
            )
            output = nerf.infer.render_rays(
                model,
                sample_rays_o,
                sample_rays_d,
                NEAR,
                FAR,
                SAMPLE_SAMPLES,
                device=DEVICE,
            )
            Image.frombytes("RGB", (width, height), (output.cpu() * 255).byte().numpy().tobytes()).save(f"inferred.png")
        model.train()

        running_start = now
        running_loss = 0

torch.save(model.state_dict(), "model.bin")
