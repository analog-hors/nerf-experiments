import time
import torch, numpy as np
import nerf.model, nerf.infer
from PIL import Image

DEVICE = torch.device("cuda:0")
ITERATIONS = 10_000
LOG_INTERVAL = 25

NEAR = 5.0
FAR = 11.0
TRAINING_SAMPLES = 64
BATCH_SIZE = 1000

SAMPLE_SCALE = 0.1
SAMPLE_POSE = 50
SAMPLE_SAMPLES = 64

def load_numpy_dataset(path: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, float]:
    data = np.load(path)
    images = torch.tensor(data["images"], device=device)
    poses = torch.tensor(data["poses"], device=device)
    focal = float(data["focal"])
    return images, poses, focal

images, poses, focal = load_numpy_dataset("datasets/horse1k.npz", device=DEVICE)
height, width = images.shape[1:3]

model = nerf.model.Model(6).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr = 0.0005)
loss_fn = torch.nn.MSELoss()

model.train()
running_start = time.time()
running_loss = 0
for iteration in range(ITERATIONS):
    batch_i = torch.randint(0, images.shape[0], (BATCH_SIZE,), device=DEVICE)
    batch_y = torch.randint(0, images.shape[1], (BATCH_SIZE,), device=DEVICE)
    batch_x = torch.randint(0, images.shape[2], (BATCH_SIZE,), device=DEVICE)

    rays_o, rays_d = nerf.infer.get_rays(
        batch_x,
        batch_y,
        width,
        height,
        focal,
        poses[batch_i],
    )
    t_vals = nerf.infer.stratified_samples(
        NEAR,
        FAR,
        TRAINING_SAMPLES,
        (BATCH_SIZE,),
        device=DEVICE,
    )
    output = nerf.infer.render_rays(
        model,
        rays_o,
        rays_d,
        t_vals,
    )

    loss = loss_fn(output, images[batch_i, batch_y, batch_x])
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
            sample_width = int(width * SAMPLE_SCALE)
            sample_height = int(height * SAMPLE_SCALE)
            sample_focal = focal * SAMPLE_SCALE
            sample_x, sample_y = torch.meshgrid(
                torch.arange(sample_width, device=DEVICE),
                torch.arange(sample_height, device=DEVICE),
                indexing="xy",
            )
            sample_rays_o, sample_rays_d = nerf.infer.get_rays(
                sample_x,
                sample_y,
                sample_width,
                sample_height,
                sample_focal,
                poses[SAMPLE_POSE],
            )
            sample_output = nerf.infer.render_rays(
                model,
                sample_rays_o,
                sample_rays_d,
                torch.linspace(NEAR, FAR, SAMPLE_SAMPLES, device=DEVICE),
            )
            sample_image_data = (sample_output.cpu() * 255).byte().numpy().tobytes()
            Image.frombytes("RGB", (sample_width, sample_height), sample_image_data).save(f"inferred.png")
        model.train()

        running_start = now
        running_loss = 0

nerf.model.save(model, "model.bin")
