import torch, numpy as np
import nerf.model, nerf.infer

import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.widgets import Slider

from typing import Callable

WIDTH = 100
HEIGHT = 100

MODEL_PATH = "model_10k.bin"
DEVICE = torch.device("cuda:0")

model = nerf.model.load(MODEL_PATH).to(DEVICE)
model.eval()

PlotCallback = Callable[[AxesImage, list[float]], None]
SliderConfig = tuple[str, float, float, float, float]

def make_plot(update: PlotCallback, slider_configs: list[SliderConfig]):
    figure = plt.figure()

    slider_margin = 0.03 * len(slider_configs)
    ax_image = figure.add_axes((0.1, 0.1 + slider_margin, 0.8, 0.8 - slider_margin))
    image = ax_image.imshow(np.zeros((HEIGHT, WIDTH, 3)))

    state: list[float] = []
    sliders: list[Slider] = []
    for i, (name, default, min, max, step) in enumerate(slider_configs):
        ax_slider = figure.add_axes((0.1, slider_margin - 0.0015 - 0.03 * i, 0.8, 0.025))
        slider = Slider(ax_slider, name, min, max, valinit=default, valstep=step)

        def slider_update(value: float, index: int = i):
            state[index] = value
            update(image, state)
        
        state.append(default)
        sliders.append(slider)
        slider.on_changed(slider_update)

    update(image, state)
    plt.show()

def update_plot(image: AxesImage, sliders: list[float]):
    theta, pi, radius, focal, near, far, samples = sliders
    pose = pose_spherical(theta, pi, radius).to(DEVICE)
    with torch.no_grad():
        rays_o, rays_d = nerf.infer.get_rays(HEIGHT, WIDTH, focal, pose, device=DEVICE)
        output = nerf.infer.render_rays(model, rays_o, rays_d, near, far, int(samples), device=DEVICE)

    image.set_data(output.cpu())

def pose_spherical(theta: float, phi: float, radius: float):
    def translate(x: float, y: float, z: float):
        return torch.tensor([
            [ 1.0,  0.0,  0.0,    x],
            [ 0.0,  1.0,  0.0,    y],
            [ 0.0,  0.0,  1.0,    z],
            [ 0.0,  0.0,  0.0,  1.0],
        ], dtype=torch.float)

    def rotate_phi(phi: float):
        sin = np.sin(phi)
        cos = np.cos(phi)
        return torch.tensor([
            [ 1.0,  0.0,  0.0,  0.0],
            [ 0.0,  cos, -sin,  0.0],
            [ 0.0,  sin,  cos,  0.0],
            [ 0.0,  0.0,  0.0,  1.0],
        ], dtype=torch.float)

    def rotate_theta(theta: float):
        sin = np.sin(theta)
        cos = np.cos(theta)
        return torch.tensor([
            [ cos,  0.0, -sin,  0.0],
            [ 0.0,  1.0,  0.0,  0.0],
            [ sin,  0.0,  cos,  0.0],
            [ 0.0,  0.0,  0.0,  1.0],
        ], dtype=torch.float)

    OPENGL_TO_BLENDER = torch.tensor([
        [-1.0,  0.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0,  0.0],
        [ 0.0,  1.0,  0.0,  0.0],
        [ 0.0,  0.0,  0.0,  1.0],
    ], dtype=torch.float)

    c2w = translate(0.0, 0.0, radius)
    c2w = rotate_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rotate_theta(theta / 180.0 * np.pi) @ c2w
    c2w = OPENGL_TO_BLENDER @ c2w
    return c2w

make_plot(update_plot, [
    ("Theta", 100, 0.0, 360.0, 0.1),
    ("Phi", -30.0, -90.0, 90.0, 0.1),
    ("Radius", 9.0, 0.0, 15.0, 0.1),
    ("Focal", 138.88888549804688, 1.0, 500.0, 0.1),
    ("Near", 5.0, 0.0, 20.0, 0.1),
    ("Far", 11.0, 0.0, 20.0, 0.1),
    ("Samples", 64.0, 1.0, 256.0, 1.0),
])
