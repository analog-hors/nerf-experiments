import torch, torch.nn.functional as F

_WIDTH = 256

def _embed_point(point: torch.Tensor, freqs: int) -> torch.Tensor:
    linear = point.unsqueeze(-1)
    angles = linear * torch.exp2(torch.arange(freqs, device=point.device))
    return torch.cat((linear, torch.sin(angles), torch.cos(angles)), -1).flatten(-2)

class Model(torch.nn.Module):
    freqs: int
    l0: torch.nn.Linear
    l1: torch.nn.Linear
    l2: torch.nn.Linear
    l3: torch.nn.Linear
    l4: torch.nn.Linear
    l5: torch.nn.Linear
    l6: torch.nn.Linear
    l7: torch.nn.Linear
    l8: torch.nn.Linear

    def __init__(self, freqs: int):
        super().__init__()
        
        self.freqs = freqs

        def linear(in_features: int, out_features: int) -> torch.nn.Linear:
            layer = torch.nn.Linear(in_features, out_features)
            torch.nn.init.xavier_uniform_(layer.weight)
            return layer
        
        input_width = 3 + 3 * 2 * freqs
        self.l0 = linear(input_width, _WIDTH)
        self.l1 = linear(_WIDTH, _WIDTH)
        self.l2 = linear(_WIDTH, _WIDTH)
        self.l3 = linear(_WIDTH, _WIDTH)
        self.l4 = linear(_WIDTH, _WIDTH)
        self.l5 = linear(_WIDTH + input_width, _WIDTH)
        self.l6 = linear(_WIDTH, _WIDTH)
        self.l7 = linear(_WIDTH, _WIDTH)
        self.l8 = linear(_WIDTH, 3 + 1)

    def forward(self, point: torch.Tensor) -> torch.Tensor:
        input = _embed_point(point, self.freqs)
        x = F.relu(self.l0(input))
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(torch.cat((x, input), -1)))
        x = F.relu(self.l6(x))
        x = F.relu(self.l7(x))
        x = self.l8(x)
        return x
