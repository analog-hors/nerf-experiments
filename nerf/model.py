import torch

def linear_with_xavier_uniform(in_features: int, out_features: int) -> torch.nn.Linear:
    linear = torch.nn.Linear(in_features, out_features)
    torch.nn.init.xavier_uniform_(linear.weight)
    return linear

class Model(torch.nn.Module):
    layers: torch.nn.ModuleList
    skip_interval: int

    def __init__(self, freqs: int, depth: int, width: int, skip_interval: int):
        super().__init__()
        
        self.layers = torch.nn.ModuleList()
        self.skip_interval = skip_interval

        input_width = 3 + 3 * 2 * freqs
        prev_width = input_width
        for i in range(depth):
            if i != 1 and i % self.skip_interval == 1:
                prev_width += input_width

            self.layers.append(linear_with_xavier_uniform(prev_width, width))
            prev_width = width

        self.layers.append(linear_with_xavier_uniform(prev_width, 4))

    def forward(self, point: torch.Tensor) -> torch.Tensor:
        output = point
        for i, layer in enumerate(self.layers):
            if i != 1 and i % self.skip_interval == 1:
                output = torch.cat((output, point), -1)
            output = layer(output)
            if i != len(self.layers) - 1:
                output = torch.relu(output)

        return output
