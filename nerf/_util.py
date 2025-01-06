import torch

CPU = torch.device("cpu")

def exclusive_cumprod(input: torch.Tensor, dim: int) -> torch.Tensor:
    assert dim in range(-len(input.shape) + 1, len(input.shape))
    dim = dim if dim >= 0 else dim + len(input.shape)
    ones = torch.ones((*input.shape[:dim], 1, *input.shape[dim + 1:]), device=input.device)
    rest = torch.cumprod(torch.narrow(input, dim, 0, input.shape[dim] - 1), dim)
    return torch.cat((ones, rest), dim)

def chunked_inference(model: torch.nn.Module, batch: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return torch.cat([model(batch[i:i + chunk_size]) for i in range(0, batch.shape[0], chunk_size)])
