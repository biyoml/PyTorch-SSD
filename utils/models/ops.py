import torch


def get_output_shapes(model, input_size):
    """
    Returns:
        : A list of [channesl, size, size]. Shape: [num_stages, [3]].
    """
    inp = torch.randn([2, 3, input_size, input_size])
    with torch.no_grad():
        out = model(inp)
    return [o.shape[1:] for o in out]
