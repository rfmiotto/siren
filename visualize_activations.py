import pathlib
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter

from src.siren import SIREN

torch.manual_seed(2)

init_functions = {
    "ones": torch.nn.init.ones_,
    "eye": torch.nn.init.eye_,
    "default": partial(torch.nn.init.kaiming_uniform_, a=5 ** (1 / 2)),
    "paper": None,
}


def func_histogram(instance, _, out, number=0):
    layer_name = f"{number}_{instance.__class__.__name__}"
    writer.add_histogram(layer_name, out)


for func_name, func in init_functions.items():
    path = pathlib.Path.cwd() / "tensorboard_runs" / func_name
    writer = SummaryWriter(path)

    model = SIREN(
        hidden_layers=10,
        hidden_features=200,
        first_omega=30,
        hidden_omega=30,
        custom_init_function=func,
    )

    for i, layer in enumerate(model.net.modules()):
        if not i:  # FIXME PRA QUE ISSO??
            continue
        layer.register_forward_hook(partial(func_histogram, number=(i + 1) // 2))

    inp = 2 * (torch.rand(10000, 2) - 0.5)
    writer.add_histogram("0", inp)
    res = model(inp)
