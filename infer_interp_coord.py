import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from scipy.ndimage import laplace

from src.checkpoint import load_checkpoint
from src.datasets import process_coordinates
from src.hyperparameters import args
from src.my_types import TensorFloatNx2
from src.read_images import CoordinatesExporter
from src.siren import SIREN

# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def main():
    device = "cpu"
    print(f"Running on {device} - CUDA, if available, was deactivated for this script")

    exporter = CoordinatesExporter()

    match args.derivatives_from:
        case "filters":
            coordinates = exporter.compute()["coordinates"]
        case "images":
            coordinates = exporter.from_image()["coordinates"]
        case "matfiles":
            coordinates = exporter.from_mat_file()["coordinates"]
        case _:
            raise ReferenceError(
                "Unkown option to read data. Options are `filters`, `images` or `matfiles`"
            )

    coordinates_noisy = process_coordinates(add_noise(coordinates), device)
    coordinates = process_coordinates(coordinates, device)

    model = SIREN()
    load_checkpoint(model=model, device=device)
    model.eval()

    with torch.no_grad():
        prediction = model(coordinates)
        prediction_noisy = model(coordinates_noisy)

    size = args.image_size

    x = coordinates[:, 0].reshape(size, size)
    y = coordinates[:, 1].reshape(size, size)
    prediction = prediction.reshape(size, size)

    x_noisy = coordinates_noisy[:, 0].reshape(size, size)
    y_noisy = coordinates_noisy[:, 1].reshape(size, size)
    prediction_noisy = prediction_noisy.reshape(size, size)

    # visualize_coordinates(coordinates, coordinates_noisy)
    # visualize_predictions(prediction, x, y, prediction_noisy, x_noisy, y_noisy)

    ground_truth = loadmat("jhtdb.mat")["representation"]
    visualize_predictions(prediction, x, y, ground_truth, x, y)

    check_frequencies(model)

    plot_laplacian_from_prediction(prediction)


def plot_laplacian_from_prediction(prediction) -> None:
    # ignore edges due to board effects
    pad = 2
    lapl = laplace(prediction)
    vmax = lapl[pad:-pad, pad:-pad].max()
    vmin = lapl[pad:-pad, pad:-pad].min()

    plt.imshow(lapl, cmap="gray", vmax=vmax, vmin=vmin)
    plt.show()


def check_frequencies(model) -> None:
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
            print(param.data)
            vmax = param.data.max()
            vmin = param.data.min()
            maximum = max(abs(vmax), abs(vmin))
            print("Max abs value:", maximum)


def visualize_predictions(pred, x, y, pred_noisy, x_noisy, y_noisy) -> None:
    x_max = x_noisy.max()
    x_min = x_noisy.min()
    y_max = y_noisy.max()
    y_min = y_noisy.min()

    _, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].contourf(x, y, pred, cmap="gray")
    axs[1].contourf(x_noisy, y_noisy, pred_noisy, cmap="gray")
    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
    plt.show()


def visualize_coordinates(coords: TensorFloatNx2, coords_noisy: TensorFloatNx2) -> None:
    plt.scatter(coords[:, 0], coords[:, 1], color="k")
    plt.scatter(coords_noisy[:, 0], coords_noisy[:, 1], color="r")
    plt.gca().set_aspect("equal")
    plt.show()


def add_noise(coordinates: TensorFloatNx2, varx: float = 0.007, vary: float = 0.008):
    size = int(np.sqrt(coordinates.shape[0]))

    coordinates = coordinates.reshape((size, size, 2))

    delta = 2 / size
    varx, vary = varx * delta, vary * delta

    mean = (0, 0)
    cov = [[varx, 0], [0, vary]]
    uncerts = np.random.multivariate_normal(mean, cov, (size, size))

    new_x = coordinates[:, :, 0] + uncerts[:, :, 0]
    new_y = coordinates[:, :, 1] + uncerts[:, :, 1]

    return np.stack([new_x.ravel(), new_y.ravel()], axis=1)


if __name__ == "__main__":
    main()
