import scipy
import torch

from src.checkpoint import load_checkpoint
from src.datasets import process_coordinates
from src.hyperparameters import args
from src.my_types import TensorFloatNx1
from src.read_images import CoordinatesExporter
from src.siren import SIREN

# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

CASES = {
    "noise_000.pth.tar": "siren_0.00_noise.mat",
    "noise_001.pth.tar": "siren_0.01_noise.mat",
    "noise_002.pth.tar": "siren_0.02_noise.mat",
    "noise_003.pth.tar": "siren_0.03_noise.mat",
    "noise_004.pth.tar": "siren_0.04_noise.mat",
    "noise_005.pth.tar": "siren_0.05_noise.mat",
    "noise_006.pth.tar": "siren_0.06_noise.mat",
    "noise_007.pth.tar": "siren_0.07_noise.mat",
    "noise_008.pth.tar": "siren_0.08_noise.mat",
    "noise_009.pth.tar": "siren_0.09_noise.mat",
    "noise_010.pth.tar": "siren_0.10_noise.mat",
}


def main(checkpoint_name: str, out_filename: str):
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

    coordinates = process_coordinates(coordinates, device)

    model = SIREN()
    load_checkpoint(model=model, device=device, filename=checkpoint_name)
    model.eval()

    with torch.no_grad():
        prediction = model(coordinates)

    save_result(out_filename, prediction)


def save_result(filename: str, prediction: TensorFloatNx1) -> None:
    size = args.image_size

    mat_dict = {
        "prediction": prediction.numpy().reshape(size, size),
    }

    scipy.io.savemat(filename, mat_dict)


if __name__ == "__main__":
    for checkpoint, outfile in CASES.items():
        main(checkpoint, outfile)
