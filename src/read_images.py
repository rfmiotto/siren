from typing import Protocol, TypedDict
from typing_extensions import NotRequired
import numpy as np
from PIL import Image
from PIL.Image import Image as ImageType
from scipy.ndimage import laplace, sobel
import scipy

from src.hyperparameters import args
from src.dtos import TrainingData
from src.my_types import ArrayFloat32NxN, ArrayBoolNxN


class ExporterReturn(TypedDict):
    representation: NotRequired[ArrayFloat32NxN]
    gradient_x: NotRequired[ArrayFloat32NxN]
    gradient_y: NotRequired[ArrayFloat32NxN]
    laplacian: NotRequired[ArrayFloat32NxN]
    mask: NotRequired[ArrayBoolNxN]


class TrainingDataExporter(Protocol):
    def from_mat_file(self) -> ExporterReturn:
        """
        Read derivatives directly from mat files (e.g. from EMD).
        Inside the mat file, data must be stored using the keys `gradient_x`,
        `gradient_y`, `laplacian` and `mask`
        """

    def from_image(self) -> ExporterReturn:
        """Read derivatives and mask directly from images"""

    def compute(self) -> ExporterReturn:
        """Compute derivatives using filters (Sobel, Laplace) and generate mask"""


class GradientsExporter:
    def from_mat_file(self) -> ExporterReturn:
        return {
            "gradient_x": read_mat(args.gradient_x_image_name)["gradient_x"],
            "gradient_y": read_mat(args.gradient_y_image_name)["gradient_y"],
        }

    def from_image(self) -> ExporterReturn:
        return {
            "gradient_x": convert_to_float(read_image(args.gradient_x_image_name)),
            "gradient_y": convert_to_float(read_image(args.gradient_y_image_name)),
        }

    def compute(self) -> ExporterReturn:
        image = convert_to_float(read_image(args.original_image_name))
        # sobel returns bad values with 2D input arrays, so we make them 3D.
        image = np.expand_dims(image, axis=0)
        return {
            "gradient_x": sobel(image, axis=1)[0],
            "gradient_y": sobel(image, axis=2)[0],
        }


class LaplacianExporter:
    def from_mat_file(self) -> ExporterReturn:
        return {
            "laplacian": read_mat(args.laplacian_image_name)["laplacian"],
        }

    def from_image(self) -> ExporterReturn:
        return {
            "laplacian": convert_to_float(read_image(args.laplacian_image_name)),
        }

    def compute(self) -> ExporterReturn:
        image = convert_to_float(read_image(args.original_image_name))
        return {
            "laplacian": laplace(image),
        }


class MaskExporter:
    def from_mat_file(self) -> ExporterReturn:
        try:
            mask = read_mat(args.mask_image_name)["mask"]
        except FileNotFoundError:
            mask = self.compute()["mask"]

        return {
            "mask": mask,
        }

    def from_image(self) -> ExporterReturn:
        try:
            mask = convert_to_bool(read_image(args.mask_image_name))
        except FileNotFoundError:
            mask = self.compute()["mask"]

        return {
            "mask": mask,
        }

    def compute(self) -> ExporterReturn:
        shape = (args.image_size, args.image_size)
        return {
            "mask": np.zeros(shape).astype(bool),
        }


class RepresentationExporter:
    def from_mat_file(self) -> ExporterReturn:
        try:
            representation = read_mat(args.mask_image_name)["representation"]
        except FileNotFoundError:
            representation = None

        return {
            "representation": representation,
        }

    def from_image(self) -> ExporterReturn:
        try:
            representation = convert_to_float(read_image(args.original_image_name))
        except FileNotFoundError:
            representation = None

        return {
            "representation": representation,
        }

    def compute(self) -> ExporterReturn:
        return self.from_image()


DERIVATIVE_EXPORTER_OPTIONS = {
    "gradients": GradientsExporter(),
    "laplacian": LaplacianExporter(),
}


class Exporters(TypedDict):
    derivatives: TrainingDataExporter
    mask: TrainingDataExporter
    representation: TrainingDataExporter


def get_exporters() -> Exporters:
    return {
        "derivatives": DERIVATIVE_EXPORTER_OPTIONS[args.fit],
        "mask": MaskExporter(),
        "representation": RepresentationExporter(),
    }


def do_export(exporters: Exporters) -> TrainingData:
    if args.derivatives_from == "filters":
        derivatives = exporters["derivatives"].compute()
        mask = exporters["mask"].compute()
        representation = exporters["representation"].compute()
    elif args.derivatives_from == "images":
        derivatives = exporters["derivatives"].from_image()
        mask = exporters["mask"].from_image()
        representation = exporters["representation"].from_image()
    elif args.derivatives_from == "matfiles":
        derivatives = exporters["derivatives"].from_mat_file()
        mask = exporters["mask"].from_mat_file()
        representation = exporters["representation"].from_mat_file()
    else:
        raise ReferenceError(
            "Unkown option to read data. Options are `filters`, `images` or `matfiles`"
        )

    training_data = dict(derivatives)
    training_data.update(mask)
    training_data.update(representation)

    return training_data


def get_training_data() -> TrainingData:
    exporters = get_exporters()
    return do_export(exporters)


def read_image(file: str) -> ImageType:
    if file is None:
        raise FileNotFoundError

    image = Image.open(file, "r").convert("L")

    check_image_is_square(image)

    resized_image = image.resize((args.image_size, args.image_size))

    return resized_image


def read_mat(file: str) -> ArrayFloat32NxN:
    if file is None:
        raise FileNotFoundError

    return scipy.io.loadmat(file)


def convert_to_float(image: ImageType) -> ArrayFloat32NxN:
    return np.float32(np.array(image) / 255.0)


def convert_to_bool(image: ImageType) -> ArrayBoolNxN:
    return np.array(image).astype(bool)


def check_image_is_square(image: ImageType) -> None:
    width, height = image.size
    is_square_image = width == height
    if not is_square_image:
        raise TypeError("Images must be squared.")
