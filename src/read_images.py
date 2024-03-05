from typing import Protocol, TypedDict
from typing_extensions import NotRequired
import numpy as np
from PIL import Image
from PIL.Image import Image as ImageType
from scipy.ndimage import laplace, sobel
import scipy

from src.hyperparameters import args
from src.dtos import TrainingData
from src.my_types import ArrayFloat32NxN, ArrayBoolNxN, ArrayUintNx2


class ExporterReturn(TypedDict):
    coordinates: NotRequired[ArrayFloat32NxN]
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
        `gradient_y`, `laplacian` and `mask`. For the coordinates, the mat file
        must contain `coords_x` and `coords_y` as column headings
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


class CoordinatesExporter:
    def from_mat_file(self) -> ExporterReturn:
        try:
            coordinates_mat = read_mat(args.coordinates_file_name)
        except FileNotFoundError:
            return {"coordinates": self.compute()["coordinates"]}

        coord_x = coordinates_mat["coord_x"]
        coord_y = coordinates_mat["coord_y"]

        return {"coordinates": np.stack([coord_x.ravel(), coord_y.ravel()], axis=-1)}

    def from_image(self) -> ExporterReturn:
        coords_idx = self.pixel_coords_to_cartesian(
            self.generate_pixel_coords(args.image_size)
        )

        coords_normalized = 2.0 * ((coords_idx / (args.image_size - 1)) - 0.5)

        return {"coordinates": coords_normalized}

    def compute(self) -> ExporterReturn:
        coords_idx = self.generate_pixel_coords(args.image_size)

        coords_normalized = 2.0 * ((coords_idx / (args.image_size - 1)) - 0.5)

        return {"coordinates": coords_normalized}

    def generate_pixel_coords(self, num_side_points: int) -> ArrayUintNx2:
        """Generate regular grid of 2D coordinates on [0, num_side_points] x [0, num_side_points].

        Parameters
        ----------
        num_side_points : int
            Number of points per dimension.

        Returns
        -------
        pixel_coords : np.ndarray
            Array of row and column coordinates (indices) of shape
            `(num_side_points ** 2, 2)`.
        """
        rows, cols = np.meshgrid(
            range(num_side_points), range(num_side_points), indexing="ij"
        )
        pixel_coords = np.stack([rows.ravel(), cols.ravel()], axis=-1)

        return pixel_coords

    def pixel_coords_to_cartesian(self, pixel_coords: ArrayUintNx2) -> ArrayUintNx2:
        """Convert regular grid of 2D coordinates from pixel domain to Cartesian.
        This process shifts the origin center from the top-left image to the bottom
        left.

        Parameters
        ----------
        pixel_coords : np.ndarray
            Array of row and column coordinates (indices) of shape
            `(num_side_points ** 2, 2)`.

        Returns
        -------
        pixel_coords_cart : np.ndarray
            Array of row and column coordinates (indices) of shape
            `(num_side_points ** 2, 2)`.
        """
        num_side_points = int(np.sqrt(pixel_coords.shape[0]))

        x = pixel_coords.reshape(num_side_points, num_side_points, 2)[:, :, 1]
        y = pixel_coords.reshape(num_side_points, num_side_points, 2)[:, :, 0]

        y = np.flip(y)

        return np.stack([x.ravel(), y.ravel()], axis=1)


DERIVATIVE_EXPORTER_OPTIONS = {
    "gradients": GradientsExporter(),
    "laplacian": LaplacianExporter(),
}


class Exporters(TypedDict):
    derivatives: TrainingDataExporter
    mask: TrainingDataExporter
    representation: TrainingDataExporter
    coordinates: TrainingDataExporter


def get_exporters() -> Exporters:
    return {
        "derivatives": DERIVATIVE_EXPORTER_OPTIONS[args.fit],
        "mask": MaskExporter(),
        "representation": RepresentationExporter(),
        "coordinates": CoordinatesExporter(),
    }


def do_export(exporters: Exporters) -> TrainingData:
    if args.derivatives_from == "filters":
        derivatives = exporters["derivatives"].compute()
        mask = exporters["mask"].compute()
        representation = exporters["representation"].compute()
        coordinates = exporters["coordinates"].compute()
    elif args.derivatives_from == "images":
        derivatives = exporters["derivatives"].from_image()
        mask = exporters["mask"].from_image()
        representation = exporters["representation"].from_image()
        coordinates = exporters["coordinates"].from_image()
    elif args.derivatives_from == "matfiles":
        derivatives = exporters["derivatives"].from_mat_file()
        mask = exporters["mask"].from_mat_file()
        representation = exporters["representation"].from_mat_file()
        coordinates = exporters["coordinates"].from_mat_file()
    else:
        raise ReferenceError(
            "Unkown option to read data. Options are `filters`, `images` or `matfiles`"
        )

    training_data = dict(derivatives)
    training_data.update(mask)
    training_data.update(representation)
    training_data.update(coordinates)

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
