from dataclasses import dataclass

import configargparse


@dataclass
class MyProgramArgs:
    """
    This is a helper to provide typehints of the arguments.
    All possible arguments must be declared in this dataclass.
    """

    config_filepath: any
    logging_root: str
    experiment_name: str
    image_size: int
    derivatives_from: str
    fit: str
    coordinates_file_name: str
    original_image_name: str
    gradient_x_image_name: str
    gradient_y_image_name: str
    laplacian_image_name: str
    mask_image_name: str
    transform_mean_option: str
    learning_rate: float
    num_epochs: int
    num_workers: int
    epochs_until_checkpoint: int
    epochs_until_summary: int
    load_checkpoint: bool


parser = configargparse.ArgumentParser()
parser.add(
    "-c",
    "--config_filepath",
    required=False,
    is_config_file=True,
    help="Path to config file.",
)

parser.add_argument(
    "--logging_root", type=str, default="./logs", help="Root for logging"
)
parser.add_argument(
    "--experiment_name",
    type=str,
    required=True,
    help="Name of subdirectory in logging_root where summaries and checkpoints"
    "will be saved.",
)
parser.add_argument(
    "--image_size",
    type=int,
    default=240,
    help="Image size in pixels (image is squared: image_size x image_size). default=240",
)
parser.add_argument(
    "--derivatives_from",
    type=str,
    choices=["images", "filters", "matfiles"],
    default="images",
    help="Method to read the derivatives. default=images",
)
parser.add_argument(
    "--fit",
    type=str,
    choices=["gradients", "laplacian"],
    default="gradients",
    help="Whether training will fit the gradient or the laplacian. default='gradient'",
)
parser.add_argument(
    "--coordinates_file_name",
    type=str,
    help="Name of the .mat file containing the coordinates."
    "It must have `coords_x` and `coords_y` column headings",
)
parser.add_argument(
    "--original_image_name",
    type=str,
    help="Name of the original image",
)
parser.add_argument(
    "--laplacian_image_name",
    type=str,
    help="Name of the image of the Laplacian",
)
parser.add_argument(
    "--mask_image_name",
    type=str,
    help="Name of the image of the mask",
)
parser.add_argument(
    "--gradient_x_image_name",
    type=str,
    help="Name of the image of the gradients in x",
)
parser.add_argument(
    "--gradient_y_image_name",
    type=str,
    help="Name of the image of the gradients in y",
)
parser.add_argument(
    "--transform_mean_option",
    type=str,
    choices=["image", "value", "self", "no_transform"],
    default="no_transform",
    help="How the mean will be evaluated. default='no_transform'",
)


# General training options
parser.add_argument(
    "--learning_rate", type=float, default=1e-4, help="learning rate. default=1e-4"
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=10_000,
    help="Number of epochs to train for. default=10,000",
)
parser.add_argument(
    "--num_workers", type=int, default=0, help="Number of workers. default=0"
)

parser.add_argument(
    "--epochs_until_checkpoint",
    type=int,
    default=1_000,
    help="Number of epochs until checkpoint is saved. default=1,000",
)
parser.add_argument(
    "--epochs_until_summary",
    type=int,
    default=200,
    help="Number of epochs until tensorboard summary is saved. default=1,000",
)
parser.add_argument(
    "--load_checkpoint",
    type=bool,
    default=False,
    help="Load checkpoint to continue training from a given point or make inference."
    "default=False",
)

args = MyProgramArgs(**vars(parser.parse_args()))
