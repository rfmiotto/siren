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
    pass_entire_image: bool
    batch_size: int
    learning_rate: float
    num_epochs: int
    num_workers: int
    epochs_until_checkpoint: int
    steps_until_summary: int
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
    "--pass_entire_image",
    type=bool,
    default=True,
    help="Flag to pass the entire image (good speedup but consumes a lot of memory)"
    "or batches of the image (slower, but allows to handle big images). default=True",
)

# General training options
parser.add_argument(
    "--batch_size",
    type=int,
    default=1_000,
    help="Make sure that the batch size is not greater than the total number of pixels"
    "of the image. Also, when `--pass_entire_image=True`, the batch_size will be"
    "set to 1. default=1_000",
)
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
    default=25,
    help="Time interval in seconds until checkpoint is saved. default=25",
)
parser.add_argument(
    "--steps_until_summary",
    type=int,
    default=1_000,
    help="Time interval in seconds until tensorboard summary is saved. default=1,000",
)
parser.add_argument(
    "--load_checkpoint",
    type=bool,
    default=False,
    help="Load checkpoint to continue training from a given point or make inference."
    "default=False",
)

args = MyProgramArgs(**vars(parser.parse_args()))
