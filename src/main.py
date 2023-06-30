import numpy as np
import torch
import matplotlib.pyplot as plt

from src.checkpoint import load_checkpoint, save_checkpoint
from src.early_stopping import EarlyStopping
from src.loaders import get_dataloader
from src.transformations import get_transforms
from src.running import Runner, run_epoch
from src.tensorboard_tracker import TensorboardTracker
from src.timeit import timeit
from src.hyperparameters import args
from src.siren import SIREN

# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


@timeit
def main():
    """
    First, train only the final layers for a few epochs to avoid messing up
    with the gradients. Then, unfreeze all layers and train the entire network
    for the desired number of epochs.
    """

    image = plt.imread("dog.png")
    downsampling_factor = 4  # make training faster (this is not necessary...)
    image = image[::downsampling_factor, ::downsampling_factor]

    dataloader = get_dataloader(
        image,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        transform=get_transforms(),
    )

    model = SIREN()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    early_stopping = EarlyStopping(patience=40)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=13, factor=0.6, verbose=True
    )

    runner = Runner(dataloader, model, optimizer=optimizer)

    tracker = TensorboardTracker(log_dir=args.logging_root + args.experiment_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on {device}")

    if args.load_checkpoint:
        (
            epoch_from_previous_run,
            _,
            best_acc,
        ) = load_checkpoint(model=model, optimizer=optimizer, device=device)

        runner.epoch = epoch_from_previous_run

    best_acc = np.inf

    for epoch in range(args.num_epochs):  # pylint: disable=unused-variable
        epoch_loss, epoch_psnr, epoch_ssim = run_epoch(
            runner=runner,
            tracker=tracker,
        )

        scheduler.step(epoch_ssim)
        early_stopping(epoch_ssim)
        if early_stopping.stop:
            print("Ealy stopping")
            break

        # Flush tracker after every epoch for live updates
        tracker.flush()

        should_save_model = False  # best_acc > epoch_ssim # FIXME
        if should_save_model:
            best_acc = epoch_ssim
            save_checkpoint(runner.model, optimizer, runner.epoch, epoch_loss, best_acc)
            print(
                f"Best penr: {epoch_psnr} \t Best ssim: {epoch_ssim} \t Best loss: {epoch_loss}"
            )

        print(
            f"Epoch psnr: {epoch_psnr} \t Epoch ssim: {epoch_ssim} \t Epoch loss: {epoch_loss}\n"
        )


if __name__ == "__main__":
    main()
