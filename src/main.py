import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data.dataloader import DataLoader
import torchvision

from src.checkpoint import load_checkpoint, save_checkpoint
from src.datasets import DerivativesPixelDataset
from src.early_stopping import EarlyStopping
from src.transformations import get_transform_fn
from src.running import Runner, run_epoch, TrainingConfig, TrainingMetrics
from src.tensorboard_tracker import TensorboardTracker
from src.timeit import timeit
from src.hyperparameters import args
from src.siren import SIREN
from src.dtos import TrainingData
from src.tracking import NetworkTracker
from src.read_images import get_training_data


# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


@timeit
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_data = get_training_data()

    transform = get_transform_fn(args.transform_mean_option)

    dataset = DerivativesPixelDataset(
        images=training_data, transform=transform, device=device
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = SIREN()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    early_stopping = EarlyStopping(patience=1000)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=300, factor=0.5, verbose=True, min_lr=1e-7
    )

    config: TrainingConfig
    config = {
        "fit_option": args.fit,
        "device": device,
        "dtype_autocast": torch.bfloat16 if device.type == "cpu" else torch.float16,
        "optimizer": optimizer,
        "scaler": torch.cuda.amp.GradScaler(),
        "use_autocast": False,
    }

    runner = Runner(dataloader, model, config, TrainingMetrics())

    tracker = TensorboardTracker(
        log_dir=args.logging_root, filename=args.experiment_name
    )

    print(f"Running on {device}")

    if args.load_checkpoint:
        (epoch_from_previous_run, *_) = load_checkpoint(
            model=model, optimizer=optimizer, device=device
        )
        runner.epoch = epoch_from_previous_run

    best_acc = np.inf

    progress_bar = tqdm(enumerate(dataloader), total=args.num_epochs, leave=True)

    transform_to_tensor = torchvision.transforms.ToTensor()
    add_true_images_into_tracker(
        imagery=training_data,
        tracker=tracker,
        transform=transform_to_tensor,
        device=device,
    )

    for epoch in range(args.num_epochs):  # pylint: disable=unused-variable
        epoch_loss, epoch_psnr, epoch_ssim = run_epoch(
            runner=runner,
            tracker=tracker,
        )

        scheduler.step(epoch_loss)
        early_stopping(epoch_ssim)
        if early_stopping.stop:
            print("Ealy stopping")
            break

        # Flush tracker after every epoch for live updates
        tracker.flush()

        if should_save_model(runner.epoch, best_acc, epoch_ssim):
            best_acc = epoch_ssim
            save_checkpoint(runner.model, optimizer, runner.epoch, epoch_loss, best_acc)
            print(
                f"Best penr: {epoch_psnr} \t Best ssim: {epoch_ssim} \t Best loss: {epoch_loss}"
            )

        progress_bar.update(1)
        progress_bar.set_postfix(
            loss=f"{epoch_loss:.5f}",
            psnr=f"{epoch_psnr:.5f}",
            ssim=f"{epoch_ssim:.5f}",
        )

    progress_bar.close()


def should_save_model(epoch: int, best_acc: float, epoch_ssim: float) -> bool:
    if (epoch > 1000) and (epoch % 200 == 0) and (best_acc > epoch_ssim):
        return True
    return False


def add_true_images_into_tracker(
    imagery: TrainingData,
    tracker: NetworkTracker,
    transform: torchvision.transforms,
    device: torch.device,
) -> None:
    filtered_imagery = filter_out_none(imagery)
    for name, image in filtered_imagery.items():
        if name == "mask":
            continue
        img = transform(Image.fromarray(image)).to(device=device)
        tracker.add_image(f"ground_truth/{name}", img.view(1, -1), step=None)


def filter_out_none(kwargs: dict):
    return {k: v for k, v in kwargs.items() if v is not None}


if __name__ == "__main__":
    main()
