import torch
from torchmetrics import Metric


class RelativeResidualError(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            "derivatives",
            default=torch.empty(0),
            dist_reduce_fx="cat",
        )
        self.add_state(
            "target",
            default=torch.empty(0),
            dist_reduce_fx="cat",
        )

    # pylint: disable=arguments-differ
    def update(self, derivatives: torch.Tensor, target: torch.Tensor) -> None:
        # pylint: disable=attribute-defined-outside-init
        self.derivatives = torch.cat((self.derivatives, derivatives.squeeze()), 0)
        self.target = torch.cat((self.target, target.squeeze()), 0)

    def compute(self) -> torch.Tensor:
        # pylint: disable=not-callable
        return torch.linalg.norm(self.target - self.derivatives) / torch.linalg.norm(
            self.target
        )


# def peak_signal_noise_ratio(preds: torch.Tensor, target: torch.Tensor):
#     batch_size = preds.shape[0]

#     preds = preds.detach().cpu().numpy()
#     target = target.detach().cpu().numpy()

#     psnrs = np.empty(batch_size)

#     for i in range(batch_size):
#         prediction = preds[i].transpose(1, 2, 0)
#         prediction = (prediction / 2.0) + 0.5
#         prediction = np.clip(prediction, a_min=0.0, a_max=1.0)

#         trgt = target[i].transpose(1, 2, 0)
#         trgt = (trgt / 2.0) + 0.5

#         psnr = skimage.metrics.peak_signal_noise_ratio(prediction, trgt, data_range=1)

#         psnrs[i] = psnr

#     return np.mean(psnrs)


# def structural_similarity(preds: torch.Tensor, target: torch.Tensor):
#     batch_size = preds.shape[0]

#     preds = preds.detach().cpu().numpy()
#     target = target.detach().cpu().numpy()

#     ssims = np.empty(batch_size)

#     for i in range(batch_size):
#         prediction = preds[i].transpose(1, 2, 0)
#         prediction = (prediction / 2.0) + 0.5
#         prediction = np.clip(prediction, a_min=0.0, a_max=1.0)

#         trgt = target[i].transpose(1, 2, 0)
#         trgt = (trgt / 2.0) + 0.5

#         ssim = skimage.metrics.structural_similarity(
#             prediction, trgt, channel_axis=2, data_range=1
#         )

#         ssims[i] = ssim

#     return np.mean(ssims)
