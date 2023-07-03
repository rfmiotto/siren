import numpy as np
from scipy.io import loadmat
from scipy import integrate
import matplotlib.pyplot as plt


index = 128

laplacian = loadmat("lucas.mat")["data"]
laplacian_fit = loadmat("outputs/preds/laplacian_20000.mat")["data"]
integrated_field = loadmat("outputs/preds_20000.mat")["data"]

laplacian_jet_line = laplacian[index, :]
integral_laplacian_jet_line = integrate.cumtrapz(integrate.cumtrapz(laplacian_jet_line))
laplacian_fit_jet_line = laplacian_fit[0, index, :]
integral_laplacian_fit_jet_line = integrate.cumtrapz(
    integrate.cumtrapz(laplacian_fit_jet_line)
)
integrated_field_jet_line = integrated_field[0, index, :]


fig, axes = plt.subplots(nrows=3, ncols=3)
fig.set_tight_layout({"pad": 0})

axes[0, 0].imshow(laplacian, cmap="gray")
axes[0, 0].hlines(index, xmin=0, xmax=255, colors="r")
axes[0, 0].set_axis_off()
axes[1, 0].plot(laplacian_jet_line)
axes[1, 0].set_title("Shadowgraph signal")
axes[2, 0].plot(integral_laplacian_jet_line)
axes[2, 0].set_title("Integral")

axes[0, 1].imshow(laplacian_fit[0, :, :], cmap="gray")
axes[0, 1].hlines(index, xmin=0, xmax=255, colors="r")
axes[1, 1].plot(laplacian_fit_jet_line)
axes[1, 1].set_title("NN Laplacian fit")
axes[2, 1].plot(integral_laplacian_fit_jet_line)
axes[2, 1].set_title("Integral")

axes[0, 2].imshow(integrated_field[0, :, :], cmap="gray")
axes[0, 2].hlines(index, xmin=0, xmax=255, colors="r")
axes[1, 2].set_axis_off()
axes[2, 2].plot(integrated_field_jet_line)
axes[2, 2].set_title("NN prediction")

plt.show()

# plt.imshow(laplacian[0, :, :], cmap="gray")
# plt.show()


# pred[0, 128, :] = pred.max()
# plt.imshow(pred[0, :, :], cmap="gray")
# # plt.plot(pred[0, :, :])
# plt.show()


# imfs = loadmat("./emd/6_imfs.mat")
# residue = loadmat("./emd/6_residue.mat")


# plt.imshow(imfs["imfs"][:, :, 0])
# plt.show()


# plt.imshow(residue["residue"])
# plt.show()


# rec = np.sum(imfs["imfs"], axis=-1)

# plt.figure(figsize=(256, 256), dpi=1)
# plt.imshow(rec, cmap="gray")
# plt.gca().set_aspect("equal")
# plt.axis("off")
# plt.tight_layout()
# plt.savefig("emd.png", dpi=1)
# plt.close("all")
# plt.show()

# num_imfs = imfs["imfs"].shape[-1]
# for i in range(num_imfs):
#     rec = imfs["imfs"]

# print(residue)
