import numpy as np
from scipy.io import loadmat, savemat
from scipy import integrate
import matplotlib.pyplot as plt

index = 128

laplacian = loadmat("filtered.mat")["laplacian"]
# laplacian = loadmat("original.mat")["data"]
laplacian_fit = loadmat("outputs/preds/laplacian_15800.mat")["data"]
integrated_field = loadmat("outputs/preds_15800.mat")["data"]

laplacian_fit = loadmat(
    "outputs_todos/outputs_filtered_102_w60/preds/laplacian_15800.mat"
)["data"]
integrated_field = loadmat("outputs_todos/outputs_filtered_102_w60/preds_15800.mat")[
    "data"
]

laplacian_jet_line = laplacian[index, :]
integral_laplacian_jet_line = integrate.cumtrapz(integrate.cumtrapz(laplacian_jet_line))
# integral_laplacian_jet_line2 = np.cumsum(np.cumsum(laplacian_jet_line))
laplacian_fit_jet_line = laplacian_fit[0, index, :]
integral_laplacian_fit_jet_line = integrate.cumtrapz(
    integrate.cumtrapz(laplacian_fit_jet_line)
)
integrated_field_jet_line = integrated_field[0, index, :]
rec_derivative = np.gradient(np.gradient(integrated_field_jet_line))


# plt.figure(figsize=(256, 256), dpi=1)
# plt.imshow(laplacian, cmap="gray")
# plt.gca().set_aspect("equal")
# plt.axis("off")
# plt.tight_layout()
# plt.savefig("meeting.png", dpi=1)
# plt.close("all")

# savemat("cumsum.mat", {"data": integral_laplacian_jet_line})
imf = loadmat("emd_cumsum.mat")["imf"].squeeze()
res = loadmat("emd_cumsum.mat")["residue"].squeeze()


fig, axes = plt.subplots(nrows=3, ncols=3)
fig.set_tight_layout({"pad": 0})

axes[0, 0].imshow(laplacian, cmap="gray")
axes[0, 0].hlines(index, xmin=0, xmax=255, colors="r")
axes[0, 0].set_axis_off()
axes[1, 0].plot(laplacian_jet_line)
axes[1, 0].set_title("Shadowgraph signal")
axes[2, 0].plot(integral_laplacian_jet_line)
# axes[2, 0].plot(integral_laplacian_jet_line2, "--")
axes[2, 0].set_title("Integral")

axes[0, 1].imshow(laplacian_fit[0, :, :], cmap="gray")
axes[0, 1].hlines(index, xmin=0, xmax=255, colors="r")
axes[1, 1].plot(laplacian_fit_jet_line)
axes[1, 1].set_title("NN Laplacian fit")
axes[2, 1].plot(integral_laplacian_fit_jet_line)
# axes[2, 1].plot(imf)
axes[2, 1].set_title("Integral")

axes[0, 2].imshow(integrated_field[0, :, :], cmap="gray")
axes[0, 2].hlines(index, xmin=0, xmax=255, colors="r")
# axes[1, 2].set_axis_off()
axes[1, 2].plot(rec_derivative)
# axes[2, 2].plot(integrated_field_jet_line)
vmax = integrated_field_jet_line.max()
vmin = integrated_field_jet_line.min()
aux = (integrated_field_jet_line - vmin) / (vmax - vmin)
aux = aux[:-2] * (imf.max() - imf.min()) + res
axes[2, 2].plot(aux)
axes[2, 2].set_title("NN prediction")

plt.show()
