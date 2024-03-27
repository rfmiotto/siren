from PIL import Image
import numpy as np
from scipy.io import loadmat
from scipy import integrate
import matplotlib.pyplot as plt


# x = np.linspace(0, 3 * np.pi, 300)
# f = 2 * np.ones_like(x)
# intf = integrate.cumtrapz(f)
# iif = integrate.cumtrapz(intf)
# plt.plot(x, f)
# plt.show()
# plt.plot(x[:-1], intf)
# plt.show()
# plt.plot(x[:-2], iif)
# plt.show()

index = 128

laplacian = loadmat("filtered.mat")["laplacian"]
# laplacian = loadmat("original.mat")["data"]

# laplacian_fit = loadmat("outputs/preds/laplacian_20000.mat")["data"]
# integrated_field = loadmat("outputs/preds_20000.mat")["data"]

laplacian_fit = loadmat(
    "outputs_todos/outputs_filtered_102_w60/preds/laplacian_15800.mat"
)["data"]
integrated_field = loadmat("outputs_todos/outputs_filtered_102_w60/preds_15800.mat")[
    "data"
]




# background = np.array(Image.open("shaddowgraph_crop_flip_reset.png").convert("L")) / 255.0
# mean = np.average(background)
# laplacian -= mean

# residue = loadmat("./residue.mat")["residue"]
# integrated_field -= integrated_field[0]
# integrated_field += residue


laplacian_jet_line = laplacian[index, :]
integral_laplacian_jet_line = integrate.cumtrapz(integrate.cumtrapz(laplacian_jet_line))
# integral_laplacian_jet_line2 = np.cumsum(np.cumsum(laplacian_jet_line))
laplacian_fit_jet_line = laplacian_fit[0, index, :]
integral_laplacian_fit_jet_line = integrate.cumtrapz(
    integrate.cumtrapz(laplacian_fit_jet_line)
)
integrated_field_jet_line = integrated_field[0, index, :]
rec_derivative = np.gradient(np.gradient(integrated_field_jet_line))


residue = loadmat("emd_cumsum.mat")["residue"].squeeze()
imf = loadmat("emd_cumsum.mat")["imf"].squeeze()
vmax = imf.max()
vmin = imf.min()
imf_amplitude = abs(vmax - vmin)
###
vmax = integrated_field_jet_line.max()
vmin = integrated_field_jet_line.min()
scaled_int_field = (integrated_field_jet_line - vmin) / abs(vmax - vmin) * imf_amplitude
scaled_int_field[:-2] += residue


# plt.figure(figsize=(256, 256), dpi=1)
# plt.imshow(laplacian, cmap="gray")
# plt.gca().set_aspect("equal")
# plt.axis("off")
# plt.tight_layout()
# plt.savefig("meeting.png", dpi=1)
# plt.close("all")


fig, axes = plt.subplots(nrows=3, ncols=3)
fig.set_tight_layout({"pad": 0})

axes[0, 0].imshow(laplacian, cmap="gray")
axes[0, 0].hlines(index, xmin=0, xmax=255, colors="r")
axes[0, 0].set_axis_off()
axes[1, 0].plot(laplacian_jet_line, color="#89FFDB")
axes[1, 0].set_title("Shadowgraph signal")
# axes[2, 0].plot(integral_laplacian_jet_line, color="#89FFDB")
axes[2, 0].plot(imf, color="#89FFDB")
axes[2, 0].set_title("Integral")

axes[0, 1].imshow(laplacian_fit[0, :, :], cmap="gray")
axes[0, 1].hlines(index, xmin=0, xmax=255, colors="r")
axes[0, 1].set_axis_off()
axes[1, 1].plot(laplacian_fit_jet_line, color="#89FFDB")
axes[1, 1].set_title("NN Laplacian fit")
axes[2, 1].plot(integral_laplacian_fit_jet_line, color="#89FFDB")
axes[2, 1].set_title("Integral")

axes[0, 2].imshow(integrated_field[0, :, :], cmap="gray")
axes[0, 2].hlines(index, xmin=0, xmax=255, colors="r")
axes[0, 2].set_axis_off()
# axes[1, 2].set_axis_off()
axes[1, 2].plot(rec_derivative, color="#89FFDB")
axes[2, 2].plot(integrated_field_jet_line, color="#89FFDB")
# axes[2, 2].plot(scaled_int_field, color="#89FFDB")
axes[2, 2].set_title("NN prediction")

for i in range(3):
    for j in range(3):
        axes[i, j].spines['bottom'].set_color('white')
        axes[i, j].spines['top'].set_color('white')
        axes[i, j].spines['left'].set_color('white')
        axes[i, j].spines['right'].set_color('white')
        axes[i, j].xaxis.label.set_color('white')
        axes[i, j].yaxis.label.set_color('white')
        axes[i, j].title.set_color('white')
        axes[i, j].tick_params(axis='x', colors='white')
        axes[i, j].tick_params(axis='y', colors='white')

plt.savefig("results.png", transparent=True)

plt.show()
