from numpy.typing import NDArray
from PIL import Image
from PIL.Image import Image as PILImage
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
from scipy import fftpack
from scipy.ndimage import gaussian_filter


def main():
    image = Image.open("shaddowgraph_crop_flip.png").convert("L")
    # imfs = loadmat("./REMOVE/emd/imfs.mat")["imfs"]
    # residue = loadmat("./REMOVE/emd/residue.mat")["residue"]
    imfs = loadmat("./imfs.mat")["imfs"]
    residue = loadmat("./residue.mat")["residue"]
    background = Image.open("shaddowgraph_crop_flip_reset.png").convert("L")

    prepare_img_for_emd(image, background)

    # The code below needs the EMD results

    num_imfs = imfs.shape[-1]

    imfs_filter = np.zeros_like(imfs)
    spectra_filter = np.zeros_like(imfs)
    spectra_original = np.zeros_like(imfs)


    masks = [None] * num_imfs
    for i in range(num_imfs):
        imfs_filter[:, :, i], masks[i] = filter_imf(imfs[:, :, i], threshold=0.02)
        spectra_filter[:, :, i] = get_spectrum(imfs_filter[:, :, i]).astype(int)
        spectra_original[:, :, i] = get_spectrum(imfs[:, :, i]).astype(int)

    # dpi = 100
    # width = (256 + 50) * num_imfs
    # height = (256 + 50) * 4
    # fig, ax = plt.subplots(
    #     nrows=4,
    #     ncols=num_imfs,
    #     figsize=(width / dpi, height / dpi),
    #     dpi=dpi,
    #     layout="compressed",
    # )
    # for i in range(num_imfs):
    #     ax[0, i].imshow(imfs[:, :, i], cmap="seismic")
    #     ax[1, i].imshow(imfs_filter[:, :, i], cmap="seismic")
    #     ax[2, i].imshow(spectra_original[:, :, i], cmap="gray")
    #     ax[3, i].imshow(spectra_filter[:, :, i], cmap="gray")
    # for i in range(num_imfs):
    #     ax[0, i].axis("off")
    #     ax[1, i].axis("off")
    #     ax[2, i].axis("off")
    #     ax[3, i].axis("off")
    # plt.savefig("emd_filtering.png")
    # plt.show()

    # residue -= np.mean(residue)

    reconstructed = np.sum(imfs_filter[:, :, :], axis=-1)  # + residue

    # reconstructed = apply_sponge(reconstructed)


    # dpi = 100
    # width = (256 + 50) * num_imfs
    # height = (256 + 50) * 2
    # fig, ax = plt.subplots(
    #     nrows=2,
    #     ncols=num_imfs,
    #     figsize=(width / dpi, height / dpi),
    #     dpi=dpi,
    #     layout="compressed",
    # )
    # for i in range(num_imfs):
    #     ax[0, i].imshow(imfs[:, :, i], cmap="seismic")
    #     ax[1, i].imshow(masks[i], cmap="gray")
    # for i in range(num_imfs):
    #     ax[0, i].axis("off")
    #     ax[1, i].axis("off")
    # plt.savefig("meeting.png", transparent=True)



    dpi = 100
    width = (256 + 50) * 5
    height = 256  # (256 + 50) * 4
    fig, ax = plt.subplots(
        nrows=1,
        ncols=5,
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        layout="compressed",
    )

    ax[0].imshow(np.sum(imfs, axis=-1), cmap="seismic")
    ax[0].set_title(r"$\sum$ IMFs", color="white")
    ax[1].imshow(reconstructed, cmap="seismic")
    ax[1].set_title(r"$\sum$ filtered IMFs", color="white")
    ax[2].imshow(residue, cmap="seismic")
    ax[2].set_title("Residue", color="white")
    ax[3].imshow(reconstructed + residue, cmap="seismic")
    ax[3].set_title("Reconstructed", color="white")
    ax[4].imshow(image, cmap="seismic")
    ax[4].set_title("Original", color="white")

    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")
    ax[3].axis("off")
    ax[4].axis("off")
    plt.savefig("meeting.png", transparent=True)
    plt.show()



    # vmin, vmax = reconstructed.min(), reconstructed.max()
    # maxabs = max(abs(vmin), abs(vmax))
    # plt.imshow(reconstructed, cmap="seismic", vmin=-maxabs, vmax=maxabs)
    # plt.show()

    # plt.plot(reconstructed[125, :], "r")
    # plt.plot(reconstructed[25, :], "b")
    # plt.show()

    # plt.imshow(residue)
    # plt.show()

    image = (np.array(image) / 255.0).astype(np.float32)

    plot_2_images(reconstructed, image, cmap="seismic")
    save_filtered_mat(reconstructed)


def get_spectrum(img):
    dft = fftpack.fft2((img).astype(float))
    shifted_dft = fftpack.fftshift(dft)
    magnitude_spectrum = 20 * np.log10(0.1 + shifted_dft)
    return magnitude_spectrum


def apply_sponge(data):
    pad = 5
    mask = np.ones_like(data)
    mask[0:pad, :] = 0
    mask[-pad - 1 :, :] = 0
    mask[:, 0:pad] = 0
    mask[:, -pad - 1 :] = 0
    mask = gaussian_filter(mask, sigma=4)
    data *= mask
    plt.imshow(data, cmap="gray")
    plt.show()
    return data


def filter_imf(imf: NDArray, threshold: float = 0.05):
    amplitude = imf.max() - imf.min()
    mask = np.ones_like(imf)
    mask[np.abs(imf) < threshold * amplitude] = 0
    mask = gaussian_filter(mask, sigma=10)
    # plt.imshow(mask, cmap="gray")
    # plt.show()
    return imf * mask, mask


def prepare_img_for_emd(image: PILImage, background: PILImage) -> None:
    image = (np.array(image) / 255.0).astype(np.float32)
    mean = np.float32(np.average(background) / 255.0)
    shiftted = image  # - mean  ³ FIXME check this
    savemat("original.mat", {"data": shiftted})


def plot_imfs_and_residue(imfs, residue, vmin, vmax) -> None:
    num_imfs = imfs.shape[-1]

    fig, ax = plt.subplots(nrows=1, ncols=num_imfs + 1)

    for i in range(num_imfs):
        ax[i].imshow(imfs[:, :, i], cmap="seismic", vmin=vmin, vmax=vmax)
        ax[i].set_title(f"IMF {i}")

    ax[i + 1].imshow(residue, cmap="seismic")
    ax[i + 1].set_title("Residue")

    plt.show()


def plot_2_images(img1, img2, cmap="gray") -> None:
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(img1, cmap=cmap)
    ax[0].axis("off")
    ax[0].set_title("Img1")
    ax[1].imshow(img2, cmap=cmap)
    ax[1].axis("off")
    ax[1].set_title("Img2")
    plt.show()


def save_filtered_image(filtered):
    plt.figure(figsize=(256, 256), dpi=1)
    plt.imshow(filtered, cmap="gray")
    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("emd.png", dpi=1)
    plt.close("all")


def save_filtered_mat(filtered):
    savemat("filtered.mat", {"laplacian": filtered})


if __name__ == "__main__":
    main()
