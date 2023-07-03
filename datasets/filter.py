from PIL.Image import Image as ImageType
from PIL import Image, ImageFilter
from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread


from src.hyperparameters import args


def apply_mask(array, radius):
    rows, cols = array.shape
    center_row, center_col = int(rows / 2), int(cols / 2)
    mask = np.ones_like(array)
    x, y = np.ogrid[:rows, :cols]
    masked_area = (x - center_row) ** 2 + (y - center_col) ** 2 <= radius**2
    mask[masked_area] = 0
    masked_array = array * mask
    return masked_array


def filter_img(img: ImageType) -> ImageType:
    """
    https://forum.image.sc/t/soft-filtering-in-fourier-space-in-python/89751/3

    https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/4f78a80260e88dc96d2e6fabd6f589e6a2f83219/notebooks/filters/imagesc_dec13_filtering.ipynb

    - generate the FFT
    - generate the binary mask of the FFT
    - run a gaussian filter on the binary mask with some suitable sigma
    - multiply the FFT by the filtered mask
    - run the inverse FFT
    """
    img = (np.array(img) / 255.0).astype(np.float32)
    dft = fftpack.fft2((img).astype(float))
    shifted_dft = fftpack.fftshift(dft)
    magnitude_spectrum = 20 * np.log10(0.1 + shifted_dft)
    # plt.imshow(magnitude_spectrum.astype(int), cmap="gray")
    # plt.show()
    masked = apply_mask(shifted_dft, radius=2)
    magnitude_spectrum = 20 * np.log10(0.1 + masked)
    # plt.imshow(magnitude_spectrum.astype(int), cmap="gray")
    # plt.show()
    filtered_img = fftpack.ifft2(fftpack.ifftshift(masked)).real
    # plt.imshow(img.astype(int), cmap="gray")
    # plt.imshow(filtered_img.astype(int), cmap="gray")
    # plt.show()
    # return img.filter(ImageFilter.GaussianBlur)
    plt.figure(figsize=(256, 256), dpi=1)
    plt.axis("off")
    plt.tight_layout()
    plt.imshow(img.astype(int), cmap="gray")
    plt.savefig("img_original", dpi=1)

    plt.figure(figsize=(256, 256), dpi=1)
    plt.axis("off")
    plt.tight_layout()
    plt.imshow(filtered_img.astype(int), cmap="gray")
    plt.savefig("img_filtered", dpi=1)
    return masked


if __name__ == "__main__":
    image = Image.open(args.laplacian_image_name, "r").convert("L")
    filtered_image = filter_img(image)
    # filtered_image.save("filtered_image.png")
