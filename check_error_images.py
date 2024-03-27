"""
The prediction of the Laplacian usually contains some shadow effects.
In this script I test if the difference between the ground truth and the 
prediction is a periodic signal. This is because the periodic signal
automatically satisfies the Laplace equation.

UPDATE: After running this script,no periodic signal was observed...
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageChops
from scipy.ndimage import laplace


def main():
    ground_truth = Image.open("jhtdb_gt.png", "r").convert("L")
    prediction = Image.open("jhtdb_pred.png", "r").convert("L")

    difference = np.array(ImageChops.difference(ground_truth, prediction))

    print(f"Manhattan norm: {manhattan_norm(difference)}")
    print(f"Zero norm: {zero_norm(difference)}")

    lapl_gt = laplace(ground_truth)
    lapl_pred = laplace(prediction)
    _, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].imshow(lapl_gt, cmap="gray")
    axs[1].imshow(lapl_pred, cmap="gray")
    axs[0].set_title("Laplacian GT")
    axs[1].set_title("Laplacian Pred")
    plt.show()

    _, axs = plt.subplots(nrows=1, ncols=3)
    axs[0].imshow(ground_truth, cmap="gray")
    axs[1].imshow(prediction, cmap="gray")
    axs[2].imshow(difference, vmax=255, vmin=0, cmap="gray")
    axs[0].set_title("GT")
    axs[1].set_title("Pred")
    axs[2].set_title("Difference (normalized)")
    plt.show()

    ground_truth = standardize(np.array(ground_truth))
    prediction = standardize(np.array(prediction))
    difference = abs(ground_truth - prediction)

    vmax = ground_truth.max()
    vmin = ground_truth.min()
    amplitude = vmax - vmin

    _, axs = plt.subplots(nrows=1, ncols=3)
    axs[0].imshow(ground_truth, cmap="gray")
    axs[1].imshow(prediction, cmap="gray")
    axs[2].imshow(difference, vmax=amplitude, vmin=0, cmap="gray")
    axs[0].set_title("GT")
    axs[1].set_title("Pred")
    axs[2].set_title("Difference (standardized)")
    plt.show()

    print(f"Relative residual norm: {relative_residual_norm(difference, ground_truth)}")
    print(f"Manhattan norm: {manhattan_norm(difference)}")
    print(f"Zero norm: {zero_norm(difference)}")


def standardize(array: NDArray) -> NDArray:
    return (array - array.mean()) / array.std()


def relative_residual_norm(difference: NDArray, original: NDArray):
    return np.linalg.norm(difference) / np.linalg.norm(original)


def manhattan_norm(array: NDArray) -> float:
    """
    Manhattan norm (the sum of the absolute values) is a measure of how much
    the image is off. Here, we give the result divided by the number of pixels.
    Also, the image of the difference is normalized between [0-1], so that the
    final output can be interpreted as the percentage of deviation of the images
    """
    rescaled_array = array / 255.0
    return np.sum(rescaled_array) / array.size


def zero_norm(array: NDArray):
    """
    Zero norm (the number of elements not equal to zero) tell how many pixels
    differ. Here, we give the result divided by the number of pixels. This means
    that this metric shows the percentage of pixels of the image that differ.
    """
    return np.linalg.norm(array.ravel(), ord=0) / array.size


if __name__ == "__main__":
    main()
