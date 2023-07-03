import matplotlib.pylab as plt
import numpy as np
from numpy.typing import NDArray
import cv2


PADDING = 0  # to avoid boundary effects with numerical derivatives
IMG_SIZE = 256


def make_plot(
    x_coords_matrix: NDArray,
    y_coords_matrix: NDArray,
    variable: NDArray,
    filename: str,
    vmax: int = 2.0,
    vmin: int = -2.0,
) -> None:
    plt.figure(figsize=(IMG_SIZE, IMG_SIZE), dpi=1)
    plt.pcolor(
        x_coords_matrix,
        y_coords_matrix,
        variable,
        cmap="gray",
        vmax=vmax,
        vmin=vmin,
    )
    plt.gca().set_aspect("equal")
    # plt.gcf().set_facecolor("gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=1, transparent=True)
    plt.close("all")
    # plt.show()


def main():
    expanded_img_size = IMG_SIZE + 2 * PADDING
    x = np.linspace(0, 1, num=expanded_img_size)
    y = np.linspace(0, 1, num=expanded_img_size)
    xgrid, ygrid = np.meshgrid(x, y)

    xgrid = xgrid - 0.5
    ygrid = ygrid - 0.5

    # signal = np.sin(10 * xgrid) + np.sin(5 * xgrid + 25 * ygrid**2)

    # signal = np.sin(10 * xgrid + 10 * ygrid)

    signal = np.sin(100 * xgrid**2 + 100 * ygrid**2)

    # signal = np.sin(10 * xgrid) + 0.2 * np.sin(5 * xgrid + 25 * ygrid**2) + xgrid / 2

    # signal = xgrid**2 + 0.5 * (ygrid + xgrid) ** 2

    grady, gradx = np.gradient(signal, edge_order=2)

    d2x, dxdy = np.gradient(grady, edge_order=2)
    dydx, d2y = np.gradient(gradx, edge_order=2)
    laplacian = d2x + d2y

    # Add noise
    # noise = 0.1 * np.random.uniform(-1, 1, size=expanded_img_size**2).reshape(
    #     (IMG_SIZE, IMG_SIZE)
    # )
    noise = np.random.normal(0, 0.1, size=expanded_img_size**2).reshape(
        (IMG_SIZE, IMG_SIZE)
    )

    noisy_gradx = gradx + noise
    noisy_grady = grady + noise
    noisy_laplacian = laplacian + noise

    # remove boundary effects

    # xgrid = xgrid[PADDING:-PADDING, PADDING:-PADDING]
    # ygrid = ygrid[PADDING:-PADDING, PADDING:-PADDING]
    # signal = signal[PADDING:-PADDING, PADDING:-PADDING]
    # gradx = gradx[PADDING:-PADDING, PADDING:-PADDING]
    # grady = grady[PADDING:-PADDING, PADDING:-PADDING]
    # laplacian = laplacian[PADDING:-PADDING, PADDING:-PADDING]

    # make plots

    max_abs = max(signal.max(), signal.min())
    make_plot(xgrid, ygrid, signal, "potential.png", vmax=max_abs, vmin=-max_abs)

    max_abs_x = max(gradx.max(), gradx.min())
    max_abs_y = max(grady.max(), grady.min())
    max_abs = max(max_abs_x, max_abs_y)
    make_plot(xgrid, ygrid, gradx, "velocity_x.png", vmax=max_abs, vmin=-max_abs)
    make_plot(xgrid, ygrid, grady, "velocity_y.png", vmax=max_abs, vmin=-max_abs)
    make_plot(
        xgrid, ygrid, noisy_gradx, "velocity_x_noise.png", vmax=max_abs, vmin=-max_abs
    )
    make_plot(
        xgrid, ygrid, noisy_grady, "velocity_y_noise.png", vmax=max_abs, vmin=-max_abs
    )

    max_abs = max(abs(laplacian.max()), abs(laplacian.min()))
    make_plot(xgrid, ygrid, laplacian, "laplacian.png", vmax=max_abs, vmin=-max_abs)
    make_plot(
        xgrid,
        ygrid,
        noisy_laplacian,
        "laplacian_noise.png",
        vmax=max_abs,
        vmin=-max_abs,
    )

    gradx = cv2.imread("velocity_x.png", cv2.IMREAD_GRAYSCALE)
    noisy_gradx = cv2.imread("velocity_x_noise.png", cv2.IMREAD_GRAYSCALE)
    psnr = cv2.PSNR(gradx, noisy_gradx)
    print(f"PSNR value for grad. x is {psnr} dB")

    grady = cv2.imread("velocity_y.png", cv2.IMREAD_GRAYSCALE)
    noisy_grady = cv2.imread("velocity_y_noise.png", cv2.IMREAD_GRAYSCALE)
    psnr = cv2.PSNR(grady, noisy_grady)
    print(f"PSNR value for grad. x is {psnr} dB")

    laplacian = cv2.imread("laplacian.png", cv2.IMREAD_GRAYSCALE)
    noisy_laplacian = cv2.imread("laplacian_noise.png", cv2.IMREAD_GRAYSCALE)
    psnr = cv2.PSNR(laplacian, noisy_laplacian)
    print(f"PSNR value for Laplacian is {psnr} dB")


if __name__ == "__main__":
    main()
