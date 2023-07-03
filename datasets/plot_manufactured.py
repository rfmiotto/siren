import matplotlib.pyplot as plt
from PIL import Image


def main():
    dpi = 100
    max_img_size = 600 * 3 / 4
    fig, axs = plt.subplots(
        2,
        3,
        figsize=(max_img_size / dpi, max_img_size / dpi),
        dpi=dpi,
        layout="compressed",
    )

    # original data
    original_image = Image.open("fig3_gt_potential.png").convert("L")
    gradx = Image.open("fig3_gt_vel_x.png").convert("L")
    grady = Image.open("fig3_gt_vel_y.png").convert("L")
    # laplace = Image.open("laplacian_noise.png").convert("L")
    axs[0, 0].imshow(original_image, cmap="gray")
    axs[0, 1].imshow(gradx, cmap="gray")
    axs[0, 2].imshow(grady, cmap="gray")
    # axs[0, 3].imshow(laplace, cmap="gray")

    # fit gradients
    original_image = Image.open("fig3_predic_potential.png").convert("L")
    gradx = Image.open("fig3_predic_vel_x.png").convert("L")
    grady = Image.open("fig3_predic_vel_y.png").convert("L")
    axs[1, 0].imshow(original_image, cmap="gray")
    axs[1, 1].imshow(gradx, cmap="gray")
    axs[1, 2].imshow(grady, cmap="gray")

    # fit laplace
    # original_image = Image.open("potential_fit_lapl_noise_2k.png").convert("L")
    # laplace = Image.open("laplacian_fit_lapl_noise_2k.png").convert("L")
    # axs[2, 0].imshow(original_image, cmap="gray")
    # axs[2, 3].imshow(laplace, cmap="gray")

    for ax in fig.axes:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        # ax.axison = False

    axs[0, 0].set_title("Image", loc="center", fontsize="10")
    axs[0, 1].set_title("Grad. x", loc="center", fontsize="10")
    axs[0, 2].set_title("Grad. y", loc="center", fontsize="10")
    # axs[0, 3].set_title("Laplacian", loc="center", fontsize="10")

    axs[0, 0].set_ylabel("Ground truth", fontsize="10")
    axs[1, 0].set_ylabel("Fit gradient", fontsize="10")
    # axs[2, 0].set_ylabel("Fit Laplacian", fontsize="10")

    # axs[1, 3].text(
    #     0.5,
    #     0.5,
    #     "N/A",
    #     transform=axs[1, 3].transAxes,
    #     fontsize=12,
    #     verticalalignment="center",
    #     horizontalalignment="center",
    # )
    # axs[2, 1].text(
    #     0.5,
    #     0.5,
    #     "N/A",
    #     transform=axs[2, 1].transAxes,
    #     fontsize=12,
    #     verticalalignment="center",
    #     horizontalalignment="center",
    # )
    # axs[2, 2].text(
    #     0.5,
    #     0.5,
    #     "N/A",
    #     transform=axs[2, 2].transAxes,
    #     fontsize=12,
    #     verticalalignment="center",
    #     horizontalalignment="center",
    # )

    # plt.subplots_adjust(wspace=0.05, hspace=0.05)

    # plt.show()
    plt.savefig("TESTE.png", bbox_inches="tight")
    # plt.savefig("TESTE.png")


if __name__ == "__main__":
    main()
