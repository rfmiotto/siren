import cv2
import numpy as np

IMG_FILE = "potential.png"


def main():
    image = cv2.imread(IMG_FILE, cv2.IMREAD_UNCHANGED)

    is_alpha = image[..., 3] == 0

    mask = np.uint8(is_alpha * 255)

    cv2.imwrite("mask.png", mask)


if __name__ == "__main__":
    main()
