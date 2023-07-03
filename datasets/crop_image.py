from PIL import Image

IMAGES_FILENAMES_TO_CROP = [
    "image.png",
    "gradx.png",
    "grady.png",
    "mask.png",
]


TOP = 400
LEFT = 180
SIZE = 256


def main():
    for filename in IMAGES_FILENAMES_TO_CROP:
        img = Image.open(filename)

        right = LEFT + SIZE
        bottom = TOP + SIZE

        cropped = img.crop((LEFT, TOP, right, bottom))

        cropped.save(f"cropped_{filename}")


if __name__ == "__main__":
    main()
