import numpy as np
from PIL import Image
from scipy.io import savemat

image = Image.open("shaddowgraph_crop_flip.png")
image = (np.array(image) / 255.0).astype(np.float32)

background = Image.open("reset_shaddowgraph_crop_flip.png")
mean = np.float32(np.average(background) / 255.0)

shifted = image - mean

savemat("lucas.mat", {"data": shifted})
