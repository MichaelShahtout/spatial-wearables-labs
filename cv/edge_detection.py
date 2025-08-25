import numpy as np
import torch

from PIL import Image


def load_image_matrix(img_path: str):
    """
    Load an image by path into a matrix
    """

    image = Image.open(img_path)
    image_mtx = np.array(image)

    return image_mtx

def compute_gradient_image(image_path: str):
    """
    Returns a matrix of the gradients for each pixel
    """

    image_mtx = load_image_matrix(image_path)
    sobel_x = torch.Tensor(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ]
    )
    sobel_y = sobel_x.T

    # Optimally we would compute gradient for every pixel at the same time
    # Is there a way to make a kernel do this for this atomic operation with torch?

    # First things first, we need to compute to mono with some weighting matrix along the RGB values


    # TODO for next few days:
    # 1. Work on convolution for computing gradient in image
    # 2. Experiment with different operators to see how it affects end image for both edge detection and depth mapping
    # 3. Code review






