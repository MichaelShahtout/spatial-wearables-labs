import argparse
import numpy as np
import time
import torch

from PIL import Image


def load_image_matrix(img_path: str):
    """
    Load an image by path into a matrix
    """

    image = Image.open(img_path).convert("RGB")
    image_mtx = np.array(image)

    return image_mtx


def grayscale_image(image_matrix: torch.Tensor):
    """
    Converts an image matrix to grayscale by normalizing and performing a grayscale matmul on each pixel RGB
    """

    mono_weights = torch.Tensor([0.299, 0.587, 0.114])
    grayscaled_image = (image_matrix * mono_weights).sum(dim=-1)
    return grayscaled_image
            

def _denoise_image(image_matrix: torch.Tensor):
    """
    Denoise an image matrix by convolving with a standard denoising kernel
    """

    kernel = torch.Tensor(
        [
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
        ]
    )

    image_matrix = image_matrix.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add in and out channel dimensions
    
    # Convolve with kernel
    denoised_image = torch.nn.functional.conv2d(image_matrix, kernel, padding=1).squeeze(0).squeeze(0)

    return denoised_image

def compute_gradient_image(image_matrix: torch.Tensor, use_torch_conv2d: bool = False):
    """
    Returns a matrix of the gradients for each pixel
    """

    image_mtx = image_matrix
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

    grayscaled_image = grayscale_image(image_mtx)


    # Convolve with sobel operators
    # Could use built in torch method to conv2d, but we're going to do this manually for the exercise of actually building the intuition of how it works
    time_start = time.time()
    if use_torch_conv2d:
        print("Using torch conv2d")
        grayscaled_image = grayscaled_image.unsqueeze(0).unsqueeze(0)
        sobel_x = sobel_x.unsqueeze(0).unsqueeze(0)
        sobel_y = sobel_y.unsqueeze(0).unsqueeze(0)
        convolved_x = torch.nn.functional.conv2d(grayscaled_image, sobel_x, padding=1).squeeze(0).squeeze(0)
        convolved_y = torch.nn.functional.conv2d(grayscaled_image, sobel_y, padding=1).squeeze(0).squeeze(0)
        
        gradient_magnitude = torch.sqrt(convolved_x**2 + convolved_y**2)
        gradient_direction = torch.atan2(convolved_y, convolved_x)
    else:
        # Naive convolution implementation
        print("Using naive implementation convolution")
        h, w = grayscaled_image.shape
        convolved_x = torch.zeros((h, w))
        convolved_y = torch.zeros((h, w))
        for i in range(1, len(grayscaled_image)-1):
            for j in range(1, len(grayscaled_image[i])-1):
                # convolved_x[i, j] = area around pixel convolved by sobel X
                # convolved_y[i, j] = area around pixel convolved by sobel Y
                area_around_pixel = torch.Tensor([
                    [grayscaled_image[i-1][j-1], grayscaled_image[i-1][j], grayscaled_image[i-1][j+1]],
                    [grayscaled_image[i][j-1], grayscaled_image[i][j], grayscaled_image[i][j+1]],
                    [grayscaled_image[i+1][j-1], grayscaled_image[i+1][j], grayscaled_image[i+1][j+1]]
                ])
                convolved_x[i][j] = (area_around_pixel * sobel_x).sum(dim=None).item()
                convolved_y[i][j] = (area_around_pixel * sobel_y).sum(dim=None).item()

        gradient_magnitude = torch.sqrt(convolved_x**2 + convolved_y**2)
        gradient_direction = torch.atan2(convolved_y, convolved_x)

    time_end = time.time()
    print(f"Time taken to compute gradient image: {time_end - time_start} seconds")
    return (gradient_magnitude, gradient_direction)

def mark_edges(gradient_strengths: torch.Tensor, threshold: float):
    """
    Marks edges in a gradient image by thresholding
    """

    edges = (gradient_strengths > threshold).float()

    return edges

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="cv/images/three_blocks.png")
    parser.add_argument("--threshold", type=float, default=40)
    parser.add_argument("--use_torch_conv2d", action="store_true", default=False)
    args = parser.parse_args()

    
    image_matrix = load_image_matrix(args.image_path)
    image_matrix = torch.Tensor(image_matrix)

    grayscaled = grayscale_image(image_matrix)

    denoised_grayscale = _denoise_image(grayscaled)

    gradient_strengths, gradient_directions = compute_gradient_image(denoised_grayscale, args.use_torch_conv2d)
    
    # Save denoised gradient image
    edges = mark_edges(gradient_strengths, args.threshold)

    # Convert to PIL image and save
    edges_image = Image.fromarray((edges.numpy() * 255).astype(np.uint8))
    edges_image.save("cv/images/three_blocks_edges.png")






