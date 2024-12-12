import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse
import os



###########################################################################################################################
#                                                       Helper Functions                                                  #
###########################################################################################################################
"""Calculate the Laplacian of an image."""
def calculate_laplacian(image):
    # Separate each RGB cahnnel to apply laplacian separetly 
    channels = cv2.split(image)
    laplacian_channels = []
    
    # Define the Laplacian kernel
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

    # Apply the Laplacian kernel using convolution to each channel
    for channel in channels:
        laplacian_channel = cv2.filter2D(channel, -1, kernel)
        laplacian_channels.append(laplacian_channel)
    
    # Merge Laplacian channels back into an RGB image
    laplacian = cv2.merge(laplacian_channels)
    
    return laplacian


"""Solve the Poisson equation for blending."""
def solve_poisson_equation(source_img, target_img, mask):
    height, width, channels = source_img.shape
    laplacian = calculate_laplacian(source_img)

    # Initialize Sparse matrix A and vector b
    A = scipy.sparse.lil_matrix((mask.size, mask.size))
    b = np.zeros((mask.size, channels))

    index = lambda y, x: y * width + x

    #fill A and b
    for y in range(height):
        for x in range(width):
            idx = index(y, x)
            #if inside mask
            if mask[y, x] > 0:
                A[idx, idx] = -4

                #last try
                if y - 1 >= 0:
                    if mask[y - 1, x] > 0:
                        A[idx, index(y - 1, x)] = 1
                    #if on the border
                    if mask[y - 1, x] == 0:
                        b[idx] -= target_img[y - 1, x]
                if y + 1 < height:
                    if mask[y + 1, x] > 0:
                        A[idx, index(y + 1, x)] = 1
                    if mask[y + 1, x] == 0:
                        b[idx] -= target_img[y + 1, x]
                if x - 1 >= 0:
                    if mask[y, x - 1] > 0:
                        A[idx, index(y, x - 1)] = 1
                    if mask[y, x - 1] == 0:
                        b[idx] -= target_img[y, x - 1]
                if x + 1 < width:
                    if mask[y, x + 1] > 0:
                        A[idx, index(y, x + 1)] = 1
                    if mask[y, x + 1] == 0:
                        b[idx] -= target_img[y, x + 1]
            else: #outside the mask
                A[idx, idx] = -1
                b[idx] = target_img[y, x]

    A = A.tocsr()
    u_vec = np.zeros((height * width, channels), dtype=np.float32)

    # Solve the equation for each RGB channel
    for c in range(channels):
        b[:, c] += laplacian[:, :, c].flatten()
        u_vec[:, c] = spsolve(A, b[:, c])

    # Create an image from the solved u_vec
    u_img = np.zeros((height, width, channels), dtype=np.float32)
    for c in range(channels):
        u_img[:, :, c] = u_vec[:, c].reshape(height, width)

    return np.clip(u_img, 0, 255).astype(np.uint8)


"""Pad the image to have a final shape matching the target shape, centered at a given point."""
def pad(img, target_shape, center):
    #get diff in dim for each direction
    pad_h_top = max((center[1]-img.shape[0]//2),0)
    pad_h_bottom = max(target_shape[0] - pad_h_top - img.shape[0], 0)
    pad_w_left = max(center[0]-(img.shape[1] // 2),0)
    pad_w_right = max(target_shape[1] - pad_w_left - img.shape[1], 0)
    
    if img.ndim == 2:  # Grayscale mask
        padded_img = np.pad(img, ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)), mode='constant', constant_values=0)
    elif img.ndim == 3:  # RGB image
        padded_img = np.pad(img, ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right), (0, 0)), mode='constant', constant_values=0)
    
    return padded_img


"""Blend the source and target images using the solved Poisson equation result."""
def blend(blended_img, target_img, mask):

    result_img = target_img.copy()

    #fill inside mask and the edge of the mask with the blended solved img 
    result_img[mask > 0] = blended_img[mask > 0]
    

    return result_img


###########################################################################################################################
#                                                end of Helper Functions                                                  #
###########################################################################################################################



def poisson_blend(im_src, im_tgt, im_mask, center):
    
    
    # pad the mask and source image to the same shape as the target image
    im_mask_padded = pad(im_mask, im_tgt.shape[:2], center)
    source_padded = pad(im_src,im_tgt.shape[:2], center)
    
    
    # Convert images to float32 for calculations
    source_img = source_padded.astype(np.float32)
    target_img = im_tgt.astype(np.float32)
    
    
    #solve the poisson equation and finding the interpolant function
    u_image = solve_poisson_equation(source_img, target_img,im_mask_padded)
    
    
    #blend the target with solved u_image
    blended_img = blend(u_image, im_tgt, im_mask_padded)
    
    # Save the blended image
    output_dir = "blended"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "blended_image.png")
    cv2.imwrite(output_path, blended_img)
    
    
    return blended_img


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/teddy.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/teddy.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/grass_mountains.jpeg', help='mask file path')
    return parser.parse_args()

if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
