import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_alignment_preview(cube_ref, cube_aligned):
    img_ref = np.mean(cube_ref, axis=0)
    img_aligned = np.mean(cube_aligned, axis=0)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_ref, cmap='gray')
    plt.title("Reference")

    plt.subplot(1, 3, 2)
    plt.imshow(img_aligned, cmap='gray')
    plt.title("Aligned")

    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(img_ref - img_aligned), cmap='hot')
    plt.title("Difference")

    plt.tight_layout()
    plt.show()

def register_cube_rigid(cube_ref, cube_to_align, max_features=500):
    """
    Aligns a hyperspectral cube (cube_to_align) to a reference cube (cube_ref)
    using ORB feature matching and rigid (translation + rotation) transform.

    Parameters:
        cube_ref (np.ndarray): Reference cube, shape (H, W, bands)
        cube_to_align (np.ndarray): Cube to align, same shape as cube_ref
        max_features (int): Maximum number of ORB features

    Returns:
        cube_aligned (np.ndarray): Aligned cube, same shape as input
        transform_matrix (np.ndarray): 2x3 affine transformation matrix
    """

    # Step 1: Convert cubes to grayscale images for feature matching
    img_ref = np.mean(cube_ref, axis=2).astype(np.uint8)
    img_to_align = np.mean(cube_to_align, axis=2).astype(np.uint8)

    # Step 2: ORB keypoints and descriptors
    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(img_ref, None)
    kp2, des2 = orb.detectAndCompute(img_to_align, None)

    if des1 is None or des2 is None:
        raise ValueError("ORB failed to detect features in one of the images.")

    # Step 3: Feature matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Step 4: Get matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Step 5: Estimate transform (rigid)
    M, inliers = cv2.estimateAffinePartial2D(pts2, pts1)

    if M is None:
        raise ValueError("Failed to estimate transform matrix.")

    # Step 6: Apply transform to each band
    H, W, bands = cube_to_align.shape
    cube_aligned = np.zeros_like(cube_to_align)

    for b in range(bands):
        cube_aligned[:, :, b] = cv2.warpAffine(cube_to_align[:, :, b], M, (W, H))

    return cube_aligned, M

if __name__ == "__main__":
    from hypercubes.open import*
    path_folder=
    cube_ref = open_hyp("hypercube_reference.npy")  # shape: (H, W, bands)
    cube_to_align = np.load("hypercube_to_align.npy")

    aligned_cube, transform = register_cube_rigid(cube_ref, cube_to_align)
    np.save("hypercube_aligned.npy", aligned_cube)

    print("Registration done.")
    print("Estimated transform:\n", transform)