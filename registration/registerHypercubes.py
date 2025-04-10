import numpy as np
import cv2
import matplotlib.pyplot as plt

def show_alignment_preview(cube_ref, cube_aligned):
    img_ref = np.mean(cube_ref, axis=2)
    img_aligned = np.mean(cube_aligned, axis=2)

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

def register_cube(cube_ref, cube_to_align, method="ORB", max_features=500,
                  warp_mode=cv2.MOTION_EUCLIDEAN, number_of_iterations=5000, termination_eps=1e-6,
                  transform_type="AFF", crop=True):
    """
    Aligns a hyperspectral cube to a reference cube using a chosen method and optionally crops to overlapping region.

    Parameters:
        cube_ref (np.ndarray): Reference cube, shape (H, W, bands)
        cube_to_align (np.ndarray): Cube to align, same shape as cube_ref
        method (str): Registration method ("ORB", "SIFT", "ECC")
        max_features (int): Max number of features for ORB/SIFT
        warp_mode (int): ECC warp mode
        number_of_iterations (int): Max ECC iterations
        termination_eps (float): ECC convergence threshold
        transform_type (str): "AFF" for affine or "HOMO" for homography
        crop (bool): Whether to crop to overlapping region

    Returns:
        cube_ref_crop (np.ndarray): Cropped reference cube
        cube_aligned_crop (np.ndarray): Cropped aligned cube
        M (np.ndarray): Transformation matrix
        crop_coords (tuple): Cropping coordinates (x, y, w, h)
    """
    import cv2
    import numpy as np

    img_ref = np.mean(cube_ref, axis=2).astype(np.float32)
    img_to_align = np.mean(cube_to_align, axis=2).astype(np.float32)

    is_homography = transform_type == "HOMO"

    if method in ["ORB", "SIFT"]:
        img_ref_8bit = (img_ref * 256 / np.max(img_ref)).astype('uint8')
        img_to_align_8bit = (img_to_align * 256 / np.max(img_to_align)).astype('uint8')

        if method == "ORB":
            detector = cv2.ORB_create(max_features)
            norm_type = cv2.NORM_HAMMING
        else:
            detector = cv2.SIFT_create()
            norm_type = cv2.NORM_L2

        kp1, des1 = detector.detectAndCompute(img_ref_8bit, None)
        kp2, des2 = detector.detectAndCompute(img_to_align_8bit, None)

        if des1 is None or des2 is None:
            raise ValueError(f"{method} failed to detect features.")

        matcher = cv2.BFMatcher(norm_type, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        if is_homography:
            M, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
        else:
            M, _ = cv2.estimateAffinePartial2D(pts2, pts1)

    elif method == "ECC":
        img_ref = np.float32(img_ref)
        img_to_align = np.float32(img_to_align)

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            M = np.eye(3, 3, dtype=np.float32)
            is_homography = True
        else:
            M = np.eye(2, 3, dtype=np.float32)
            is_homography = False

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
        _, M = cv2.findTransformECC(img_ref, img_to_align, M, warp_mode, criteria)

    else:
        raise ValueError("Unsupported method. Choose from 'ORB', 'SIFT', or 'ECC'.")

    H, W, bands_ref = cube_ref.shape
    H_align, W_align, bands_align = cube_to_align.shape

    # Apply the transformation to each band separately, preserving the original number of bands
    cube_aligned = np.zeros_like(cube_to_align, dtype=np.float32)

    for b in range(bands_align):
        if is_homography:
            cube_aligned[:, :, b] = cv2.warpPerspective(cube_to_align[:, :, b], M, (W_align, H_align),
                                                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            cube_aligned[:, :, b] = cv2.warpAffine(cube_to_align[:, :, b], M, (W_align, H_align),
                                                   flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    if crop:
        # Create a binary mask to identify the region of overlap
        mask = np.ones((H_align, W_align), dtype=np.uint8)
        if is_homography:
            transformed_mask = cv2.warpPerspective(mask, M, (W_align, H_align), flags=cv2.WARP_INVERSE_MAP)
        else:
            transformed_mask = cv2.warpAffine(mask, M, (W_align, H_align), flags=cv2.WARP_INVERSE_MAP)

        overlap_mask = (transformed_mask > 0).astype(np.uint8)

        # Find the bounding rectangle for the overlapping region
        coords = cv2.findNonZero(overlap_mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            print(x,y,w,h)
            # Crop the reference and aligned cubes to the overlapping region
            cube_ref_crop = cube_ref[y:y + h, x:x + w, :]
            cube_aligned_crop = cube_aligned[y:y + h, x:x + w, :]
            return cube_ref_crop, cube_aligned_crop, M, (x, y, w, h)
        else:
            print('No overlap found')
            # If no overlap, return the original cubes
            return cube_ref, cube_aligned, M, None
    else :
        return cube_ref, cube_aligned, M, None


if __name__ == "__main__":
    from hypercubes.open import*
    path_ref=r'C:\Users\Usuario\Documents\DOC_Yannick\Hyperdoc_Test\Archivo chancilleria\SWIR\mat/MPD41a.mat'
    path_to_align = r'C:\Users\Usuario\Documents\DOC_Yannick\Hyperdoc_Test\Archivo chancilleria\VNIR/mat/MPD41a.mat'
    _,cube_ref = open_hyp(path_ref,open_window=False)
    print('Cube ref loaded')
    _,cube_to_align = open_hyp(path_to_align,open_window=False)
    print('Cube to align loaded')

    # Choose the method you want to test
    method = "ORB"  # ORB, SIFT, or ECC
    transform_type = "HOMO" # HOMO or AFF

    ref_crop, aligned_crop, transform, _ = register_cube(cube_ref, cube_to_align,crop=True)
    show_alignment_preview(ref_crop,aligned_crop)

    print("Registration done.")
    # print("Estimated transform:\n", transform)