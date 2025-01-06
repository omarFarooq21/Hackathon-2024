import cv2
import os
from pathlib import Path

def calculate_centroid(mask_path: str):
    """
    Calculate the centroid of a binary mask image.
    :param mask_path: File path to the mask (expects grayscale).
    :return: (cx, cy) if valid, else None
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: could not load mask file {mask_path}")
        return None

    moments = cv2.moments(mask)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)
    else:
        print(f"Warning: mask file {mask_path} has zero area.")

        return None


def process_centroids(mask_folder: str):
    centroids = []
    for fname in os.listdir(mask_folder):
        if fname.startswith("mask_") and fname.endswith(".png"):
            mask_path = os.path.join(mask_folder, fname)
            centroid = calculate_centroid(mask_path)
            centroids.append((fname, centroid))
    return centroids
