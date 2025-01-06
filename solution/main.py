from pathlib import Path
from argparse import ArgumentParser

import pandas as pd
import cv2
import numpy as np
from rich.progress import track

from CentroidProcessor import calculate_centroid
from GripperGeometry import overlay_gripper

def load_part_and_mask(part_path):
    """
    Loads the same image twice:
      1) As a color (BGR) image (part_img)
      2) As a grayscale image (part_mask)
    Returns (part_img, part_mask).
    """
    part_img = cv2.imread(str(part_path), cv2.IMREAD_COLOR)
    part_mask = cv2.imread(str(part_path), cv2.IMREAD_GRAYSCALE)
    return part_img, part_mask

def compute_amazing_solution(part_image_path: Path, gripper_image_path: Path) -> tuple[float, float, float]:
    """
    1) Compute the centroid of the part via CentroidProcessor.
    2) Overlay the gripper onto the part using that centroid.
    3) Return (x, y, angle).
    """

    centroid = calculate_centroid(str(part_image_path))  # returns (cx, cy) or None
    if centroid is None:
        print(f"Warning: No valid centroid found for {part_image_path}. Using (0,0).")
        x, y = 0, 0
    else:
        x, y = centroid

    angle = 0.0

    part_img, part_mask = load_part_and_mask(part_image_path)
    gripper_img, gripper_mask = load_part_and_mask(gripper_image_path)

    if part_img is None:
        print(f"Warning: Could not read part image: {part_image_path}")
        return x, y, angle
    if gripper_img is None:
        print(f"Warning: Could not read gripper image: {gripper_image_path}")
        return x, y, angle


    result, alpha = overlay_gripper(
    part_img, 
    part_mask,
    gripper_img, 
    gripper_mask,
    pixels_per_mm=1.0,
    angle_search_range=np.arange(0, 360, 1.0),
    x_center=x,  # Pass the computed centroid
    y_center=y,  # Pass the computed centroid
    )

    output_dir = Path("visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{part_image_path.stem}_with_{gripper_image_path.stem}.png"
    out_path = output_dir / out_name

    cv2.imwrite(str(out_path), result)
    print(f"Saved overlay -> {out_path}")

    return float(x), float(y), float(alpha)


def main():
    """The main function of your solution.

    Feel free to change it, as long as it maintains the same interface.
    """

    parser = ArgumentParser()
    parser.add_argument("input", help="input csv file")
    parser.add_argument("output", help="output csv file")
    args = parser.parse_args()

    # read the input csv file
    input_df = pd.read_csv(args.input)

    # compute the solution for each row
    results = []
    for _, row in track(
        input_df.iterrows(),
        description="Computing the solutions for each row",
        total=len(input_df),
    ):
        part_image_path = Path(row["part"])
        gripper_image_path = Path(row["gripper"])
        assert part_image_path.exists(), f"{part_image_path} does not exist"
        assert gripper_image_path.exists(), f"{gripper_image_path} does not exist"
        x, y, angle = compute_amazing_solution(part_image_path, gripper_image_path)
        results.append([str(part_image_path), str(gripper_image_path), x, y, angle])

    # save the results to the output csv file
    output_df = pd.DataFrame(results, columns=["part", "gripper", "x", "y", "angle"])
    output_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
