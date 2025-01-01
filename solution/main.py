from pathlib import Path
from argparse import ArgumentParser
from typing import Tuple, List

from rich.progress import track
import pandas as pd
import os
from PIL import Image
import numpy as np


def load_image_and_mask(image_path: Path, mask_path: Path) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Load an image and its mask, converting mask to list of pixel coordinates"""
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img_arr = np.array(img)

    mask_img = Image.open(mask_path).convert("L")
    mask_arr = np.array(mask_img)
    mask_coords = []
    for y in range(mask_arr.shape[0]):
      for x in range(mask_arr.shape[1]):
        if mask_arr[y, x] != 0:
          mask_coords.append((x, y))
    return img_arr, mask_coords

def calculate_center(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Calculate the center of a list of points"""
    if not points:
      return 0,0
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    return sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)


def compute_amazing_solution(
    part_image_path: Path, gripper_image_path: Path
) -> tuple[float, float, float]:
    """Compute the solution for the given part and gripper images.

    :param part_image_path: Path to the part image
    :param gripper_image_path: Path to the gripper image
    :return: The x, y and angle of the gripper
    """
    part_mask_path = part_image_path.with_name(f"{part_image_path.stem}_mask.png")
    _, part_mask_coords = load_image_and_mask(part_image_path, part_mask_path)
    part_center = calculate_center(part_mask_coords)
    return part_center[0], part_center[1], 0


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
    input_path = Path(args.input).parent
    output_path = Path(args.output).parent

    print(f"Input CSV Path: {args.input}")
    print(f"Input Path: {input_path}")
    print(f"Output Path: {output_path}")
    # compute the solution for each row
    results = []
    for _, row in track(
        input_df.iterrows(),
        description="Computing the solutions for each row",
        total=len(input_df),
    ):
        part_image_path = Path(row["part"])
        gripper_image_path = Path(row["gripper"])
        print(f"Part Image Path: {part_image_path}")
        print(f"Gripper Image Path: {gripper_image_path}")
        assert part_image_path.exists(), f"{part_image_path} does not exist"
        assert gripper_image_path.exists(), f"{gripper_image_path} does not exist"
        x, y, angle = compute_amazing_solution(part_image_path, gripper_image_path)
        results.append([str(part_image_path), str(gripper_image_path), x, y, angle])

    # save the results to the output csv file
    output_df = pd.DataFrame(results, columns=["part", "gripper", "x", "y", "angle"])
    output_df.to_csv(os.path.join(output_path,"output.csv"), index=False)


if __name__ == "__main__":
    main()