"""
DataLoader.py
-------------
Generates a tasks.csv file that pairs each part's masks with available grippers.
If a part is missing gripper data, a default gripper is used.
"""

import os
import csv
from pathlib import Path

# Path to a fallback gripper image.
DEFAULT_GRIPPER_PATH = "defaultgripper_2.png"
def gather_all_masks_recursively(part_folder):
    mask_files = []
    for root, dirs, files in os.walk(part_folder):
        for file in files:
            if "mask" in file.lower():
                mask_files.append(os.path.join(root, file))
    return mask_files

def gather_sanity_check_warnings(tasks):
    """
    Collect warnings/errors for the list of [mask_file, gripper_file] pairs.
    Returns a list of warning strings. 
    """
    warnings = []
    if not tasks:
        warnings.append("Error: No tasks were generated.")
        return warnings

    for mask, gripper in tasks:
        if "mask_" not in mask.lower():
            warnings.append(f"Warning: {mask} does not appear to be a proper mask file.")
        if "filtered" in gripper.lower():
            warnings.append(f"Warning: {gripper} appears to be a filtered image.")
    return warnings


# optional function to simulate a tasks.csv file for testing purposes
def generate_tasks_csv(data_dir: Path, output_file: Path, use_default_gripper: bool = True):
    """
    Generate a tasks.csv file by scanning the data directory.

    :param data_dir: Path to the data directory containing part folders
    :param output_file: Path to save the generated tasks.csv file
    """
    tasks = []

    for part_dir in data_dir.iterdir():
        if not part_dir.is_dir() or part_dir.name.startswith("dummy"):
            continue

        # mask_files = sorted(part_dir.glob("mask_*.png"))
        mask_files = sorted(part_dir.rglob("mask_*.png"))

        gripper_files = sorted([
            gf for gf in part_dir.glob("*.png")
            if "filtered" not in gf.name.lower() and "mask_" not in gf.name.lower()
        ])

        # Default gripper: 
        if use_default_gripper:
            gripper_files = [Path(DEFAULT_GRIPPER_PATH)]
            

        for mask_file in mask_files:
            for gripper_file in gripper_files:
                tasks.append([str(mask_file), str(gripper_file)])

    warnings = gather_sanity_check_warnings(tasks)
    for w in warnings:
        print(w)

    if any("Error:" in w for w in warnings):
        print("Sanity check failed. Aborting CSV generation.")
        return

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["part", "gripper"]) 
        writer.writerows(tasks)

    print(f"Tasks CSV generated: {output_file} with {len(tasks)} entries.")


def parse_data(parts_dir: str):
    """
    Example function to parse and categorize parts, grippers, and variations.
    Not strictly required for the pipeline, but can be used for debugging/analysis.
    """
    from pathlib import Path

    def get_file_paths(base_dir, extensions=(".png", ".jpg")):
        """Retrieve all file paths with specific extensions from a directory."""
        file_paths = []
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(extensions):
                    file_paths.append(os.path.join(root, file))
        return file_paths

    part_folders = [os.path.join(parts_dir, f) for f in os.listdir(parts_dir)
                    if os.path.isdir(os.path.join(parts_dir, f)) and f.startswith("part_")]

    data = []
    for part_folder in part_folders:
        grippers = [os.path.join(part_folder, f) for f in os.listdir(part_folder)
                    if f.split('.')[0].isdigit() and f.endswith(".png")]
        
        # masks = [os.path.join(part_folder, f) for f in os.listdir(part_folder)
        #          if "mask" in f.lower()]
        masks = gather_all_masks_recursively(part_folder)
        variations = {
            "positional": get_file_paths(os.path.join(part_folder, "positional_variation")),
            "mirrored": get_file_paths(os.path.join(part_folder, "Mirrored")),
            "inverted_color": get_file_paths(os.path.join(part_folder, "inverted_color"))
        }
        data.append({
            "part_folder": part_folder,
            "grippers": grippers,
            "masks": masks,
            "variations": variations
        })
    return data


data_directory = Path("data")       
output_csv = Path("evaluate/tasks.csv")
generate_tasks_csv(data_directory, output_csv, use_default_gripper=False)
data_summary = parse_data(str(data_directory))
print("======Data Summary======")
for entry in data_summary:
    part_folder = entry["part_folder"]
    num_grippers = len(entry["grippers"])
    num_masks = len(entry["masks"])
    num_positional = len(entry["variations"]["positional"])
    num_mirrored = len(entry["variations"]["mirrored"])
    num_inverted = len(entry["variations"]["inverted_color"])

    print(f"Part Folder: {part_folder}")
    print(f"  - Grippers: {num_grippers}")
    print(f"  - Masks: {num_masks}")
    print(f"  - Positional Variations: {num_positional}")
    print(f"  - Mirrored Variations: {num_mirrored}")
    print(f"  - Inverted Color Variations: {num_inverted}")
    print("")

total_parts = len(data_summary)
total_grippers = sum(len(e["grippers"]) for e in data_summary)
total_masks = sum(len(e["masks"]) for e in data_summary)
print(f"Total part folders: {total_parts}")
print(f"Total grippers: {total_grippers}")
print(f"Total masks: {total_masks}")