import cv2
import numpy as np

def get_gripper_dots(png_path):
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read gripper PNG: {png_path}")

    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dot_positions = []
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < 5 or radius > 40:
            continue
        dot_positions.append((int(x), int(y)))

    return dot_positions

def overlay_gripper(
    part_img,
    part_mask,
    gripper_img,
    gripper_mask,
    pixels_per_mm: float = 1.0,
    angle_search_range=range(0, 360, 5),
    x_center: int = None,
    y_center: int = None,
):
    """
    :param part_img: BGR image of the part
    :param part_mask: GRAYSCALE mask for the part (255 = solid, 0 = hole)
    :param gripper_img: BGR image for the gripper
    :param gripper_mask: GRAYSCALE mask for the gripper (255=gripper, 0=background)
    :param pixels_per_mm: Uniform scale factor for both images (1 => no scale)
    :param angle_search_range: angles to try, e.g. range(0,360,5) or np.arange(...)
    :return: (result_image, chosen_angle)
    """
    # 1) Scale part & gripper by the same factor (if pixels_per_mm != 1)
    part_img_s, part_mask_s = _scale_by_factor(part_img, part_mask, pixels_per_mm)
    gripper_img_s, gripper_mask_s = _scale_by_factor(gripper_img, gripper_mask, pixels_per_mm)

    # 2) Find an angle that avoids holes -> Naive approach
    best_alpha = None
    for alpha in angle_search_range:
        if not _intersects_holes(part_mask_s, gripper_mask_s, alpha, x_center, y_center):
            best_alpha = alpha
            break

    if best_alpha is None:
        print("Warning: No angle found that avoids holes. Using alpha=0.")
        best_alpha = 0

    # 3) Rotate the color gripper
    rotated_gripper = _rotate_image(gripper_img_s, best_alpha)
    result_img = _overlay_black_transparent(part_img_s.copy(), rotated_gripper, x_center, y_center)

    return (result_img, best_alpha)


def _scale_by_factor(color_img, mask_img, pixels_per_mm: float):
    """
    Uniformly scale both a color image and its mask by 'pixels_per_mm'.
    If pixels_per_mm=1, no scale is applied.
    Returns (scaled_color, scaled_mask).
    """
    if pixels_per_mm == 1.0:
        return color_img, mask_img  

    oh, ow = color_img.shape[:2]

    new_w = int(round(ow * pixels_per_mm))
    new_h = int(round(oh * pixels_per_mm))

    scaled_color = cv2.resize(color_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scaled_mask = cv2.resize(mask_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return scaled_color, scaled_mask


def _intersects_holes(part_mask, gripper_mask, angle_deg, x_center, y_center):
    """
    Rotate 'gripper_mask' by angle_deg, place its center at (x_center, y_center)
    in 'part_mask', then check if any (mask=255) in the gripper 
    overlaps a hole (mask=0) in the part.
    """
    rotated = _rotate_image(gripper_mask, angle_deg)

    gh, gw = rotated.shape[:2]
    ph, pw = part_mask.shape[:2]

    top_left_x = x_center - gw // 2
    top_left_y = y_center - gh // 2

    roi_x1 = max(0, top_left_x)
    roi_y1 = max(0, top_left_y)
    roi_x2 = min(pw, top_left_x + gw)
    roi_y2 = min(ph, top_left_y + gh)

    if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
        return False

    part_roi = part_mask[roi_y1:roi_y2, roi_x1:roi_x2]

    gx1 = roi_x1 - top_left_x
    gy1 = roi_y1 - top_left_y
    gx2 = roi_x2 - top_left_x
    gy2 = roi_y2 - top_left_y
    grip_roi = rotated[gy1:gy2, gx1:gx2]

    overlap = np.logical_and(grip_roi == 255, part_roi == 0)
    return bool(np.any(overlap))


def _rotate_image(gray_or_bgr_img, angle_deg):
    """
    Rotates an image (grayscale or BGR) around its center by angle_deg 
    (in degrees), clockwise.
    """
    h, w = gray_or_bgr_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    rotated = cv2.warpAffine(
        gray_or_bgr_img, 
        M, 
        (w, h), 
        flags=cv2.INTER_LINEAR, 
        borderValue=0  # fill outside with black
    )
    return rotated


def _overlay_black_transparent(part_img, gripper_img, x_center, y_center):
    """
    Places 'gripper_img' at (x_center, y_center) in 'part_img',
    treating black pixels (val <10) in the gripper as transparent.
    """
    gh, gw = gripper_img.shape[:2]
    ph, pw = part_img.shape[:2]

    top_left_x = x_center - gw // 2
    top_left_y = y_center - gh // 2

    roi_x1 = max(0, top_left_x)
    roi_y1 = max(0, top_left_y)
    roi_x2 = min(pw, top_left_x + gw)
    roi_y2 = min(ph, top_left_y + gh)

    if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
        return part_img  

    part_roi = part_img[roi_y1:roi_y2, roi_x1:roi_x2]
    gx1 = roi_x1 - top_left_x
    gy1 = roi_y1 - top_left_y
    gx2 = roi_x2 - top_left_x
    gy2 = roi_y2 - top_left_y
    gr_crop = gripper_img[gy1:gy2, gx1:gx2]

    if len(gr_crop.shape) == 3 and gr_crop.shape[2] == 3:
        gray_gr = cv2.cvtColor(gr_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray_gr = gr_crop  # grayscale already

    _, mask = cv2.threshold(gray_gr, 10, 255, cv2.THRESH_BINARY)

    if len(gr_crop.shape) == 2:
        # The gripper is grayscale
        part_roi[:] = np.where(mask[..., None] == 255, gr_crop[..., None], part_roi)
    else:
        for c in range(3):
            part_roi[:, :, c] = np.where(
                mask == 255,
                gr_crop[:, :, c],
                part_roi[:, :, c]
            )

    return part_img
