import os
import cv2
import json
import math
import scipy
import random

import statistics
import numpy as np
from typing import List, Tuple
import numpy.typing as npt
from math import ceil
import matplotlib.pyplot as plt
from tps import ThinPlateSpline
from Lib.Data import (
    LayoutDetectionConfig,
    Line,
    BBox,
    LineData,
    OCRConfig,
    LineDetectionConfig,
    LineXMLData,
)
from datetime import datetime
from xml.dom import minidom


def show_image(
    image: npt.NDArray, cmap: str = "", axis="off", fig_x: int = 24, fix_y: int = 13
) -> None:
    plt.figure(figsize=(fig_x, fix_y))
    plt.axis(axis)

    if cmap != "":
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)


def show_overlay(
    image: npt.NDArray,
    mask: npt.NDArray,
    alpha=0.4,
    axis="off",
    fig_x: int = 24,
    fix_y: int = 13,
):
    plt.figure(figsize=(fig_x, fix_y))
    plt.axis(axis)
    plt.imshow(image)
    plt.imshow(mask, alpha=alpha)


def get_utc_time():
    utc_time = datetime.now()
    utc_time = utc_time.strftime("%Y-%m-%dT%H:%M:%S")

    return utc_time


def get_file_name(file_path: str) -> str:
    name_segments = os.path.basename(file_path).split(".")[:-1]
    name = "".join(f"{x}." for x in name_segments)
    return name.rstrip(".")


def create_dir(dir_name: str) -> None:
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            print(f"Created directory at  {dir_name}")
        except IOError as e:
            print(f"Failed to create directory at: {dir_name}, {e}")


def get_random_color() -> Tuple[int, int, int]:
    rand_r = random.randint(10, 255)
    rand_g = random.randint(10, 200)
    rand_b = random.randint(10, 220)
    color = (rand_r, rand_g, rand_b)

    return color


def resize_to_height(image, target_height: int) -> Tuple[npt.NDArray, float]:
    scale_ratio = target_height / image.shape[0]
    image = cv2.resize(
        image,
        (int(image.shape[1] * scale_ratio), target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return image, scale_ratio


def resize_to_width(image, target_width: int = 2048) -> Tuple[npt.NDArray, float]:
    scale_ratio = target_width / image.shape[1]
    image = cv2.resize(
        image,
        (target_width, int(image.shape[0] * scale_ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    return image, scale_ratio


def binarize(
    image: npt.ArrayLike,
    adaptive: bool = True,
    block_size: int = 51,
    c: int = 13,
    denoise: bool = False,
) -> npt.NDArray:

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if denoise:
        image = cv2.GaussianBlur(image, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(10, 10))
        image = clahe.apply(image)
        image = cv2.fastNlMeansDenoising(image, None, 20, 4, 21)

    if adaptive:
        image = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c,
        )

    else:
        _, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def get_paddings(image: npt.NDArray, patch_size: int = 512) -> Tuple[int, int]:
    max_x = ceil(image.shape[1] / patch_size) * patch_size
    max_y = ceil(image.shape[0] / patch_size) * patch_size
    pad_x = max_x - image.shape[1]
    pad_y = max_y - image.shape[0]

    return pad_x, pad_y


def tile_image(padded_img: npt.NDArray, patch_size: int = 512):
    x_steps = int(padded_img.shape[1] / patch_size)
    y_steps = int(padded_img.shape[0] / patch_size)
    y_splits = np.split(padded_img, y_steps, axis=0)

    patches = [np.split(x, x_steps, axis=1) for x in y_splits]
    patches = [x for xs in patches for x in xs]

    return patches, y_steps


def stitch_predictions(prediction: npt.NDArray, y_steps: int) -> npt.NDArray:
    pred_y_split = np.split(prediction, y_steps, axis=0)
    x_slices = [np.hstack(x) for x in pred_y_split]
    concat_img = np.vstack(x_slices)

    return concat_img


def pad_image(
    image: npt.NDArray, pad_x: int, pad_y: int, pad_value: int = 0
) -> npt.NDArray:
    padded_img = np.pad(
        image,
        pad_width=((0, pad_y), (0, pad_x), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )

    return padded_img


def preprocess_image(
    image: npt.NDArray,
    patch_size: int = 512,
    clamp_width: int = 4096,
    clamp_height: int = 2048,
    clamp_size: bool = True,
):
    """
    Some dimension checking and resizing to avoid very large inputs on which the line(s) on the resulting tiles could be too big and cause troubles with the current line model.
    """
    if clamp_size and image.shape[1] > image.shape[0] and image.shape[1] > clamp_width:
        image, _ = resize_to_width(image, clamp_width)

    elif (
        clamp_size and image.shape[0] > image.shape[1] and image.shape[0] > clamp_height
    ):
        image, _ = resize_to_height(image, clamp_height)

    elif image.shape[0] < patch_size:
        image, _ = resize_to_height(image, patch_size)

    pad_x, pad_y = get_paddings(image, patch_size)
    padded_img = pad_image(image, pad_x, pad_y, pad_value=255)

    return padded_img, pad_x, pad_y


def normalize(image: npt.NDArray) -> npt.NDArray:
    image = image.astype(np.float32)
    image /= 255.0
    return image


def get_contours(image: npt.NDArray) -> list:
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))


def get_rotation_angle_from_lines(
    line_mask: npt.NDArray,
    max_angle: float = 5.0,
    debug_angles: bool = False,
) -> float:
    contours = get_contours(line_mask)
    mask_threshold = (line_mask.shape[0] * line_mask.shape[1]) * 0.001
    contours = [x for x in contours if cv2.contourArea(x) > mask_threshold]
    angles = [cv2.minAreaRect(x)[2] for x in contours]

    low_angles = [x for x in angles if abs(x) != 0.0 and x < max_angle]
    high_angles = [x for x in angles if abs(x) != 90.0 and x > (90 - max_angle)]

    if len(low_angles) > len(high_angles) and len(low_angles) > 0:
        mean_angle = float(np.mean(low_angles))

    # check for clockwise rotation
    elif len(high_angles) > 0:
        mean_angle = float(-(90 - np.mean(high_angles)))

    else:
        mean_angle = 0.0

    return mean_angle


def get_rotation_angle_from_houghlines(
    image: npt.NDArray, min_length: int = 80
) -> float:
    clahe = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(8, 8))
    cl_img = clahe.apply(image)
    blurred = cv2.GaussianBlur(cl_img, (13, 13), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 11
    )

    lines = cv2.HoughLinesP(
        thresh, 1, np.pi / 180, threshold=130, minLineLength=min_length, maxLineGap=8
    )

    if lines is None or len(lines) == 0:
        return image, 0

    angles = []
    zero_angles = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi

        if 5 > abs(angle) > 0:
            angles.append(angle)

        elif int(angle) == 0:
            zero_angles.append(angle)

    if len(angles) != 0:
        avg_angle = statistics.median(angles)
        ratio = len(zero_angles) / len(angles)

        if ratio < 0.5:
            rot_angle = avg_angle
        elif 0.5 < ratio < 0.9:
            rot_angle = avg_angle / 2
        else:
            rot_angle = 0.0
    else:
        rot_angle = 0

    return rot_angle


def rotate_from_angle(image: npt.NDArray, angle: float) -> npt.NDArray:
    rows, cols = image.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    rotated_img = cv2.warpAffine(image, rot_matrix, (cols, rows), borderValue=(0, 0, 0))

    return rotated_img


def get_global_center(slice_image: npt.NDArray, start_x: int, bbox_y: int):
    """
    Transfers the coordinates of a 'local' bbox taken from a line back to the image space
    """
    contours, _ = cv2.findContours(slice_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(x) for x in contours]
    biggest_idx = areas.index(max(areas))
    biggest_contour = contours[biggest_idx]
    _, _, _, bbox_h = cv2.boundingRect(biggest_contour)
    center, _, _ = cv2.minAreaRect(biggest_contour)

    center_x = int(center[0])
    center_y = int(center[1])

    global_x = start_x + center_x
    global_y = bbox_y + center_y

    return global_x, global_y, bbox_h


def check_line_tps(image: npt.NDArray, contour: npt.NDArray, slice_width: int = 40):

    mask = np.zeros(image.shape, dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(contour)

    cv2.drawContours(mask, [contour], contourIdx=0, color=(255, 255, 255), thickness=-1)

    slice1_start_x = x
    slice1_end_x = x + slice_width

    slice2_start_x = x + w // 4 - slice_width
    slice2_end_x = x + w // 4

    slice3_start_x = x + w // 2
    slice3_end_x = x + w // 2 + slice_width

    slice4_start_x = x + w // 2 + w // 4
    slice4_end_x = x + w // 2 + (w // 4) + slice_width

    slice5_start_x = x + w - slice_width
    slice5_end_x = x + w

    # define slices along the bbox from left to right
    slice_1 = mask[y: y + h, slice1_start_x:slice1_end_x, 0]
    slice_2 = mask[y: y + h, slice2_start_x:slice2_end_x, 0]
    slice_3 = mask[y: y + h, slice3_start_x:slice3_end_x, 0]
    slice_4 = mask[y: y + h, slice4_start_x:slice4_end_x, 0]
    slice_5 = mask[y: y + h, slice5_start_x:slice5_end_x, 0]

    slice1_center_x, slice1_center_y, bbox1_h = get_global_center(
        slice_1, slice1_start_x, y
    )
    slice2_center_x, slice2_center_y, bbox2_h = get_global_center(
        slice_2, slice2_start_x, y
    )
    slice3_center_x, slice3_center_y, bbox3_h = get_global_center(
        slice_3, slice3_start_x, y
    )
    slice4_center_x, slice4_center_y, bbox4_h = get_global_center(
        slice_4, slice4_start_x, y
    )
    slice5_center_x, slice5_center_y, bbox5_h = get_global_center(
        slice_5, slice5_start_x, y
    )

    all_bboxes = [bbox1_h, bbox2_h, bbox3_h, bbox4_h, bbox5_h]
    all_centers = [
        slice1_center_y,
        slice2_center_y,
        slice3_center_y,
        slice4_center_y,
        slice5_center_y,
    ]

    min_value = min(all_centers)
    max_value = max(all_centers)
    max_ydelta = max_value - min_value
    mean_bbox_h = np.mean(all_bboxes)
    mean_center_y = np.mean(all_centers)

    if max_ydelta > mean_bbox_h:
        target_y = round(mean_center_y)

        input_pts = [
            [slice1_center_y, slice1_center_x],
            [slice2_center_y, slice2_center_x],
            [slice3_center_y, slice3_center_x],
            [slice4_center_y, slice4_center_x],
            [slice5_center_y, slice5_center_x],
        ]

        output_pts = [
            [target_y, slice1_center_x],
            [target_y, slice2_center_x],
            [target_y, slice3_center_x],
            [target_y, slice4_center_x],
            [target_y, slice5_center_x],
        ]

        return True, input_pts, output_pts, max_ydelta
    else:
        return False, None, None, 0.0


def run_tps(image: npt.NDArray, input_pts, output_pts, add_corners=True, alpha=0.5):

    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        height, width, _ = image.shape

    input_pts = np.array(input_pts)
    output_pts = np.array(output_pts)

    if add_corners:
        corners = np.array(  # Add corners ctrl points
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )

        corners *= [height, width]
        corners *= [height, width]

        input_pts = np.concatenate((input_pts, corners))
        output_pts = np.concatenate((output_pts, corners))

    # Fit the thin plate spline from output to input
    tps = ThinPlateSpline(alpha)
    tps.fit(input_pts, output_pts)

    # Create the 2d meshgrid of indices for output image
    output_indices = np.indices((height, width), dtype=np.float64).transpose(
        1, 2, 0
    )  # Shape: (H, W, 2)

    # Transform it into the input indices
    input_indices = tps.transform(output_indices.reshape(-1, 2)).reshape(
        height, width, 2
    )

    # Interpolate the resulting image
    warped = np.concatenate(
        [
            scipy.ndimage.map_coordinates(
                image[..., channel], input_indices.transpose(2, 0, 1)
            )[..., None]
            for channel in (0, 1, 2)
        ],
        axis=-1,
    )

    return warped


def check_for_tps(image: npt.NDArray, line_contours: List[npt.NDArray]):
    line_data = []
    for _, line_cnt in enumerate(line_contours):

        _, y, _, _ = cv2.boundingRect(line_cnt)
        # TODO: store input and output points to avoid running that step twice
        tps_status, input_pts, output_pts, max_yd = check_line_tps(image, line_cnt)

        line = {
            "contour": line_cnt,
            "tps": tps_status,
            "input_pts": input_pts,
            "output_pts": output_pts,
            "max_yd": max_yd,
        }

        line_data.append(line)

    do_tps = [x["tps"] for x in line_data if x["tps"] is True]
    ratio = len(do_tps) / len(line_contours)

    return ratio, line_data


def get_global_tps_line(line_data: dict):
    """
    A simple approach to the most representative curved line in the image assuming that the overall distortion is relatively uniform
    """
    all_y_deltas = []

    for line in line_data:
        if line["tps"] is True:
            all_y_deltas.append(line["max_yd"])
        else:
            all_y_deltas.append(0.0)

    mean_delta = np.mean(all_y_deltas)
    best_diff = max(all_y_deltas)  # just set it to the highest value
    best_y = None

    for yd in all_y_deltas:
        if yd > 0:
            delta = abs(mean_delta - yd)
            if delta < best_diff:
                best_diff = delta
                best_y = yd

    target_idx = all_y_deltas.index(best_y)

    return target_idx


def get_line_images_via_local_tps(
    image: npt.NDArray, line_data: list, k_factor: float = 1.7
):

    default_k_factor = k_factor
    current_k = default_k_factor
    line_images = []

    for line in line_data:
        if line["tps"] is True:
            output_pts = line["output_pts"]
            input_pts = line["input_pts"]

            assert input_pts is not None and output_pts is not None

            tmp_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.drawContours(tmp_mask, [line["contour"]], -1, (255, 255, 255), -1)

            # TODO: check channel dim here..
            warped_img = run_tps(image, output_pts, input_pts)
            warped_mask = run_tps(tmp_mask, output_pts, input_pts)

            _, _, _, bbox_h = cv2.boundingRect(line["contour"])

            line_img, adapted_k = get_line_image(
                warped_img, warped_mask, bbox_h, bbox_tolerance=2.0, k_factor=current_k
            )
            line_images.append(line_img)

            if current_k != adapted_k:
                current_k = adapted_k

        else:
            tmp_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.drawContours(tmp_mask, [line["contour"]], -1, (255, 255, 255), -1)

            _, _, _, h = cv2.boundingRect(line["contour"])
            line_img, adapted_k = get_line_image(
                image, tmp_mask, h, bbox_tolerance=2.0, k_factor=current_k
            )
            line_images.append(line_img)

    return line_images


def apply_global_tps(image: npt.NDArray, line_mask: npt.NDArray, line_data: List):
    best_idx = get_global_tps_line(line_data)
    output_pts = line_data[best_idx]["output_pts"]
    input_pts = line_data[best_idx]["input_pts"]

    assert input_pts is not None and output_pts is not None

    warped_img = run_tps(image, output_pts, input_pts)
    warped_mask = run_tps(line_mask, output_pts, input_pts)

    return warped_img, warped_mask


def mask_n_crop(image: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    image = image.astype(np.uint8)
    mask = mask.astype(np.uint8)

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    image_masked = cv2.bitwise_and(image, image, mask, mask)
    image_masked = np.delete(
        image_masked, np.where(~image_masked.any(axis=1))[0], axis=0
    )
    image_masked = np.delete(
        image_masked, np.where(~image_masked.any(axis=0))[0], axis=1
    )

    return image_masked


def get_line_threshold(line_prediction: npt.NDArray, slice_width: int = 20):
    """
    This function generates n slices (of n = steps) width the width of slice_width across the bbox of the detected lines.
    The slice with the max. number of contained contours is taken to be the canditate to calculate the bbox center of each contour and
    take the median distance between each bbox center as estimated line cut-off threshold to sort each line segment across the horizontal

    Note: This approach might turn out to be problematic in case of sparsely spread line segments across a page
    """

    if len(line_prediction.shape) == 3:
        line_prediction = cv2.cvtColor(line_prediction, cv2.COLOR_BGR2GRAY)

    x, y, w, h = cv2.boundingRect(line_prediction)
    x_steps = (w // slice_width) // 2

    bbox_numbers = []

    for step in range(1, x_steps + 1):
        x_offset = x_steps * step
        x_start = x + x_offset
        x_end = x_start + slice_width

        _slice = line_prediction[y : y + h, x_start:x_end]
        contours, _ = cv2.findContours(_slice, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        bbox_numbers.append((len(contours), contours))

    sorted_list = sorted(bbox_numbers, key=lambda x: x[0], reverse=True)

    if len(sorted_list) > 0:
        reference_slice = sorted_list[0]

        y_points = []
        n_contours, contours = reference_slice

        if n_contours == 0:
            line_threshold = 0.0
        else:
            for _, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                y_center = y + (h // 2)
                y_points.append(y_center)

            line_threshold = float(np.median(y_points) // n_contours)
    else:
        line_threshold = 0.0

    return line_threshold


def sort_bbox_centers(
    bbox_centers: List[Tuple[int, int]], line_threshold: int = 20
) -> List:
    sorted_bbox_centers = []
    tmp_line = []

    for i in range(0, len(bbox_centers)):
        if len(tmp_line) > 0:
            for s in range(0, len(tmp_line)):

                # TODO: refactor this to make this calculation an enum to choose between both methods
                # y_diff = abs(tmp_line[s][1] - bbox_centers[i][1])
                """
                I use the mean of the hitherto present line chunks in tmp_line since
                the precalculated fixed threshold can break the sorting if
                there is some slight bending in the line. This part may need some tweaking after
                some further practical review
                """
                ys = [y[1] for y in tmp_line]
                mean_y = np.mean(ys)
                y_diff = abs(mean_y - bbox_centers[i][1])

                if y_diff > line_threshold:
                    tmp_line.sort(key=lambda x: x[0])
                    sorted_bbox_centers.append(tmp_line.copy())
                    tmp_line.clear()

                    tmp_line.append(bbox_centers[i])
                    break
                else:
                    tmp_line.append(bbox_centers[i])
                    break
        else:
            tmp_line.append(bbox_centers[i])

    sorted_bbox_centers.append(tmp_line)

    for y in sorted_bbox_centers:
        y.sort(key=lambda x: x[0])

    sorted_bbox_centers = list(reversed(sorted_bbox_centers))

    return sorted_bbox_centers


def group_line_chunks(
    sorted_bbox_centers, lines: List[Line], adaptive_grouping: bool = True
):
    new_line_data = []
    for bbox_centers in sorted_bbox_centers:

        if len(bbox_centers) > 1:  # i.e. more than 1 bbox center in a group
            contour_stack = []

            for box_center in bbox_centers:
                for line_data in lines:
                    if box_center == line_data.center:
                        contour_stack.append(line_data.contour)
                        break

            if adaptive_grouping:
                for contour in contour_stack:
                    x, y, w, h = cv2.boundingRect(contour)
                    width_offset = int(w * 0.05)
                    height_offset = int(h * 0.05)
                    w += width_offset
                    h += height_offset

            stacked_contour = np.vstack(contour_stack)
            stacked_contour = cv2.convexHull(stacked_contour)

            # TODO: both calls necessary?
            x, y, w, h = cv2.boundingRect(stacked_contour)
            _, _, angle = cv2.minAreaRect(stacked_contour)

            _bbox = BBox(x, y, w, h)
            x_center = _bbox.x + (_bbox.w // 2)
            y_center = _bbox.y + (_bbox.h // 2)

            new_line = Line(
                contour=stacked_contour,
                bbox=_bbox,
                center=(x_center, y_center),
                angle=angle,
            )

            new_line_data.append(new_line)

        else:
            for _bcenter in bbox_centers:
                for line_data in lines:
                    if _bcenter == line_data.center:
                        new_line_data.append(line_data)
                        break

    return new_line_data


def sort_lines_by_threshold2(
    line_mask: npt.NDArray,
    lines: List[Line],
    threshold: int = 20,
    calculate_threshold: bool = True,
    group_lines: bool = True,
    debug: bool = False,
):

    bbox_centers = [x.center for x in lines]

    if calculate_threshold:
        line_treshold = get_line_threshold(line_mask)
    else:
        line_treshold = threshold

    if debug:
        print(f"Line threshold: {threshold}")

    sorted_bbox_centers = sort_bbox_centers(bbox_centers, line_threshold=line_treshold)

    if debug:
        print(sorted_bbox_centers)

    if group_lines:
        new_lines = group_line_chunks(sorted_bbox_centers, lines)
    else:
        _bboxes = [x for xs in sorted_bbox_centers for x in xs]

        new_lines = []
        for _bbox in _bboxes:
            for _line in lines:
                if _bbox == _line.center:
                    new_lines.append(_line)

    return new_lines, line_treshold


def filter_line_contours(
    image: npt.NDArray, line_contours, threshold: float = 0.01
) -> List:
    filtered_contours = []
    for _, line_cnt in enumerate(line_contours):

        _, _, w, h = cv2.boundingRect(line_cnt)

        if w > image.shape[1] * threshold and h > 10:
            filtered_contours.append(line_cnt)

    return filtered_contours


def build_raw_line_data(image: npt.NDArray, line_mask: npt.NDArray):

    if len(line_mask.shape) == 3:
        line_mask = cv2.cvtColor(line_mask, cv2.COLOR_BGR2GRAY)

    angle = get_rotation_angle_from_lines(line_mask)
    rot_mask = rotate_from_angle(line_mask, angle)
    rot_img = rotate_from_angle(image, angle)

    line_contours = get_contours(rot_mask)
    line_contours = [x for x in line_contours if cv2.contourArea(x) > 10]

    rot_mask = cv2.cvtColor(rot_mask, cv2.COLOR_GRAY2RGB)

    return rot_img, rot_mask, line_contours, angle


def build_line_data(contour: npt.NDArray) -> Line:
    _, _, angle = cv2.minAreaRect(contour)
    x, y, w, h = cv2.boundingRect(contour)
    x_center = x + (w // 2)
    y_center = y + (h // 2)

    bbox = BBox(x, y, w, h)
    return Line(contour, bbox, (x_center, y_center), angle)


def get_text_bbox(lines: List[Line]):
    all_bboxes = [x.bbox for x in lines]
    min_x = min(a.x for a in all_bboxes)
    min_y = min(a.y for a in all_bboxes)

    max_w = max(a.w for a in all_bboxes)
    max_h = all_bboxes[-1].y + all_bboxes[-1].h

    bbox = BBox(min_x, min_y, max_w, max_h)

    return bbox


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def rotate_contour(cnt, center_x: int, center_y: int, angle: float):
    cnt_norm = cnt - [center_x, center_y]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [center_x, center_y]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated


def optimize_countour(cnt, e=0.001):
    epsilon = e * cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, epsilon, True)


def pad_to_width(
    img: npt.NDArray, target_width: int, target_height: int, padding: str
) -> npt.NDArray:

    if len(img.shape) == 3:
        _, _, channels = img.shape
    else:
        channels = 1

    tmp_img, _ = resize_to_width(img, target_width)

    height = tmp_img.shape[0]
    middle = (target_height - tmp_img.shape[0]) // 2

    if padding == "white":
        upper_stack = np.ones(shape=(middle, target_width, channels), dtype=np.uint8)
        lower_stack = np.ones(
            shape=(target_height - height - middle, target_width, channels),
            dtype=np.uint8,
        )

        upper_stack *= 255
        lower_stack *= 255
    else:
        upper_stack = np.zeros(shape=(middle, target_width, channels), dtype=np.uint8)
        lower_stack = np.zeros(
            shape=(target_height - height - middle, target_width, channels),
            dtype=np.uint8,
        )

    out_img = np.vstack([upper_stack, tmp_img, lower_stack])

    return out_img


def pad_to_height(
    img: npt.NDArray, target_width: int, target_height: int, padding: str
) -> npt.NDArray:
    height, _, channels = img.shape
    tmp_img, ratio = resize_to_height(img, target_height)

    width = tmp_img.shape[1]
    middle = (target_width - width) // 2

    if padding == "white":
        left_stack = np.ones(shape=(target_height, middle, channels), dtype=np.uint8)
        right_stack = np.ones(
            shape=(target_height, target_width - width - middle, channels),
            dtype=np.uint8,
        )

        left_stack *= 255
        right_stack *= 255

    else:
        left_stack = np.zeros(shape=(target_height, middle, channels), dtype=np.uint8)
        right_stack = np.zeros(
            shape=(target_height, target_width - width - middle, channels),
            dtype=np.uint8,
        )

    out_img = np.hstack([left_stack, tmp_img, right_stack])

    return out_img


"""
These are basically the two step inbetween the raw line prediction and the OCR pass
"""


def get_line_data(
    image: npt.NDArray, line_mask: npt.NDArray, group_chunks: bool = True
) -> LineData:
    angle = get_rotation_angle_from_lines(line_mask)

    rot_mask = rotate_from_angle(line_mask, angle)
    rot_img = rotate_from_angle(image, angle)

    line_contours = get_contours(rot_mask)
    line_data = [build_line_data(x) for x in line_contours]
    line_data = [x for x in line_data if x.bbox.h > 10]
    sorted_lines, _ = sort_lines_by_threshold2(
        rot_mask, line_data, group_lines=group_chunks
    )

    data = LineData(rot_img, rot_mask, angle, sorted_lines)

    return data


def extract_line(
    image: npt.NDArray, mask: npt.NDArray, bbox_h: int, k_factor: float = 1.2
) -> npt.NDArray:
    iterations = 2
    k_size = int(bbox_h * k_factor)
    morph_multiplier = k_factor

    morph_rect = cv2.getStructuringElement(
        shape=cv2.MORPH_RECT, ksize=(k_size, int(k_size * morph_multiplier))
    )
    iterations = 1
    dilated_mask = cv2.dilate(mask, kernel=morph_rect, iterations=iterations)
    masked_line = mask_n_crop(image, dilated_mask)

    return masked_line


def get_line_image(
    image: npt.NDArray,
    mask: npt.NDArray,
    bbox_h: int,
    bbox_tolerance: float = 2.5,
    k_factor: float = 1.2,
)->Tuple[npt.NDArray, float]:
    tmp_k = k_factor
    line_img = extract_line(image, mask, bbox_h, k_factor=tmp_k)

    while line_img.shape[0] > bbox_h * bbox_tolerance:
        tmp_k = tmp_k - 0.1
        line_img = extract_line(image, mask, bbox_h, k_factor=tmp_k)

    return line_img, tmp_k


def extract_line_images(
    image: npt.NDArray,
    line_data: List[npt.NDArray],
    default_k: float = 1.7,
    bbox_tolerance: float = 2.5,
):
    default_k_factor = default_k
    current_k = default_k_factor

    current_k = default_k_factor
    line_images = []

    for _, line in enumerate(line_data):
        _, _, _, h = cv2.boundingRect(line.contour)
        tmp_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.drawContours(tmp_mask, [line.contour], -1, (255, 255, 255), -1)

        line_img, adapted_k = get_line_image(
            image, tmp_mask, h, bbox_tolerance=bbox_tolerance, k_factor=current_k
        )
        line_images.append(line_img)

        if current_k != adapted_k:
            current_k = adapted_k

    return line_images


def get_charset(charset: str) -> List[str]:
    if isinstance(charset, str):
        charset = [x for x in charset]

    elif isinstance(charset, List):
        charset = charset

    return [x for x in charset]


def read_ocr_model_config(config_file: str):
    model_dir = os.path.dirname(config_file)
    file = open(config_file, encoding="utf-8")
    json_content = json.loads(file.read())

    onnx_model_file = f"{model_dir}/{json_content['onnx-model']}"

    input_width = json_content["input_width"]
    input_height = json_content["input_height"]
    input_layer = json_content["input_layer"]
    output_layer = json_content["output_layer"]
    squeeze_channel_dim = (
        True if json_content["squeeze_channel_dim"] == "yes" else False
    )
    swap_hw = True if json_content["swap_hw"] == "yes" else False
    characters = get_charset(json_content["charset"])

    config = OCRConfig(
        onnx_model_file,
        input_width,
        input_height,
        input_layer,
        output_layer,
        squeeze_channel_dim,
        swap_hw,
        characters,
    )

    return config


def read_line_model_config(config_file: str) -> LineDetectionConfig:
    model_dir = os.path.dirname(config_file)
    file = open(config_file, encoding="utf-8")
    json_content = json.loads(file.read())

    onnx_model_file = f"{model_dir}/{json_content['onnx-model']}"
    patch_size = int(json_content["patch_size"])

    config = LineDetectionConfig(onnx_model_file, patch_size)

    return config


def read_layout_model_config(config_file: str) -> LayoutDetectionConfig:
    model_dir = os.path.dirname(config_file)
    file = open(config_file, encoding="utf-8")
    json_content = json.loads(file.read())

    onnx_model_file = f"{model_dir}/{json_content['onnx-model']}"
    patch_size = int(json_content["patch_size"])
    classes = json_content["classes"]

    config = LayoutDetectionConfig(onnx_model_file, patch_size, classes)

    return config


"""
functions for ocr datset creation from XML
"""


def preprocess_img(image: npt.NDArray) -> npt.NDArray:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(24, 24))
    bw = clahe.apply(image)
    bw = cv2.adaptiveThreshold(
        bw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 11
    )

    thresh_c = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)

    return thresh_c


def parse_labels(textlines: List):
    line_data = []

    for text_line_idx in range(len(textlines)):
        label = textlines[text_line_idx].getElementsByTagName("Unicode")
        line_id = textlines[text_line_idx].attributes["id"].value
        label = label[0].firstChild.nodeValue
        box_coords = textlines[text_line_idx].getElementsByTagName("Coords")
        img_box = box_coords[0].attributes["points"].value
        box_coordinates = img_box.split(" ")
        box_coordinates = [x for x in box_coordinates if x != ""]

        z = []
        for c in box_coordinates:
            x, y = c.split(",")
            a = [int(x), int(y)]
            z.append(a)

        pts = np.array(z, dtype=np.int32)

        xml_data = LineXMLData(id=line_id, points=pts, label=label)

        line_data.append(xml_data)

    return line_data


def generate_line_info(annotation_file: str) -> List[LineXMLData]:
    annotation_tree = minidom.parse(annotation_file)
    textlines = annotation_tree.getElementsByTagName("TextLine")
    lines_info = parse_labels(textlines)

    return lines_info


def generate_line_image(
    image: npt.NDArray, contour: npt.NDArray, kernel: Tuple = (10, 14), kernel_iterations: int = 4
) -> npt.NDArray:
    image_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    cv2.drawContours(
        image_mask, [contour], contourIdx=-1, color=(255, 255, 255), thickness=-1
    )

    dilate_k = np.ones(kernel, dtype=np.uint8)

    image_mask = cv2.dilate(image_mask, dilate_k, iterations=kernel_iterations)
    image_masked = cv2.bitwise_and(image, image, mask=image_mask)

    cropped_img = np.delete(
        image_masked, np.where(~image_masked.any(axis=1))[0], axis=0
    )
    cropped_img = np.delete(cropped_img, np.where(~cropped_img.any(axis=0))[0], axis=1)

    return cropped_img


def save_line_transcription(
    file_name: str,
    image: npt.NDArray,
    line_id: str,
    label: str,
    images_out_path: str,
    labels_out_path: str,
):

    line_file = os.path.join(labels_out_path, f"{file_name}_{line_id}.txt")

    with open(line_file, "w", encoding="utf-8") as f:
        f.write(label)

    target_img_file = os.path.join(images_out_path, f"{file_name}_{line_id}.jpg")
    cv2.imwrite(target_img_file, image)


def create_dataset(
    image_file: str,
    xml_file: str,
    images_out_path: str,
    labels_out_path: str,
    binarize_image: bool = True,
    kernel_iterations: int = 6,
):
    file_name = get_file_name(image_file)
    image = cv2.imread(image_file)

    if binarize_image:
        image = preprocess_img(image)

    line_info = generate_line_info(xml_file)

    for xml_info in line_info:
        line_img = generate_line_image(
            image, xml_info.points, kernel_iterations=kernel_iterations
        )
        save_line_transcription(
            file_name,
            line_img,
            xml_info.id,
            xml_info.label,
            images_out_path,
            labels_out_path,
        )
