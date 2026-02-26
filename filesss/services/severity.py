import cv2
import numpy as np


def estimate_severity(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return {"infected_percentage": 0.0, "severity_level": "Unknown"}

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([15, 60, 40], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)

    lower_brown = np.array([5, 40, 20], dtype=np.uint8)
    upper_brown = np.array([20, 255, 200], dtype=np.uint8)

    lower_dark = np.array([0, 0, 0], dtype=np.uint8)
    upper_dark = np.array([180, 255, 55], dtype=np.uint8)

    unhealthy_mask = (
        cv2.inRange(hsv, lower_yellow, upper_yellow)
        | cv2.inRange(hsv, lower_brown, upper_brown)
        | cv2.inRange(hsv, lower_dark, upper_dark)
    )

    saturation_mask = hsv[:, :, 1] > 25
    value_mask = hsv[:, :, 2] > 20
    leaf_mask = saturation_mask & value_mask

    leaf_pixels = int(np.count_nonzero(leaf_mask))
    if leaf_pixels == 0:
        return {"infected_percentage": 0.0, "severity_level": "Unknown"}

    infected_pixels = int(np.count_nonzero((unhealthy_mask > 0) & leaf_mask))
    infected_percentage = (infected_pixels / leaf_pixels) * 100.0

    if infected_percentage <= 10:
        severity_level = "Mild"
    elif infected_percentage <= 30:
        severity_level = "Moderate"
    else:
        severity_level = "Severe"

    return {
        "infected_percentage": round(float(infected_percentage), 2),
        "severity_level": severity_level,
    }
