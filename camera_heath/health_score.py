def calculate_health_score(
    blur_result,
    brightness_result,
    color_result,
    freeze_result,
    obstruction_result,
    tilt_result,
    fov_result
):

    score = 100

    if blur_result["blur"]:
        score -= 20

    if brightness_result["brightness_change"]:
        score -= 10

    if color_result["color_change"]:
        score -= 10

    if freeze_result["freeze"]:
        score -= 20

    if obstruction_result["obstruction"]:
        score -= 25

    if tilt_result["tilt"]:
        score -= 10

    if fov_result["fov_change"]:
        score -= 5

    if score < 0:
        score = 0

    if score >= 90:
        status = "Excellent"

    elif score >= 75:
        status = "Good"

    elif score >= 50:
        status = "Warning"

    else:
        status = "Critical"

    return {
        "score": score,
        "status": status
    }