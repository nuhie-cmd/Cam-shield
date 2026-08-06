import json


def generate_json(
    blur_result,
    brightness_result,
    color_result,
    freeze_result,
    obstruction_result,
    tilt_result,
    fov_result,
    health
):

    output = {

        "blur": blur_result["blur"],

        "brightness_change": brightness_result["brightness_change"],

        "color_change": color_result["color_change"],

        "freeze": freeze_result["freeze"],

        "obstruction": obstruction_result["obstruction"],

        "tilt": tilt_result["tilt"],

        "fov_change": fov_result["fov_change"],

        "camera_health_score": health["score"],

        "camera_status": health["status"]

    }

    return json.dumps(output, indent=4)