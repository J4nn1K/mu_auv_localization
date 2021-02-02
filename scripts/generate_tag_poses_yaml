#!/usr/bin/env python

import yaml
import os
import tf.transformations

X_DIST_BETWEEN_TAGS = 0.25
X_OFFSET = 0.06
Y_DIST_BETWEEN_TAGS = 0.39375
Y_OFFSET = 0.06
Z_OFFSET = -1.3
ROWS = 9
COLS = 7
N_TAGS = ROWS * COLS
SIZE_TAG = 0.096


def generate_even_grid():
    data = {}
    data["tag_poses"] = []
    for i in range(ROWS):
        for j in range(COLS):
            # use gazebo coordinate frame
            tag_id = i * COLS + j
            pos_x = ((COLS - 1) - j) * X_DIST_BETWEEN_TAGS + X_OFFSET
            pos_y = i * Y_DIST_BETWEEN_TAGS + Y_OFFSET
            pos_z = Z_OFFSET
            # order of quaternion is x, y, z, w
            quat = tf.transformations.quaternion_from_euler(0, 0, 0)
            data["tag_poses"].append({
                "frame_id": "map",
                "id": tag_id,
                "size": SIZE_TAG,
                "x": pos_x,
                "y": pos_y,
                "z": pos_z,
                "qx": float(quat[0]),
                "qy": float(quat[1]),
                "qz": float(quat[2]),
                "qw": float(quat[3]),
            })
    return data


def main():
    filename = "tag_poses.yaml"
    with open(filename, "w") as file_handle:
        data = generate_even_grid()
        yaml.dump(data, file_handle)
        print("Created file '{}'".format(os.path.join(os.getcwd(), filename)))
        print("Probably you want to move it to this package's config directory")


if __name__ == "__main__":
    main()
