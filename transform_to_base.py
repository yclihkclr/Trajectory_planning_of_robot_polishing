import numpy as np
from scipy.spatial.transform import Rotation


# Given transformation from base to object
base_to_object_translation = np.array([1.221890, -0.098359, 0.032958])  # (x, y, z)
base_to_object_rpy = np.array([-0.007415, -0.005143, -1.561820])  # (roll, pitch, yaw)

# Convert RPY to rotation matrix
base_to_object_rotation = Rotation.from_euler('xyz', base_to_object_rpy).as_matrix()

# Construct homogeneous transformation matrix (4x4)
T_base_object = np.eye(4)
T_base_object[:3, :3] = base_to_object_rotation
T_base_object[:3, 3] = base_to_object_translation

# Read the path.txt file
traj_file_name ="full_polishing_path_1_1_7"
poses = []
with open(traj_file_name+".txt", "r") as f:
    for line in f:
        vals = list(map(float, line.strip().split()))
        poses.append(vals)

# Transform poses
transformed_poses = []
for pose in poses:
    x, y, z, ox, oy, oz, ow = pose
    
    # Orientation as rotation matrix
    r_obj = Rotation.from_quat([ox, oy, oz, ow]).as_matrix()

    # Construct homogeneous transform of pose in object frame
    T_obj_pose = np.eye(4)
    T_obj_pose[:3, :3] = r_obj
    T_obj_pose[:3, 3] = [x, y, z]

    # Transform pose to base frame
    T_base_pose = T_base_object @ T_obj_pose

    # Extract transformed position and orientation
    t_base = T_base_pose[:3, 3]
    r_base = Rotation.from_matrix(T_base_pose[:3, :3])
    quat_base = r_base.as_quat()  # returns [x, y, z, w]

    transformed_poses.append([
        t_base[0], t_base[1], t_base[2],
        quat_base[0], quat_base[1], quat_base[2], quat_base[3]
    ])

# Save to new file in required format
tranformed_file_name = traj_file_name + "_transformed.txt"
with open(tranformed_file_name, "w") as f:
    f.write("[\n")
    for i, pose in enumerate(transformed_poses):
        f.write("  [" + ", ".join(f"{v:.6f}" for v in pose) + "]")
        if i < len(transformed_poses) - 1:
            f.write(",\n")
        else:
            f.write("\n")
    f.write("]\n")
