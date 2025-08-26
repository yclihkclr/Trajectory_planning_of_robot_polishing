
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree

def load_mesh_as_pointcloud_with_normals(mesh_path, voxel_size=1.0):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=1000000)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
    return pcd

def get_principal_axes(pcd):
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    return eigvecs, centroid

def generate_zigzag_trajectory(pcd, point_spacing, line_spacing, offset=0.0, thickness=0.005):

    eigvecs, center = get_principal_axes(pcd)
    primary_dir = eigvecs[:, 0]   # slicing direction
    secondary_dir = eigvecs[:, 1] # sweep direction
    normal_dir = eigvecs[:, 2]    # surface normal approx

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    projections = points @ primary_dir
    min_proj, max_proj = projections.min(), projections.max()
    num_slices = int((max_proj - min_proj) / line_spacing) + 1

    fixed_x_axis = primary_dir / np.linalg.norm(primary_dir)  # global X-axis as in primary direction to reduce twist

    all_slices = []

    for i in range(num_slices):
        proj_val = min_proj + i * line_spacing
        slab_mask = np.abs(projections - proj_val) < thickness
        slab_points = points[slab_mask]
        slab_normals = normals[slab_mask]

        if len(slab_points) < 10:
            continue

        # Local 2D frame
        x_axis_local = secondary_dir / np.linalg.norm(secondary_dir)
        y_axis_local = np.cross(primary_dir, x_axis_local)
        y_axis_local /= np.linalg.norm(y_axis_local)
        local_frame = np.stack([x_axis_local, y_axis_local], axis=1)
        local_origin = center

        vecs_to_points = slab_points - local_origin
        local_2d = vecs_to_points @ local_frame
        min_xy = np.min(local_2d, axis=0)
        max_xy = np.max(local_2d, axis=0)

        # ✅ Extend sampling grid slightly beyond object edges, this is to solve edge gap problem
        margin = 1.0 * point_spacing
        x_vals = np.arange(min_xy[0] - margin, max_xy[0] + margin, point_spacing)
        y_vals = np.arange(min_xy[1] - margin, max_xy[1] + margin, point_spacing)

        kdtree = KDTree(slab_points)
        slice_traj = []

        for j, y in enumerate(y_vals):
            line_traj = []
            x_iter = x_vals if j % 2 == 0 else x_vals[::-1]

            for x in x_iter:
                sample_2d = np.array([x, y])
                slice_origin = center + primary_dir * proj_val
                sample_3d = slice_origin + sample_2d @ local_frame.T

                dist, idx = kdtree.query(sample_3d)
                surface_point = slab_points[idx]
                normal = slab_normals[idx]

                z_axis = -normal / np.linalg.norm(normal)

                # ✅ Fixed X-axis logic
                if np.abs(np.dot(z_axis, fixed_x_axis)) > 0.99:  # nearly parallel
                    x_axis = np.array([-1, 0, 0])
                else:
                    x_axis = fixed_x_axis
                y_axis = np.cross(z_axis, x_axis)
                y_axis /= np.linalg.norm(y_axis)
                x_axis = np.cross(y_axis, z_axis)
                x_axis /= np.linalg.norm(x_axis)
                rot = np.column_stack((x_axis, y_axis, z_axis))

                offset_pos = surface_point + offset * z_axis
                line_traj.append((offset_pos, rot))

            slice_traj.extend(line_traj)

        # ✅ Alternate whole slice direction for smooth U-turn stitching
        if i % 2 == 1:
            slice_traj = slice_traj[::-1]

        all_slices.append(slice_traj)

    # Flatten into final trajectory
    trajectory = []
    for s in all_slices:
        trajectory.extend(s)

    return trajectory, secondary_dir

def diagnose_zigzag_behavior(trajectory, secondary_dir, tolerance=0.03):
    """
    trajectory: list of (position, rotation) tuples
    secondary_dir: the direction zigzag should alternate along
    tolerance: max allowed distance between consecutive line ends if zigzag works
    """
    if not trajectory:
        print("Empty trajectory!")
        return

    segment_starts = []
    segment_ends = []
    current_segment = []

    prev_rot = trajectory[0][1]

    for i, (pos, rot) in enumerate(trajectory):
        current_segment.append((pos, rot))
        if i == len(trajectory) - 1 or rot is not trajectory[i+1][1]:
            # new segment
            segment_starts.append(current_segment[0][0])
            segment_ends.append(current_segment[-1][0])
            current_segment = []

    print(f"Detected {len(segment_starts)} lines in the trajectory")

    for i in range(1, len(segment_starts)):
        end_prev = segment_ends[i-1]
        start_curr = segment_starts[i]

        move_vec = start_curr - end_prev
        projection = np.dot(move_vec, secondary_dir)
        dist = np.linalg.norm(move_vec)

        print(f"Line {i}: move {dist:.4f}, projection along zigzag dir {projection:.4f}", end='')

        if abs(projection) > tolerance:
            print("  <-- large jump / not alternating")
        else:
            print("  OK")




def save_trajectory_as_pose_array(trajectory, filename):
    with open(filename, 'w') as f:
        for pos, rot in trajectory:
            pose = np.eye(4)
            pose[:3, :3] = rot
            pose[:3, 3] = pos
            pose_str = ' '.join(map(str, pose.flatten()))
            f.write(pose_str + '\n')


def load_downsampled_ply(ply_path, scale_to_meter=True):
    pcd = o3d.io.read_point_cloud(ply_path)

    if scale_to_meter:
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * 0.001)  # mm → m


    return pcd


def load_downsampled_ply_with_normals(ply_path, scale_to_meter=True):
    pcd = o3d.io.read_point_cloud(ply_path)
    if scale_to_meter:
        # pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * 1000)
        pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * 0.001) # mm → m
    # print(pcd.has_normals())
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
    # pcd.orient_normals_consistent_tangent_plane(k=10)

    # if orient_outward_from is not None:
    #     points = np.asarray(pcd.points)
    #     normals = np.asarray(pcd.normals)
    #     for i in range(len(normals)):
    #         to_point = points[i] - orient_outward_from
    #         if np.dot(normals[i], to_point) < 0:
    #             normals[i] = -normals[i]
    #     pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd


def create_centroid_marker(centroid, radius=0.005, color=[1, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(centroid)
    sphere.paint_uniform_color(color)
    return sphere


def visualize_trajectory(pcd, trajectory, centroid=None, frame_size=0.01, stride=1, show_path=True, show_arrows=True):
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size * 2.0)
    vis_elems = [pcd, origin_frame]
    
    if centroid is not None:
        centroid_marker = create_centroid_marker(centroid)
        vis_elems.append(centroid_marker)

    # Add coordinate frames at poses
    for i in range(0, len(trajectory), stride):
        pos, rot = trajectory[i]
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        frame.transform(T)
        vis_elems.append(frame)

    # Add lines (path) and arrows
    if show_path or show_arrows:
        for i in range(stride, len(trajectory), stride):
            p1, _ = trajectory[i - stride]
            p2, _ = trajectory[i]

            if show_path:
                line = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector([p1, p2]),
                    lines=o3d.utility.Vector2iVector([[0, 1]])
                )
                line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # red
                vis_elems.append(line)

            if show_arrows:
                direction = p2 - p1
                length = np.linalg.norm(direction)
                if length < 1e-6:
                    continue
                direction = direction / length
                arrow = o3d.geometry.TriangleMesh.create_arrow(
                    cylinder_radius=0.0005,
                    cone_radius=0.001,
                    cylinder_height=0.8 * length,
                    cone_height=0.2 * length
                )
                arrow.paint_uniform_color([0.1, 0.6, 1.0])  # light blue
                # Align arrow with direction vector
                z_axis = np.array([0, 0, 1])
                rot_axis = np.cross(z_axis, direction)
                angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
                if np.linalg.norm(rot_axis) > 1e-6:
                    rot_axis = rot_axis / np.linalg.norm(rot_axis)
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * angle)
                    arrow.rotate(R, center=np.zeros(3))
                arrow.translate(p1)
                vis_elems.append(arrow)

    if len(trajectory) > 1:
        start_pos, _ = trajectory[0]
        end_pos, _ = trajectory[-1]

        start_marker = o3d.geometry.TriangleMesh.create_sphere(radius=frame_size * 0.2)
        start_marker.translate(start_pos)
        start_marker.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow
        vis_elems.append(start_marker)

        end_marker = o3d.geometry.TriangleMesh.create_sphere(radius=frame_size * 0.2)
        end_marker.translate(end_pos)
        end_marker.paint_uniform_color([1.0, 0.5, 0.0])  # Orange
        vis_elems.append(end_marker)


    o3d.visualization.draw_geometries(vis_elems)


def save_trajectory_as_quaternions(trajectory, filename):
    with open(filename, 'w') as f:
        for pos, rot in trajectory:
            r = R.from_matrix(rot)
            qx, qy, qz, qw = r.as_quat()  # returns (x, y, z, w)
            x, y, z = pos
            line = f"{x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            f.write(line)

def connect_slices_with_safe_offset(pose_A, pose_B, offset=0.01):
    """Returns 3 poses:
    - retreat from end of A
    - move to start of B
    - approach to start of B
    """
    pos_A, rot_A = pose_A
    pos_B, rot_B = pose_B

    normal_A = rot_A[:, 2]
    normal_B = rot_B[:, 2]

    # Step 1: retreat from A
    retreat_pose = (pos_A + offset * normal_A, rot_A)

    # Step 2: move in free space (linear move between retreat and approach)
    move_pose = (pos_B + offset * normal_B, rot_B)

    # Step 3: approach surface B
    approach_pose = (pos_B, rot_B)

    return [retreat_pose, move_pose, approach_pose]


def compute_distance(pose1, pose2):
    return np.linalg.norm(pose1[0] - pose2[0])


def plan_slice_order(slice_data):
    visited = [False] * len(slice_data)
    order = []

    current_index = 0  # start with first slice arbitrarily
    visited[current_index] = True
    order.append(current_index)

    while len(order) < len(slice_data):
        _, current_traj = slice_data[current_index]
        current_end_pose = current_traj[-1]
        min_dist = float('inf')
        next_index = -1

        for i, (fname, traj) in enumerate(slice_data):
            if visited[i] or len(traj) == 0:
                continue
            dist = compute_distance(current_end_pose, traj[0])
            if dist < min_dist:
                min_dist = dist
                next_index = i

        if next_index == -1:
            break  # all visited or disconnected

        visited[next_index] = True
        order.append(next_index)
        current_index = next_index

    return order


def merge_ordered_trajectories(slice_data, order, transition_offset=0.01):
    merged = []
    for i, idx in enumerate(order):
        fname, traj = slice_data[idx]
        if i > 0:
            _, prev_traj = slice_data[order[i-1]]
            transition = connect_slices_with_safe_offset(prev_traj[-1], traj[0], offset=transition_offset)
            merged.extend(transition)
        merged.extend(traj)
    return merged


def reverse_the_trajectory(trajectory):
    reversed_traj = trajectory[::-1]
    return reversed_traj

# this is for individual slice processing
# if __name__ == "__main__":
#     ply_file = "pointnum7400235_1.1.ply"
#     pcd = load_downsampled_ply_with_normals(ply_file)

#     # stl_file = "select-00.stl"
#     # pcd = load_mesh_as_pointcloud_with_normals(stl_file)

#     trajectory = generate_zigzag_trajectory(
#         pcd,
#         point_spacing=0.005,   
#         line_spacing=0.01,     
#         offset=0.003, 
#         thickness=0.001        
#     )

#     visualize_trajectory(pcd, trajectory)
#     save_trajectory_as_quaternions(trajectory, "trajectory_piece1.txt")


# this is for a batch of slices processing (this is better)
if __name__ == "__main__":
    # get centroid of the component
    # full_ply_file = "pointnum739188.ply"
    full_ply_file = "select-00_downsampled1000w.ply"
    pcd_full = load_downsampled_ply(full_ply_file)
    pcd_full.points = o3d.utility.Vector3dVector(np.asarray(pcd_full.points) * 1000)
    points = np.asarray(pcd_full.points)

    centroid = np.mean(points, axis=0)
    
    bbox = pcd_full.get_axis_aligned_bounding_box()
    print("Bounding box center:", bbox.get_center())


    points = np.asarray(pcd_full.points)
    print("Centroid:", np.mean(points, axis=0))
    print("First few points:", points[:5])


    # min_bound = points.min(axis=0)
    # max_bound = points.max(axis=0)
    # extent = max_bound - min_bound

    # print("Bounding box min:", min_bound)
    # print("Bounding box max:", max_bound)
    # print("Extent (size):", extent)


    # #mm unit
    # ply_file_list= ["pointnum7400235_1.1.ply","pointnum7400235_1.2.ply","pointnum7400235_1.3.ply","pointnum7400235_2.1.ply","pointnum7400235_2.2.ply","pointnum7400235_3.1.ply","pointnum7400235_4.1.ply","pointnum7400235_4.2.ply"]
    
    # #mannual slice, m unit
    # ply_file_list= ["select-00_downsampled100w_1.1.ply","select-00_downsampled100w_1.2.ply","select-00_downsampled100w_1.3.ply","select-00_downsampled100w_2.1.ply","select-00_downsampled100w_2.2.ply","select-00_downsampled100w_3.1.ply","select-00_downsampled100w_3.2.ply","select-00_downsampled100w_4.1.ply","select-00_downsampled100w_4.2.ply","select-00_downsampled100w_5.ply"]

    #algo slice, m unit
    ply_file_list= ["CombinedFace_20250815_103713_downsampled50wmm.ply"]
    # ply_file_list= ["class_0_mask_1_surface.ply","class_0_mask_3_surface.ply","class_0_mask_7_surface.ply","class_1_mask_6_surface.ply","class_1_mask_9_surface.ply","class_2_mask_0_surface.ply","class_3_mask_4_surface.ply","class_4_mask_8_surface.ply","class_5_mask_2_surface.ply","class_6_mask_5_surface.ply"]
    # ply_file_list= ["class_0_mask_1_surface.ply","class_0_mask_3_surface.ply","class_0_mask_7_surface.ply","class_1_mask_6_surface.ply","class_1_mask_9_surface.ply","class_2_mask_0_surface.ply","class_3_mask_4_surface.ply"]
    # ply_file_list= ["class_6_mask_5_surface.ply"]
    
    # ply_file_list= ["class_5_mask_2_surface.ply","class_6_mask_5_surface.ply"]

    slice_data = []  # list of (ply_filename, trajectory)
    for ply_file in ply_file_list:
        pcd = load_downsampled_ply_with_normals(ply_file, scale_to_meter=True)
        points = np.asarray(pcd.points)

        traj, secondary_dir = generate_zigzag_trajectory(
            pcd, 
            point_spacing=0.01, 
            line_spacing=0.005, 
            offset=-0.01, 
            thickness=0.002
            )
        if len(traj) > 0:
            slice_data.append((ply_file, traj))

        # diagnose_zigzag_behavior(traj,secondary_dir)

        visualize_trajectory(pcd, traj, centroid=centroid)
        # visualize_trajectory(pcd, reverse_the_trajectory(traj), centroid=centroid)

    order = plan_slice_order(slice_data)
    full_trajectory = merge_ordered_trajectories(slice_data, order, transition_offset=-0.05)

    # Visualization
    mesh_file = "select-00.stl"
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    visualize_trajectory(mesh, full_trajectory, centroid=centroid)

    visualize_trajectory(pcd_full, full_trajectory, centroid=centroid)
    save_trajectory_as_quaternions(full_trajectory, "full_polishing_path_1_1_10_new.txt")
    reversed_traj = reverse_the_trajectory(full_trajectory) 
    save_trajectory_as_quaternions(reversed_traj,"full_polishing_path_1_1_10_new_reversed.txt")





# # this is for the whole stl processing (which is not reasonable)
# if __name__ == "__main__":
#     # get centroid of the component
#     full_ply_file = "select-00_downsampled100w.ply"
    
#     pcd = load_downsampled_ply(full_ply_file,scale_to_meter=False)
#     points = np.asarray(pcd.points)
#     centroid = np.mean(points, axis=0)
    

#     traj = generate_zigzag_trajectory(
#             pcd, 
#             point_spacing=0.005, 
#             line_spacing=0.01, 
#             offset=-0.01, 
#             thickness=0.001, 
#             centroid=centroid
#             )

#     visualize_trajectory(pcd, traj, centroid=centroid)
#     save_trajectory_as_quaternions(traj, "full_polishing_path_stl.txt")


 