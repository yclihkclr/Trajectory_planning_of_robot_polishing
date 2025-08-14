import open3d as o3d

# def convert_stl_to_ply_downsampled(
#     stl_path, 
#     ply_output_path, 
#     target_num_points=10000000
# ):
#     # Load the mesh
#     mesh = o3d.io.read_triangle_mesh(stl_path)
#     mesh.compute_vertex_normals()
#     pcd = mesh.sample_points_poisson_disk(target_num_points)

#     # Save to PLY
#     o3d.io.write_point_cloud(ply_output_path, pcd)
#     print(f"Saved downsampled point cloud ({len(pcd.points)} points) to: {ply_output_path}")
#     return pcd



def convert_stl_to_ply_downsampled(
    stl_path,
    ply_output_path,
    target_num_points=1000000
):
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.compute_vertex_normals()

    # SCALE: Convert mm → meters
    mesh.scale(0.001, center=mesh.get_center())

    # Downsample
    pcd = mesh.sample_points_poisson_disk(target_num_points)

    # Save
    o3d.io.write_point_cloud(ply_output_path, pcd)
    print(f"Saved downsampled point cloud ({len(pcd.points)} points) to: {ply_output_path}")
    return pcd

# Example usage:
stl_file = "select-00.stl"
ply_file = "select-00_downsampled100w_meter.ply"
convert_stl_to_ply_downsampled(stl_file, ply_file, target_num_points=1000000)
