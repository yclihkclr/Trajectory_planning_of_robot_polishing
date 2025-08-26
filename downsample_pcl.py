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
    target_num_points=1000000,
    scale_to_meters=True
):
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.compute_vertex_normals()

    # SCALE: Convert mm → meters
    if scale_to_meters:
        mesh.scale(0.001, center=mesh.get_center())

    # Downsample
    pcd = mesh.sample_points_poisson_disk(target_num_points)

    # Save
    o3d.io.write_point_cloud(ply_output_path, pcd)
    print(f"Saved downsampled point cloud ({len(pcd.points)} points) to: {ply_output_path}")
    return pcd

# Example usage:
#for this "CombinedFace_20250815_103713.ply" we remain the unit as mm
stl_file = "CombinedFace_20250815_103713.ply"
ply_file = "CombinedFace_20250815_103713_downsampled50wmm.ply"
convert_stl_to_ply_downsampled(stl_file, ply_file, target_num_points=100000, scale_to_meters=False)
