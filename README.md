
# Trajectory planning of robot polishing
This repository provides Python scripts for trajectroy planning of robot polishing given point clouds and mesh models. The tools support mesh downsampling, trajectory generation, and trajectory coordinate frame transformation, ideal for the Robot polishing applications.

## Dependencies
To run the scripts, you need:

- Python 3.6+
- open3d
- numpy
- scipy

No install needed, easy to deploy

# Usage

```
git clone https://github.com/yclihkclr/Trajectory_planning_of_robot_polishing
```
1. Downsample the original mesh CAD file (.stl) to pointcloud (.ply) for further segementation.
The parameter of stl_file and ply_file in downsample_pcl.py should be changed as your needs accordingly.

```
cd Trajectory_planning_of_robot_polishing
python downsample_pcl.py
```

2. After we ger the pointcloud fragements from segementation as "class_0_mask_1_surface.ply" etc. We can run the TPRP to plan the polihsing trajectroy with respect to mesh frame. Where the parameters (point_spacing, line_spacing, offset, thickness) should be defined as your nees accordingly.

```
python polishing_traj_planning.py 
```

3. We need to transform the planned trajectry (such as full_polishing_path_1_1_7) from mesh frame to robot base frame ( full_polishing_path_1_1_7_transformed) given the relationship between these two frame.
```
python transform_to_base.py
```

Then the final planned trajectory can be executed via robot.

# Future updates
Currently the generated trajectory result is under testing in simualtion and the upgrades will be updated soon.