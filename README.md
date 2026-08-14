# Launch robot driver (SIM)
ros2 launch ur_robot_driver ur_platform_control.launch.py ur_type:=ur5e robot_ip:=yyy.yyy.yyy.yyy use_fake_hardware:=true launch_rviz:=false initial_joint_controller:=joint_trajectory_controller

# Launch moveit
ros2 launch ur_moveit_config ur_platform_moveit.launch.py ur_type:=ur5e launch_rviz:=true

# Launch sampler
ros2 launch reach_pkg generic_broadcaster.launch.py

# Launch optimizer
ros2 run moveit_cpp_demo ik_checker   --ros-args   --params-file install/ur_moveit_config/share/ur_moveit_config/config/kinematics.yaml

# Just view the robot
ros2 launch ur_description view_ur.launch.py ur_type:=ur5e

# Generate heatmap
python3 src/reach_pkg/reach_pkg/heatmap_v2.py