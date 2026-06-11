from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    vineyard_length = 0.8
    cordon_height = 0.75 #From the surface on which the robot is standing to the center of the cordon

    pruning_positions_service = Node(
        package='reach_pkg',
        executable='pruning_positions_service',
        name='pruning_positions_service',
        output='screen'
    )

    orientation_service = Node(
        package='reach_pkg',
        executable='orientation_cone_service',
        name='orientation_cone_service',
        output='screen'
    )

    br1 = Node(
        package='reach_pkg',
        executable='generic_br',
        name='prune_pose_broadcaster',
        output='screen',
        parameters=[{
            'n': 100,
            'tf_prefix': 'pruning_pose',
            'parent_frame': 'vineyard_base',
            'points_service': 'sample_points',
            'orientations_service': 'sample_orientations',
            'max_angles': [0.87, 0.87, 0.87],
            'pos_center1': [-vineyard_length/2, 0.0, cordon_height+0.1], 
            'pos_center2': [vineyard_length/2, 0.0, cordon_height+0.1],
            'pos_sigma': [0.2, 0.1, 0.2, 0.15],
            'pos_boundaries': [-(vineyard_length/2)*1.22, (vineyard_length/2)*1.22, -0.25, 0.25, cordon_height, cordon_height+0.5]
        }]
    )

    br2 = Node(
        package='reach_pkg',
        executable='generic_br',
        name='scan_pose_broadcaster',
        output='screen',
        parameters=[{
            'n': 100,
            'tf_prefix': 'scaning_pose',
            'parent_frame': 'vineyard_base',
            'points_service': 'sample_points',
            'orientations_service': 'sample_orientations',
            'max_angles': [0.1, 0.1, 0.1],
            'pos_center1': [-vineyard_length/2, 0.2, cordon_height+0.1], 
            'pos_center2': [vineyard_length/2, 0.2, cordon_height+0.1],
            'pos_sigma': [0.2, 0.2, 0.3, 0.2],
            'pos_boundaries': [-(vineyard_length/2)*1.22, (vineyard_length/2)*1.22, -0.05, 0.3, cordon_height, cordon_height+0.6]
        }]
    )

    br3 = Node(
        package='reach_pkg',
        executable='generic_br',
        name='grasp_pose_broadcaster',
        output='screen',
        parameters=[{
            'n': 100,
            'tf_prefix': 'grasping_pose',
            'parent_frame': 'vineyard_base',
            'points_service': 'sample_points',
            'orientations_service': 'sample_orientations',
            'max_angles': [0.1, 0.1, 0.1],
            'pos_center1': [-vineyard_length/2, 0.0, cordon_height + 0.25], 
            'pos_center2': [vineyard_length/2, 0.0, cordon_height + 0.25],
            'pos_sigma': [0.2, 0.2, 0.2, 0.15],
            'pos_boundaries': [-(vineyard_length/2)*1.22, (vineyard_length/2)*1.22, -0.25, 0.25, cordon_height+0.1, cordon_height+0.6]
        }]
    )

    collisions_prob = Node(
        package='reach_pkg',
        executable='collisions_cost_server',
        name='collisions_cost_server',
        output='screen',
        parameters=[{
            'cordon_height': cordon_height,
            'cordon_length': vineyard_length,
        }]
    )

    return LaunchDescription([
        pruning_positions_service,
        orientation_service,
        br1,
        br2,
        br3,
        collisions_prob
    ])